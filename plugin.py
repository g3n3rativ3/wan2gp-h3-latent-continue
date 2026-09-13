"""Controls injected into the existing Media Generator; no new application tab."""
import copy
import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
from .integration import KEY, make_generation_wrapper, make_save_wrapper, make_record_wrapper, make_prepare_wrapper, options, state_mapping
from .integration import remember_options, make_settings_wrapper, make_queue_wrapper


class H3LatentPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = 'H3 Latent Continue'
        self.version = '0.2.3'
        self.description = 'Single-phase H3 comparative latent continuation in the existing video form.'
        self._bridge_installed = False

    def setup_ui(self):
        for name in ('state','plugin_data','video_source','image_prompt_type','model_choice'):
            self.request_component(name)
        for name in ('generate_media','save_video','record_file_metadata','prepare_inputs_dict','get_model_settings','add_video_task','get_base_model_type','get_state_model_type'):
            self.request_global(name)

    def post_ui_setup(self, components):
        required = ('state','plugin_data','video_source','image_prompt_type')
        missing = [k for k in required if k not in components]
        if missing: raise RuntimeError('H3 Latent UI: missing native components: '+', '.join(missing))
        if not self._bridge_installed:
            self.set_global('add_video_task', make_queue_wrapper(self.add_video_task,self.get_base_model_type,self.get_state_model_type))
            self.set_global('get_model_settings', make_settings_wrapper(self.get_model_settings,self.get_base_model_type))
            self.set_global('prepare_inputs_dict', make_prepare_wrapper(self.prepare_inputs_dict,self.get_base_model_type,self.get_state_model_type))
            self.set_global('generate_media', make_generation_wrapper(self.generate_media,self.get_base_model_type))
            self.set_global('save_video', make_save_wrapper(self.save_video))
            self.set_global('record_file_metadata', make_record_wrapper(self.record_file_metadata))
            self._bridge_installed = True
        state, data, mode = (components[k] for k in ('state','plugin_data','image_prompt_type'))

        def is_latent_continue(st):
            st = state_mapping(st)
            model = self.get_state_model_type(st) if st and ('model_type' in st or 'edit_model_type' in st) else ''
            return str(self.get_base_model_type(model)).startswith('h3_latent_')

        def create():
            initial_state = getattr(state,'value',{}) or {}
            active = is_latent_continue(initial_state)
            initial_options = options(getattr(data,'value',{}) or {})
            with gr.Column(visible=active) as panel:
                save = gr.Checkbox(value=initial_options['save'], label='Save latent checkpoint next to output video')
                with gr.Accordion('Continue with latent',open=False,
                                  visible='V' in str(getattr(mode,'value','') or '')) as continuation:
                    use = gr.Checkbox(value=initial_options['continue'],label='Enable Continue with latent (experimental)')
                    latent = gr.File(label='Matching H3 latent checkpoint (.safetensors)',
                                     file_types=['.safetensors'],type='filepath',
                                     value=initial_options['latent_path'] or None)
                    join_mode = gr.Dropdown(choices=[('Reference (0.1.7)', 'baseline'),
                                                     ('Longer video/audio context', 'context'),
                                                     ('Longer context + frozen audio prefix', 'audio_prefix')],
                                            value=initial_options.get('join_mode','baseline'), label='Join experiment')
                    video_context = gr.Dropdown(choices=[18,35,52],value=initial_options.get('video_context_frames',35),
                                                label='Video latent context (frames)')
                    audio_context = gr.Dropdown(choices=[0.5,1.0,2.0],value=initial_options.get('audio_context_seconds',1.0),
                                                label='Audio latent context (seconds)')
                    gr.Markdown('For the frozen audio prefix select **No Skipping**. Context is limited to the saved segment; '
                                'these experiments do not promise a seamless join. Initial clip: use Reference.')
                    gr.Markdown('Keep the source video in the usual **Video to Continue** field. '
                                'Single phase, same resolution/FPS. Original video + matching checkpoint required.')
            def update_data(saved, enabled, path, existing, session_state, experiment='baseline', video_frames=35, audio_seconds=1.0):
                result = copy.deepcopy(state_mapping(existing))
                result[KEY] = {'save':bool(saved),'continue':bool(enabled),'latent_path':path or '',
                               'join_mode':experiment,'video_context_frames':video_frames,'audio_context_seconds':audio_seconds}
                model = self.get_state_model_type(state_mapping(session_state))
                remember_options(session_state, model, result)
                return result
            gr.on(triggers=[save.input,use.input,latent.upload,latent.clear,join_mode.input,video_context.input,audio_context.input],fn=update_data,
                  inputs=[save,use,latent,data,state,join_mode,video_context,audio_context],outputs=[data],queue=False)
            def visibility(st, flags):
                supported = is_latent_continue(st)
                return gr.update(visible=supported), gr.update(visible=supported and 'V' in str(flags or ''))
            gr.on(triggers=[mode.change,state.change],fn=visibility,inputs=[state,mode],outputs=[panel,continuation],queue=False)
            # Model selection may change without rebuilding the complete form.
            if 'model_choice' in components:
                components['model_choice'].change(fn=visibility,inputs=[state,mode],outputs=[panel,continuation],queue=False)
            def restore(existing):
                if KEY not in state_mapping(existing):
                    return (gr.skip(),)*6
                opts = options(existing)
                return (opts['save'],opts['continue'],opts['latent_path'] or None,
                        opts.get('join_mode','baseline'),opts.get('video_context_frames',35),opts.get('audio_context_seconds',1.0))
            data.change(fn=restore,inputs=[data],outputs=[save,use,latent,join_mode,video_context,audio_context],queue=False)
            from gradio.context import get_blocks_context
            from .form_bridge import bind_form_events
            count = bind_form_events(get_blocks_context(), data, [save,use,latent,join_mode,video_context,audio_context], self.get_state_model_type)
            print(f'[H3 Latent] Inline controls connected to {count} native form event(s).')
            return panel
        self.insert_after('video_source',create)
