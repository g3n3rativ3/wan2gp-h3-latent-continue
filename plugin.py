"""Controls injected into the existing Media Generator; no new application tab."""
import copy
import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
from .integration import KEY, make_generation_wrapper, make_save_wrapper, make_record_wrapper, make_prepare_wrapper, options, state_mapping
from .native_bridge import SUPPORTED, install_loader
from .integration import make_validation_wrapper, make_runtime_task_wrapper, make_edit_wrapper, remember_options, make_settings_wrapper, make_queue_wrapper


class H3LatentPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = 'H3 Latent Continue'
        self.version = '0.6.5'
        self.description = 'One- or two-phase H3 comparative latent continuation in the existing video form.'
        self._bridge_installed = False

    def setup_ui(self):
        install_loader()
        for name in ('state','plugin_data','video_source','image_prompt_type','model_choice'):
            self.request_component(name)
        for name in ('generate_media','save_video','record_file_metadata','prepare_inputs_dict','get_model_settings','add_video_task','get_base_model_type','get_state_model_type','validate_settings','_build_runtime_task','edit_task_in_queue','server_config'):
            self.request_global(name)

    def post_ui_setup(self, components):
        required = ('state','plugin_data','video_source','image_prompt_type')
        missing = [k for k in required if k not in components]
        if missing: raise RuntimeError('H3 Latent UI: missing native components: '+', '.join(missing))
        if not self._bridge_installed:
            self.set_global('validate_settings', make_validation_wrapper(self.validate_settings))
            self.set_global('_build_runtime_task', make_runtime_task_wrapper(self._build_runtime_task))
            self.set_global('edit_task_in_queue', make_edit_wrapper(self.edit_task_in_queue))
            self.set_global('add_video_task', make_queue_wrapper(self.add_video_task,self.get_base_model_type,self.get_state_model_type))
            self.set_global('get_model_settings', make_settings_wrapper(self.get_model_settings,self.get_base_model_type))
            self.set_global('prepare_inputs_dict', make_prepare_wrapper(self.prepare_inputs_dict,self.get_base_model_type,self.get_state_model_type))
            self.set_global('generate_media', make_generation_wrapper(self.generate_media,self.get_base_model_type))
            self.set_global('save_video', make_save_wrapper(self.save_video))
            self.set_global('record_file_metadata', make_record_wrapper(self.record_file_metadata, lambda: getattr(self, 'server_config', None)))
            self._bridge_installed = True
        state, data, mode = (components[k] for k in ('state','plugin_data','image_prompt_type'))

        def is_latent_continue(st):
            st = state_mapping(st)
            model = self.get_state_model_type(st) if st and ('model_type' in st or 'edit_model_type' in st) else ''
            return self.get_base_model_type(model) in SUPPORTED

        def create():
            initial_state = getattr(state,'value',{}) or {}
            active = is_latent_continue(initial_state)
            initial_options = options(getattr(data,'value',{}) or {})
            with gr.Column(visible=active) as panel:
                refresh_signal = gr.Textbox(value='', visible=False)
                save = gr.Checkbox(value=initial_options['save'], label='Save latent checkpoint next to output video')
                noise1818 = gr.Checkbox(value=initial_options.get('audio_noise_1818', False),
                                        label='Audio noise as WanGP 18.18 (same seed reproduces renders made before 18.19; applies to every H3 render when checked: Text Only, Start with Image, Continue Video)')
                with gr.Accordion('Continue with latent',open=False,
                                  visible='V' in str(getattr(mode,'value','') or '')) as continuation:
                    use = gr.Checkbox(value=initial_options['continue'],label='Enable Continue with latent (experimental)')
                    latent = gr.File(label='Matching H3 latent checkpoint (.safetensors)',
                                     file_types=['.safetensors'],type='filepath',
                                     value=initial_options['latent_path'] or None)
                    join_mode = gr.Dropdown(choices=[('Reference (minimum 18-frame context)', 'baseline'),
                                                     ('Longer video/audio context', 'context'),
                                                     ('Longer context + frozen audio prefix', 'audio_prefix'),
                                                     ('Exact join: last frames pinned to saved latents + frozen audio', 'pinned')],
                                            value=initial_options.get('join_mode','baseline'), label='Join experiment')
                    video_context = gr.Dropdown(choices=[18,35,52],value=initial_options.get('video_context_frames',35),
                                                label='Video latent context (frames)')
                    audio_context = gr.Dropdown(choices=[0.5,1.0,2.0],value=initial_options.get('audio_context_seconds',1.0),
                                                label='Audio latent context (seconds)')
                    seamless = gr.Checkbox(value=initial_options.get('seamless_end', True),
                                           label='Exact join: seamless end (re-decode the last source frames; needs WanGP Windows Frames Overlap >= 35)')
                    phase1_final = gr.Checkbox(value=initial_options.get('phase1_from_final', True),
                                               label='Exact join, two phases: phase 1 follows the final video (re-encode the pinned frames at phase-1 size)')
                    gr.Markdown('**Exact join** starts the new window on a VAE chunk boundary: its first frames are the saved '
                                'latents of the last source frames, kept fixed while the rest is generated, so the new frames '
                                'continue the same latent sequence. The number of pinned frames follows WanGP\'s '
                                '**Windows Frames Overlap** slider (Sliding Window section): 18 -> 5 pinned frames, '
                                '**35 -> 22 pinned frames + seamless end** (recommended), 52 -> 39. *Video latent context* above '
                                'is a different setting (extra history given to the model). Audio context must cover the pinned '
                                'frames; source length 5+17n frames.')
                    gr.Markdown('For the frozen audio prefix select **No Skipping**. Context is limited to the saved segment; '
                                'these experiments do not promise a seamless join. Initial clip: use Reference. '
                                'Reference retains at least 18 source frames when available, independently of assembly overlap.')
                    gr.Markdown('Keep the source video in the usual **Video to Continue** field. '
                                'One or two phases (untiled phase 2), same resolution/FPS. Original video + matching checkpoint required. '
                                'Two-phase continuation needs a checkpoint saved from a two-phase render. '
                                'With Continue Video, the new part alone is also saved as *_part2* next to the complete video.')
            def update_data(saved, enabled, path, existing, session_state, experiment='baseline', video_frames=35, audio_seconds=1.0, seamless_end=True, phase1_from_final=True, audio_noise_1818=False):
                result = copy.deepcopy(state_mapping(existing))
                result[KEY] = {'save':bool(saved),'continue':bool(enabled),'latent_path':path or '',
                               'join_mode':experiment,'video_context_frames':video_frames,'audio_context_seconds':audio_seconds,
                               'seamless_end':bool(seamless_end),'phase1_from_final':bool(phase1_from_final),
                               'audio_noise_1818':bool(audio_noise_1818)}
                model = self.get_state_model_type(state_mapping(session_state))
                remember_options(session_state, model, result)
                return result
            gr.on(triggers=[save.input,use.input,latent.upload,latent.clear,join_mode.input,video_context.input,audio_context.input,seamless.input,phase1_final.input,noise1818.input],fn=update_data,
                  inputs=[save,use,latent,data,state,join_mode,video_context,audio_context,seamless,phase1_final,noise1818],outputs=[data],queue=False)
            def visibility(st, flags):
                supported = is_latent_continue(st)
                return gr.update(visible=supported), gr.update(visible=supported and 'V' in str(flags or ''))
            # Native queues reference their owning state. A state.change listener
            # makes Gradio deep-hash that cyclic graph on import and recurse.
            # Reading state as an input is safe; subscribe to scalar controls only.
            gr.on(triggers=[mode.change],fn=visibility,inputs=[state,mode],outputs=[panel,continuation],queue=False)
            # Model selection may change without rebuilding the complete form.
            if 'model_choice' in components:
                components['model_choice'].change(fn=visibility,inputs=[state,mode],outputs=[panel,continuation],queue=False)
            def restore(existing):
                if KEY not in state_mapping(existing):
                    return (gr.skip(),)*9
                opts = options(existing)
                return (opts['save'],opts['continue'],opts['latent_path'] or None,
                        opts.get('join_mode','baseline'),opts.get('video_context_frames',35),opts.get('audio_context_seconds',1.0),
                        opts.get('seamless_end',True),opts.get('phase1_from_final',True),opts.get('audio_noise_1818',False))
            # Watching shared State asks Gradio to recursively hash foreign data.
            # Native writers emit only a scalar notification; restoration reads
            # the payload without registering a change listener on either State.
            refresh_signal.change(fn=restore,inputs=[data],outputs=[save,use,latent,join_mode,video_context,audio_context,seamless,phase1_final,noise1818],queue=False)
            from gradio.context import get_blocks_context
            from .form_bridge import bind_form_events
            count = bind_form_events(get_blocks_context(), data, [save,use,latent,join_mode,video_context,audio_context,seamless,phase1_final,noise1818], self.get_state_model_type)
            from .refresh_bridge import bind_refresh_events
            bind_refresh_events(get_blocks_context(), data, refresh_signal, exclude=(update_data,))
            print(f'[H3 Latent] Inline controls connected to {count} native form event(s).')
            return panel
        self.insert_after('video_source',create)
