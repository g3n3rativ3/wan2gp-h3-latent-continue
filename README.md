# H3 Latent Continue for Wan2GP

Version **0.6.0**. Tested with Wan2GP **18.17 and 18.19**. MiniMax H3 latent continuation for Wan2GP. **One or two generation phases** (untiled phase 2),
with or without the **Audio Refinement Extra Phase**.

## Version 0.6.5: WanGP 19.02

Compatible with WanGP 19.02 (and still 18.17 to 18.19). The new Pika H3 Flex sampler works with every latent option,
including Exact join; "Audio noise as WanGP 18.18" leaves Pika's own noise unchanged.

## Version 0.6.3: exact join no longer cut at the end

H3 only generates 5+17n frames. With Exact join the window starts on a VAE chunk boundary, so the frames left to
generate are usually not such a length (e.g. 349 instead of 362). The model used to plan the rounded-up length and
the end was cut (speech interrupted). The window is now rounded down: the scene ends where the video ends, and the
new part is a few frames shorter than requested (4 frames with sources of 5+17n frames, at most 16).

## Version 0.6.1: audio noise as WanGP 18.18

WanGP 18.19 generates 5 extra audio latents past the last frame (the "audio tail"). The seeded audio noise is drawn
as one block of channel-major rows (all of channel 0, then all of channel 1), so with a longer audio the second
channel receives different noise: the same seed no longer reproduces a render made with WanGP 18.18 or earlier
(the video noise is unchanged; speech, pauses and who speaks can change, and the image follows).

**Audio noise as WanGP 18.18** (unchecked by default, under *Save latent checkpoint*) draws the seeded audio noise
for the audio without its tail, exactly as before 18.19, and the tail noise from a separate generator. The same
rule applies to the Audio Refinement re-noise and to er_sde step noise. The tail is still generated and saved.
Applies to every H3 render when checked (Text Only, Start with Image, Continue Video, sliding windows), alone or with the latent options; recorded in the checkpoint settings (`audio_noise`) when a checkpoint is saved. Expect a very
close render, not a bit-identical one: the 5 tail latents are still in the sequence, and WanGP 18.19 also
computes LoRA deltas in the weight dtype.

## Version 0.6.0: two phases, phase 1 follows the final video

Reported: in two phases the people are stable but walls and ceiling "morph" during the first second of
part 2. Measured on the supplied clip: the ceiling drifts away from the last source frame by 1.8 -> 5.0 ->
5.5 (after 6 / 24 / 48 frames) against 1.3-1.8 inside part 1 over the same distances, then settles; the
4-frame flicker pattern is the same in both parts (native two-phase behaviour).

Cause: phase 1 continued the source's *phase-1 draft* latents. The final video is the phase-2 result, which
redraws details (panel shapes) on top of that draft; the saved draft differs from the reduced final by more
than one latent step of motion. The first new frames follow the pinned final frames, the next ones go back
to the draft's structure: the morph.

**Exact join, two phases: phase 1 follows the final video** (checked by default) does what native H3 does
for phase 1: the pinned frames are the final source frames (overlap pixels) encoded at the phase-1 size by
the native VAE, on the window's own chunk grid; the phase-1 context comes from the final latents reduced to
phase 1. Phase 2 still pins the exact full-resolution latents. A checkpoint saved in one phase can now be
continued in two phases with Exact join. Untick to use the saved phase-1 draft (0.5.x behaviour, no encode).
tests/vae_seam_check.py also checks on WanGP's encoder code that the re-encoded pinned latents do not depend
on the chunk padding.

## Version 0.5.3: overlap wording

The number of pinned frames and the seamless end depend on WanGP's **Windows Frames Overlap** slider
(Sliding Window section; H3 steps 1, 18, 35, 52), not on the plugin's *Video latent context*. The panel text,
the checkbox label and the console lines now name that slider (with an 18 overlap the console says the seamless
end needs 35).

## Version 0.5.2: Exact join seamless end

Measured after Exact join (35-frame overlap): seam 2.3x the normal motion (was 7.7x), no colour step,
residual spread evenly over the image. The H3 decoder decodes 7-latent windows (a 17-frame chunk plus two
look-ahead latents) and cross-fades 5 frames. The source's last 5 frames were decoded without their future;
the new window decodes the same latents with the real future.

**Exact join: seamless end** (checked by default, acts only with Exact join) replaces the source's last
frames in the complete video by the new window's decode of the same latents. On WanGP's real decoder code
(tests/vae_seam_check.py) frames 226..238 of a 243-frame segment are decoded identically by the source and the
exact-join window, and only 239..242 change: the switch happens between identical frames.

- Needs the pin to include the full chunk before the last partial one: overlap >= 35 for 5+17n-frame sources
  (otherwise a console line says it is skipped).
- Before replacing, the export checks the switch frame against the source (codec noise only) and the
  replaced frames (small decoder differences); otherwise the source end is kept, with a console message.
- The complete video then differs from the source file on those last frames; `_part2` starts that many
  frames earlier so that (source minus its last frames) + `_part2` = complete video. The audio is unchanged.
- The checkpoint is bound to the complete video as exported (with the replaced frames).

## Version 0.5.1: audio references with frozen audio

Exact join and the frozen audio prefix refused any audio guide. In Ref2VA, audio guides are voice
*references* (separate condition rows, like audio RefMods): they are now accepted in every join mode.
The only real conflict is an audio guide that drives the generated soundtrack itself (FL2VA audio
used as soundtrack); it is refused with an explicit message when the frozen audio is placed.

## Version 0.5.0: Exact join

### Why the previous modes show a visible seam

Measured on a reported clip (243 + 242 frames, context mode, 52 frames): the frame difference
at the seam is 7.7 times the normal frame-to-frame motion, with no global displacement (0.5 px)
and an exposure step of about 2 levels; hair, faces and background textures are all re-drawn.
The composition is followed but the first new frame is not the continuation of the last one.

- In the previous modes **no target frame is pinned**. The whole new window starts from noise;
  the saved latents are only attention conditions. A low-step (Turbo) render reinterprets them,
  exactly like H3 reinterprets a start image (frame 0 is generated, not written).
- These conditions are sequence latents presented as image keyframes, placed *before* frame 0.
  With 52 frames of context they reach about 85 RoPE positions before the target start, in the
  time range of the prompt tokens: out of what H3 was trained on.
- WanGP notes the same limit for native continuation ("Native H3 history alone loses that
  appearance constraint") and pins the overlap inside the target, but only for ControlNet models.

### What Exact join does

Choose **Exact join: last frames pinned to saved latents + frozen audio** in *Join experiment*.

- The new window starts on a VAE chunk boundary of the saved segment. The H3 VAE encodes each
  17-frame chunk on its own ([1, 4, 4, 4, 4] frames per latent), so the window's first latents
  *are* the saved latents of the last source frames: 5 frames with an 18-frame overlap, 22 with
  35, 39 with 52.
- Those latents are kept fixed while the rest is generated, with WanGP's own masked denoising
  (prediction replaced on the pinned frames, exact at the last step), in one and two phases
  (phase-1 latents for phase 1). The new frames continue the same latent sequence.
- The soundtrack of the pinned frames is frozen too (frozen audio prefix moved with the window),
  and the audio is decoded jointly with it.
- Earlier history (video context 18/35/52) is still given as context before the pinned frames.
- WanGP still drops the overlap, so the pinned frames are not repeated in the output.

Requirements: Continue Video with a checkpoint, overlap >= 5 frames (the source segment must end
on a latent boundary, which WanGP's 5+17n lengths do), audio context covering the pinned frames
(1 s for up to 22 pinned frames), No Skipping. A few frames more than the window are generated
and cut (the shifted window is rounded up to the H3 frame grid). Checkpoints of earlier versions
work unchanged.

Known limit: the H3 decoder looks two latents ahead. The last source frames were decoded without
the future; the new frames are decoded with the exact past. A small residual step at the very
seam frame is possible; this needs a GPU comparison.

## Version 0.4.2: Wan2GP 18.19

Wan2GP 18.19 generates 5 audio latents past the last video frame (`H3_AUDIO_TAIL_LATENTS`) and keeps
about 100 ms of audio after the last frame in the saved video (`audio_tail_ms`). The audio-length line
of `generate` changed, so hook installation failed again ("no reachable native method contains the
required latent insertion points"), in one and two phases.

- The audio-length edit now has one variant per WanGP version (18.19 with the audio tail, 18.17
  without): the first variant that matches exactly once is used. The extended-context modes add the
  same tail as native.
- Checkpoints store the audio latents including the tail; the join conditions still stop at the last
  delivered frame, so continuations are unchanged.
- `_part2` keeps the audio tail exactly like the complete video (no cut at the last frame).
- 18.19 disables H3's native audio repair at joins; the plugin's neutralisation simply has nothing to do.

## Version 0.4.1: phase-1 size changed since the source render

Phase 1 of a two-phase render is generated at the target divided by
`H3_TWO_PHASE_SCALE`, rounded to 32 pixels. When that size differs between the
source render and the continuation (scale changed by a setting or another plugin,
or a resolution whose rounding moves), 0.4.0 stopped with "no saved latents match
the WxH generation stage". The saved phase-1 latents are now resampled spatially
to the new phase-1 size, with a console message. This is approximate; phase 2
still uses the exact full-resolution latents. The checkpoint manifest records it
(`effective_context.phase1_resampled_from`). Tracebacks inside the latent version
of `generate` now report the real `pipeline.py` lines.

## Version 0.4.0: Audio Refinement Extra Phase

The extra audio refinement phase is now allowed: WanGP's native one, and the
**H3 Audio Refinement Controls** plugin (`wan2gp-h3-audio-refinement`, steps,
denoising strength, Keep Audio References).

- Any `audio_refinement` value is accepted (`none`, `enabled`, `disabled:6:0.5`,
  `enabled:1:0.1:audio_refs`). The refusal of 0.3.9 also blocked the refinement
  plugin's *disabled* value.
- **Join kept during the refinement.** Native refinement re-noises the whole
  soundtrack with no condition. During a latent continuation the plugin keeps the
  join's saved-latent audio conditions. In the frozen audio-prefix mode it also
  restores the frozen prefix after the re-noise and keeps its time offset (without
  it, the audio would be shifted by the pre-roll during the refinement). Renders
  without a latent join refine exactly as WanGP / the refinement plugin do.
- The checkpoint stores the refined audio (captured after the refinement).
- **Combined with H3 Audio Refinement Controls.** That plugin rewrites the body of
  `generate` from the WanGP source (exact schedule, kept audio references). The
  latent version, also built from the source, now applies the same rewrites, so
  they stay active with the latent options on. The installed body is checked by
  bytecode against "WanGP source + those rewrites" before combining.
- If another plugin rewrote `generate` in place in an unknown way, a console
  warning says its change is not applied with latent options on (wrappers, such as
  RefMods, are not concerned).
- Combined with two phases: the refinement runs after phase 2, as native.

## Version 0.3.9: second part saved alone

With **Continue Video**, when Save latent checkpoint or Continue with latent is
checked, the plugin also writes the part created by the render alone, next to
the complete video: `<name>_part2.mp4` (same container as the output).

- Frames: only the new frames, i.e. the complete video minus the source video.
  The overlap frames regenerated by the window are excluded, as in the complete
  video, so source video + second part = complete video.
- Video: encoded by WanGP's own `save_video` with the same codec settings as the
  complete video.
- Audio: the matching segment of the complete video's soundtrack, re-encoded with
  WanGP's audio output codec (sample-accurate start).
- Start frame: found from the exact fingerprints of the exported tail, so a tail
  trimmed by WanGP to match the requested length is handled.
- The complete video and the latent checkpoint are unchanged and remain the
  files to continue from. If the second part cannot be written, a warning is
  printed and the render still succeeds.
- The second part is a plain file: it is not added to the WanGP gallery.

## Version 0.3.8: two-phase support

A two-phase H3 render generates at half resolution (phase 1), upscales the
latents, then refines at full resolution (phase 2) with the phase-1 audio frozen.
The two phases need conditioning video at two different resolutions.

- **Save (two phases):** the checkpoint also stores the phase-1 video latents
  (`video_phase1`, captured just before the latent upscaler) and their size in
  the manifest (`guide_phases`, `phase1`). The final latents are unchanged.
- **Continue (two phases):** phase 1 is conditioned on the saved phase-1 latents,
  phase 2 on the saved full-resolution latents. Both are exact, with no VAE
  re-encode. The saved last frame is shown to the prompt encoder at each phase's
  size, as native does. Audio conditions are built once in phase 1 and stay
  frozen in phase 2, as native. All three join modes are supported.
- A two-phase checkpoint can also be continued with one phase (full-resolution
  latents). A **single-phase** checkpoint cannot be continued with two phases:
  it has no phase-1 latents, and the plugin refuses it with an explicit message
  rather than re-encoding or downscaling latents.
- Still unsupported: tiled phase 2 (`~`), more than two phases, extra audio
  refinement phase, PDD, ControlNet-Union.
- Single-phase checkpoints keep the exact same format (readable by 0.3.7).

## Version 0.3.7: Wan2GP 18.17 audio-timeline adaptation

Wan2GP 18.x added a native audio time offset to H3 (`audio_time_offset`), used to
repair the audio at a continuation join (H3 now declares `audio_overlap_repair_ms`
= 25). The audio length, the audio decode length and the transformer layout cache
key now include this offset. Three of the plugin's anchors no longer matched, so
hook installation failed with "no reachable native method contains the required
latent insertion points", even for a plain *save latent* render.

- The audio timeline, audio decode and layout edits are re-anchored on the new
  native lines. The native offset is kept in every formula, and the plugin's own
  offset is added to the layout cache key next to it.
- During an active latent continuation, a non-zero native repair offset is
  reset to 0 and logged. The join is conditioned on the saved latent audio, so
  stacking both offsets would place the audio wrongly. With the default 25 ms
  repair at 24 FPS the native offset is already 0 (repair stays inside the last
  overlap frame). It only applies when the repair is longer than one frame.
  Native jobs (latent options off) keep the native repair unchanged.
- Native 18.x decodes video straight to uint8 with WGP truncation; the tail
  fingerprints already accept uint8 frames, so saved video and checkpoint match.

Checkpoint format is unchanged; 0.3.6 checkpoints continue to load. This release
targets Wan2GP 18.17; keep 0.3.6 for Wan2GP 13.141.

## Version 0.3.6: Wan2GP 13.141 decode-hook adaptation

The supplied Wan2GP-main(4).zip adds ControlNet-related temporary variables to
H3's cleanup immediately before VAE decoding. Version 0.3.5 matched the entire
cleanup assignment to insert latent capture, so this unrelated addition made
hook installation fail, including ordinary Ref2VA generation.

Capture now anchors to the unique native video-decoding branch. It still runs
after sampling and before the VAE dtype conversion, preserving original inference
latents. The installed native cleanup remains intact. This removes a brittle
text dependency without adding version checks or accepting ambiguous injection
sites. Existing wrappers continue to delegate to the adapted native method.

The new ControlNet-Union target-anchor path has different conditioning and pixel
restoration semantics. Active ControlNet layers are explicitly rejected when
latent operations are enabled; use native rendering with both latent options off
for ControlNet. Ordinary H3 FL2VA/Ref2VA latent workflows remain supported.

Tests pass with supplied Wan2GP **13.141**, with backward pipeline checks on the
previous supplied **13.14**. Ref2VA save/continue, all three continuation modes,
pre-decode float32 capture with an FP16 decoder, wrapper preservation, the Gradio
form/queue/export path and strict setting transport are exercised on CPU. No real
GPU render or complete third-party plugin installation was tested.

Stop Wan2GP, replace the plugin folder with this release, and restart. Existing
latent checkpoint formats are unchanged; use the corresponding original video.

## Version 0.3.5: Wan2GP 13.14 setting validation

Wan2GP now rejects unrecognized top-level generation settings. Older plugin
versions stored `_h3_latent_task_options_v1` there even when both latent options
were disabled. That caused the reported "Unknown setting" error before rendering.

Options now travel inside native `plugin_data`. Queued jobs also retain an
independent snapshot in their task envelope, outside generation parameters, so
the native worker can pop plugin_data without losing the selected options.
Task editing and imported runtime tasks normalize this transport before dispatch,
avoiding duplicate plugin_data keyword arguments. Legacy snapshots are migrated
when they reach the wrapped native validation/settings/import boundaries. Native
validation still rejects unrelated unknown settings; no catalog entries, hashes
or version allowlists are added. Jobs with both latent options disabled use the
native renderer, including fused Turbo catalog variants.

Compatibility was checked against the supplied **Wan2GP 13.14** source and its
pinned **Gradio 5.29.0**. The native H3 pipeline, packing and layout injections pass
CPU tests in all three continuation modes. The new cooperative inference
checkpoint remains native. The frozen-audio-prefix restriction now includes the
new third audio guide. Kept-soundtrack modes (`AS` / `KS`) are explicitly rejected
only when latent operations are enabled: the final audio may differ from decoded
inference audio, so these modes cannot promise a matching AV checkpoint.

Existing safetensors formats are unchanged. Stop Wan2GP, replace the plugin folder
with this release, and restart. Retry generation with both options unchecked
first. Old options reaching the migration hooks do not require manual editing.
This is CPU/source-level validation, not a GPU-quality or complete third-party
plugin interoperability certification. See VALIDATION.md and tests/settings_smoke.py.

## Version 0.3.4: shared plugin-data import recursion

Version 0.3.3 removed the global `state.change` listener but still listened to
`plugin_data.change`. This second object is shared with other extensions. If it
contains a cycle, Gradio recursively hashes it before an import/refresh callback
runs and raises `RecursionError`, even with both latent options disabled.

This release removes that listener too. Existing callbacks that write this form's
plugin data also emit a hidden, scalar refresh token. Its change event restores
the six latent controls by reading the payload without asking Gradio to hash it.
The plugin does not delete foreign data, patch Gradio's hashing, or raise Python's
recursion limit. Sync, async and generator callbacks retain their execution style.
The checkbox input callbacks still capture options directly; queued task snapshots
and all inference/continuation algorithms remain unchanged.

The remaining 0.3.3 failure is reproduced with cyclic shared plugin data in real
Gradio 5.29.0. This is a mechanism-level regression, not a reproduction of the
reporter's unavailable export or complete installation. If another extension
independently subscribes to a cyclic State, that extension can still cause the
same Gradio error; this release removes only this plugin's subscriptions.

### Update

Stop Wan2GP completely. Replace `plugins/wan2gp-h3-latent-continue` with the folder
from this ZIP (keep any backup outside `plugins`), then restart. This cleans up
obsolete files from older model-plugin releases as well as installing the fix.
Reinstallation of unchanged 0.3.3 files is not the fix. Retry the existing settings
export; no GPU render or new export should be needed to check the import.

## Version 0.2.3: safe user abort

Aborting an active generation now discards any incomplete latent capture and returns control to Wan2GP's normal abort handling. No checkpoint is written for an interrupted generation, and the missing-checkpoint safety check is not reported as a rendering error. That safety check remains active after a generation that claims normal completion but fails to publish its requested checkpoint.

## Version 0.3.1: closure-based plugin wrappers

Latent hook installation now follows captured function references as well as standard `__wrapped__` links. This supports plugins that wrap native H3 methods in closures without using `functools.wraps`. The wrapper and its captured state remain intact; the reachable native body receives the latent edits. If no unique matching native body can be reached, installation stops before changing any method instead of guessing.

CPU regression testing uses an undecorated closure around the installed native H3 generate method and checks that it still runs before and after all three continuation modes. This simulates the reported wrapper pattern; it does not validate the exact third-party plugins from the user's installation.

## Version 0.3.3: settings import recursion fix

Removed the plugin's listener on the global Gradio `state.change` event. Native queued/imported tasks reference their owning state, producing a cycle. That listener caused Gradio to recursively hash the entire state whenever a native callback returned it, raising `RecursionError` even with both latent options disabled.

Panel visibility still follows model selection and video-mode changes. The plugin still reads state as an input but no longer subscribes to its change events. No native state data is removed and Gradio's hashing function is not modified. Restart Wan2GP after replacing the plugin directory, then import the existing settings archive again; re-exporting should not be necessary for this failure.

The regression test first reproduced the failure on 0.3.2, then verified repeated Gradio callbacks returning the real native queued state without recursion. Full Stability Matrix startup and the reporting user's archive were not available locally. Generation and the minimum 18-frame context remain unchanged.

## Version 0.3.2: minimum latent history (comparative fix)

Reference mode now retains at least 18 source frames of original video/audio latent history when available, independently of the overlap used to assemble the output. On a 124-frame H3 source, a one-frame overlap previously supplied only one video latent block and two audio tokens. This update supplies six video latent blocks and 31 audio tokens. The native assembly overlap and output duration do not change. Larger existing Reference overlaps and the explicit 18/35/52-frame extended modes retain their requested context.

This is a targeted correction for insufficient context observed in the supplied test files, not a GPU-validated guarantee that all visual degradation is fixed. The provided pair genuinely used latent continuation, with valid finite tensors. It also used Spectrum and a Turbo LoRA; their contribution has not been isolated. First retry from the original source/checkpoint with the same settings to evaluate the context change. Test No Skipping separately if necessary, without changing several settings at once.

Checkpoints now record effective context (native overlap, selected latent indices and temporal positions), plus the native LoRA selection and acceleration settings for diagnosis. See `DIAGNOSIS-0.3.2.md` for the supplied-pair findings and the separate final-frame-count limitation.

## Purpose

Wan2GP's standard Continue Video workflow encodes the supplied MP4 again through the video VAE. Repeating that operation can progressively damage fine details, contrast, texture, and identity. H3 Latent Continue can save the original video and audio latents produced during inference and reuse them in a later continuation. The source context therefore does not undergo another VAE encode cycle.

User testing of the earlier 0.1.7 prototype with five successive continuations showed a clear visual improvement over the standard pixel-based workflow on the tested material. A slight visual join and a more noticeable audio join can still remain. Version 0.2.0 added experimental context modes aimed at those joins.

This plugin extends the existing **Media Generator** form. It does not create a separate generation tab or duplicate Wan2GP's prompt, model, LoRA, sampler, resolution, seed, or video settings.

## Installation

1. Close Wan2GP.
2. Delete the former `Wan2GP/plugins/wan2gp-h3-latent-prototype` directory if it exists. Keeping both folders would load two copies of the plugin.
3. Extract the ZIP. It contains one root directory named `wan2gp-h3-latent-continue`.
4. Copy that directory to `Wan2GP/plugins/`. The resulting path must be:

   `Wan2GP/plugins/wan2gp-h3-latent-continue/plugin.py`

5. Start Wan2GP, open **Plugins**, enable **H3 Latent Continue**, save, and restart Wan2GP if requested.
6. Reload the browser page completely with `Ctrl+F5`.
7. In the normal model selector, choose the corresponding ordinary **MiniMax H3** model.

Do not replace or edit any Wan2GP core file.

## Native MiniMax H3 models (0.3.0)

Use Wan2GP's ordinary **MiniMax H3** family. No additional model family, model weights, or duplicate model entries are registered by this release. Supported base architectures are FL2VA, FL2VA Pruned, Ref2VA and Ref2VA Pruned; native catalog variants resolving to those architectures share the controls.

This is an extension-only plugin. The former `MiniMax H3 Latent Continue` family and its four model definitions have been removed. On upgrade, close Wan2GP and replace the entire plugin directory, rather than copying files over the old release. Keep only `plugins/wan2gp-h3-latent-continue` and enable **H3 Latent Continue** in Plugins.

Select the corresponding native model and recreate old queued jobs or presets that reference `h3_latent_*` architectures. Existing video/checkpoint v1/v2 pairs remain readable: their saved identity already uses the native architecture. Keep the same underlying weight and VAE files, resolution and FPS.

Native model instances retain their class. Existing wrappers from other plugins are preserved when they delegate to the native H3 method. Native LoRA loading needs no identifier translation. Integration with RefMods, image-mode plugins, or refiners still depends on what those plugins do; an output-changing postprocessor can invalidate the saved latent/video match. Those third-party plugins have not been tested together in a full GPU session.

With both latent options unchecked, the original native rendering path runs normally, including configurations supported by Wan2GP itself. With latent saving or continuation enabled, only the supported single-phase video configurations are accepted.

## User interface

When a supported native MiniMax H3 model is selected, the existing Media Generator form receives these controls:

- **Save latent checkpoint next to output video**: saves the inference latents. Disabled by default.
- **Continue with latent**: expandable panel shown with Continue Video.
- **Enable Continue with latent (experimental)**: enables latent continuation. Disabled by default.
- **Matching H3 latent checkpoint (.safetensors)**: loads the checkpoint associated with the source MP4.
- **Join experiment**: selects the continuation method.
- **Video latent context (frames)**: 18, 35, or 52 frames for the extended modes.
- **Audio latent context (seconds)**: 0.5, 1, or 2 seconds for the extended modes.

The source MP4 remains in Wan2GP's normal **Video to Continue** field.

## Creating a video and its latent checkpoint

1. Select a supported native **MiniMax H3** model.
2. Configure the generation normally.
3. Enable **Save latent checkpoint next to output video**.
4. Generate the video.

After the final MP4 has been written, muxed, and given its metadata, the plugin writes a safetensors file in the same output directory with the same base name:

```text
outputs/example.mp4
outputs/example.safetensors
```

The console should report:

```text
[H3 Latent] Form captured: save=True, continue=False, model=...
[H3 Latent] Queued model=...: save=True, continue=False
[H3 Latent] Job options: save=True, continue=False
[H3 Latent] Saved ...example.safetensors
```

Latents cannot be recovered retroactively from a video generated without the save option. Encoding that MP4 later would produce reconstructed latents rather than the original inference latents.

## Continuing with saved latents

1. Select the same native MiniMax H3 model variant used for the source.
2. Select **Continue Video**.
3. Load the original, unmodified source MP4 in **Video to Continue**.
4. Open **Continue with latent**.
5. Enable **Enable Continue with latent (experimental)**.
6. Load the matching `.safetensors` file.
7. Select a join mode.
8. Enable latent saving as well if the result will be continued again.
9. Generate normally.

The model variant, configured video VAE, resolution, and FPS must match the checkpoint. The plugin also verifies that the MP4 bytes match the video recorded in the checkpoint. Moving or renaming both files is acceptable; modifying, remuxing, trimming, or re-exporting the MP4 changes its hash and is rejected.

The latent path never silently falls back to pixel encoding. Missing or incompatible data produces an explicit error.

## Join modes

| Mode | Behaviour | Recommended use |
| --- | --- | --- |
| **Reference (minimum 18-frame context)** | Uses original latents with at least 18 frames of history when available. The extended duration controls are ignored. | Initial reference and visual-quality baseline. |
| **Longer video/audio context** | Supplies 18, 35, or 52 saved video frames independently of Wan2GP's native overlap, plus 0.5, 1, or 2 seconds of time-positioned audio context. | Tests whether additional history reduces the join. |
| **Longer context + frozen audio prefix** | Uses the extended video context and copies the preceding audio latents into an immutable target prefix. The prefix and continuation are decoded together, then the pre-roll is removed at audio-sample precision. | Primary experiment for the audible join. Requires **No Skipping**. |

Start testing with **Longer context + frozen audio prefix**, **35 frames**, **1 second**, and **No Skipping**. A larger context consumes more memory and processing time and may make scene changes harder; 52 frames is not automatically better than 35.

The available context comes from the saved latent segment, not from the complete historical duration of a concatenated MP4.

### How the extended modes work

The video context consists of original inference latents placed at their H3 temporal positions. It is not re-encoded and no visual crossfade is added. The current implementation provides more video history but does not freeze a video prefix during denoising.

In the frozen-audio mode, the past audio occupies H3 target rows intended for fixed conditioning and remains unchanged during denoising. Its position is shifted into the past while any Ref2VA references retain their own positions. The audio VAE decodes the past and future together to reduce boundary artefacts. The pre-roll is then cut at 32 kHz without inserting silence.

Audio latent positions use a 40 Hz grid. Fractional timing at the video boundary is retained, and extended modes round the latent duration upward before cutting the decoded waveform to the requested output duration. This improves temporal handling but cannot force the model to continue the same music, voice, rhythm, or room tone perfectly.

## Recommended comparison

Create one source clip A with latent saving enabled and keep its MP4/checkpoint pair unchanged. From that pair, make three independent continuations:

1. Reference (minimum 18-frame context).
2. Longer video/audio context.
3. Longer context + frozen audio prefix.

Use **No Skipping** for all three branches when comparing the join methods. Keep the prompt, seed, model, LoRAs, resolution, FPS, sampler, step count, flow shift, and native overlap identical. Change only one context setting at a time.

Inspect several seconds before and after the join:

- image texture, skin, hair, identity, contrast, colour, framing, and motion;
- audio rhythm, timbre, level, clicks, silence, repeated syllables, and stereo ambience.

Then repeat the comparison across several sources and seeds and build separate chains of five continuations. Each branch must continue its own previous result and use the matching checkpoint.

If Turbo and Spectrum are normally combined, first compare the three branches with the same Turbo LoRA and Spectrum disabled. The frozen-audio mode explicitly rejects Spectrum and other skipped-step caches because they can alter conditioned target rows. A separate native-speed test without Turbo can help identify its effect on audio continuity.

## Checkpoint format

Version 0.3.0 reads checkpoint formats v1 and v2 and writes v2. Safetensors metadata identifies the format as:

```text
format=wan2gp.h3.latent-continuation
version=2
```

The file stores:

| Data | Purpose |
| --- | --- |
| Video latent tensor | Original final inference segment used as visual history. |
| Audio latent tensor | Original audio history; frozen-prefix checkpoints may include bounded pre-roll. |
| Final-frame reference | Associates the captured inference segment with the exported output tail. |
| `audio_origin_seconds` | Places saved audio relative to the first delivered video frame. |
| Generation metadata | Model/VAE identity, resolution, FPS, delivered frame count, generation settings, LoRAs, continuation mode, and join settings. |
| Video SHA-256 | Binds the checkpoint to the final MP4 after metadata writing. |

Tensors retain their original FP32, BF16, or FP16 dtype. Saving is atomic and refuses to overwrite an existing checkpoint. Use a new output name if a safetensors file with the intended name already exists.

At 1344×768 and 124 frames, expect roughly 26 MB for FP32 latent video, the auxiliary final-frame tensor, and audio, excluding small metadata. Exact size depends on the generated segment and dtype.

H3 Latent Continue checkpoints are not LoRAs and should remain in the selected Wan2GP output directory rather than a LoRA directory.

## Supported and rejected configurations

The plugin supports one completed video window per job and **one generation phase only**. Two-phase rendering is intentionally outside both the current and planned scope.

The plugin explicitly rejects configurations that can invalidate the latent/video association or the continuation geometry, including:

- two-phase generation;
- batch sizes other than one;
- resolution or FPS changes between source and continuation;
- masked or control-video editing on the latent-continuation path;
- custom frame scheduling;
- HDR or phase-two tiling;
- extra audio refinement;
- more than one generated window in a single job;
- Start Image combined with latent Continue Video;
- incompatible post-processing, trimming, or output-tail changes;
- skipped-step caches and additional audio guides in frozen-audio-prefix mode.

Ref2VA image, video, and audio references remain separate from the frozen prefix. Their perceptual interaction with the extended context still requires real-world testing.

## Wan2GP update compatibility

Version 0.2.1 removed the former source-file hash gate and its allow-list. Rendering does not compare Wan2GP versions, commits, or Python file contents. Updating `wgp.py`, adding models, or changing MiniMax H3 source code therefore does not trigger rejection merely because a file changed.

The plugin was exercised against the supplied Wan2GP 12.72 and 13.0 archives. On Wan2GP 13.0 it uses the native generation and VAE progress reporting. On 12.72 that additional progress integration is simply absent. The video VAE, transformer, weight loader, and most model components are imported from the installed Wan2GP version, so native fixes in those components are used automatically.

The plugin now runs the installed native H3 code. It inserts local capture/continuation edits into `generate`, audio-position packing and the transformer layout method when a latent job is first requested. Existing wrapper chains remain in place; native jobs with the options disabled use the original functions. No source hash, version or commit allow-list is used.

Changes outside those insertion points are retained. If a required insertion point is removed or becomes ambiguous, only the latent operation is rejected with a precise error; ordinary native generation remains available. This is a functional dependency, not a whole-file compatibility check. A third-party plugin that replaces a method without delegating to its original cannot be guaranteed to compose with this integration.

`check_compatibility.py` is an optional installation diagnostic. It checks only that three Wan2GP/H3 entry files exist. It performs no hash or version comparison, does not certify runtime compatibility, and is never called during generation:

```bash
python plugins/wan2gp-h3-latent-continue/check_compatibility.py .
```

Official latent capture, injection, and final-export hooks in Wan2GP would be the best long-term way to reduce this dependency on internal functions.

## Architecture

- `native_bridge.py`: wraps native weight loading to record checkpoint identity, installs local edits without replacing model classes, and preserves existing method wrappers.
- `native_edits.json`: the small insertion/replacement blocks applied to the installed H3 generation function. Unrelated installed code is retained.
- `latent_runtime.py`: per-task state, checkpoint selection, temporal context, validation and CPU capture.
- `checkpoint.py`: safetensors validation, video binding and atomic output.
- `plugin.py`: adds controls to the existing Media Generator form on native H3 models.
- `form_bridge.py` and `integration.py`: transport options through native form events, queueing, generation and final video export.
- `seams.py`: time positioning of the extended audio/video context.
- `packing.py` and `progress_bridge.py`: test support retained from the earlier implementation; production rendering uses the installed native packing and progress modules.

No Wan2GP file is changed on disk. The former copied pipeline, custom model handler, model definitions and LoRA identifier adapter have been removed. Options are frozen in queued tasks, and abort cleanup is retained. A checkpoint represents a completed video segment, not an interrupted sampler state.

## Validation

The 32 passing CPU unit tests cover checkpoints, export and native job routing. The obsolete model-alias LoRA tests have been removed. `pipeline_smoke.py` and `ui_smoke.py` also pass independently against both supplied Wan2GP 12.72 and 13.0 source trees.

The tests cover:

- Gradio controls disabled by default and embedded in the native form;
- checkbox and file values reaching an immutable queued-task snapshot;
- checkpoint saving beside a simulated final video in a custom output directory;
- exact tensor dtype and value preservation;
- v1/v2 loading and fractional audio origins;
- final-video hash binding and output-tail matching;
- single and multiple video export blocks;
- Reference, extended-context, and frozen-audio-prefix execution;
- exact preservation of the frozen audio prefix during denoising;
- output duration, reload, and a second continuation;
- absence of video/audio encoder calls on the latent continuation path;
- Wan2GP 13.0 `SharedState` handling and optional progress integration.

Run the tests from the plugin directory with the required Python dependencies available:

```bash
python -m unittest discover -s tests -q
python tests/pipeline_smoke.py /path/to/Wan2GP
python tests/ui_smoke.py /path/to/Wan2GP
```

The pipeline tests use tiny deterministic CPU substitutes for the H3 DiT and VAEs. They do not generate H3 images. Real model weights, CUDA/MMGP behaviour, production VRAM/RAM use, real codecs, audio muxing, interactions with every third-party plugin, and perceptual join quality cannot be validated by those tests.

## Known limitations

- Original latents greatly reduce repeated VAE damage but do not guarantee stable identity, texture, contrast, or motion in newly generated frames.
- A visible join can remain because the new segment is generated rather than copied from a continuous original inference.
- Audio continuity is model-dependent. The frozen prefix reduces boundary problems but cannot guarantee a continuous musical phrase, voice, ambience, or loudness.
- Longer context costs additional memory and computation.
- The complete historical video is not fed into the model; only a bounded saved context is used.
- The plugin does not keep all old segments as untouched master media. Re-encoding of already assembled output by external editing or later workflows remains outside its control.
- No frozen video prefix is implemented in the current extended modes.
- One phase is a permanent scope restriction.

## Troubleshooting

- **Plugin appears twice:** close Wan2GP and delete `plugins/wan2gp-h3-latent-prototype`. Keep only `plugins/wan2gp-h3-latent-continue`.
- **Controls are missing:** enable H3 Latent Continue, restart Wan2GP, use `Ctrl+F5`, and select a supported native **MiniMax H3** model. Check for `Inline controls connected to N native form event(s)` in the console.
- **The console reports `save=False`:** create a new task after enabling the checkbox. Existing queued tasks keep their original snapshot. Check the `Form captured`, `Queued`, and `Job options` lines.
- **Video exists but no checkpoint was written:** inspect the final save error and verify that no safetensors file already uses the same name.
- **Source mismatch:** use the exact original MP4 associated with the checkpoint. Do not remux or edit it.
- **Resolution/FPS mismatch:** restore the original settings; latent tensors cannot be resized safely.
- **Model/VAE mismatch:** use the same model variant and configured VAE files as the source generation.
- **`Output tail no longer matches`:** disable incompatible processing or trimming. The plugin refuses to publish a checkpoint associated with a different output.
- **Old `source compatibility check failed` message:** an earlier plugin version is still installed. Replace its complete directory with version 0.3.0 and restart Wan2GP.
- **Frozen audio mode is rejected:** choose **No Skipping**, disable Spectrum or other skipped-step caches, and remove additional audio guides.

## Version history

- **0.3.0:** moves all controls to native MiniMax H3 models, removes the separate model family, and applies local latent hooks while retaining native classes and existing wrapper chains.

- **0.2.3:** treats a user abort as a normal interruption, discards incomplete latent data, and preserves the missing-checkpoint guard for genuinely completed jobs.
- **0.2.2:** renamed the plugin, folder, model labels, and ZIP to H3 Latent Continue while preserving internal identifiers and checkpoint compatibility.
- **0.2.1:** removed source hash/version blocking and added optional Wan2GP 13.0 progress integration.
- **0.2.0:** added extended video/audio context and frozen audio prefix experiments; introduced checkpoint format v2.
- **0.1.7:** accepted Wan2GP's list of exported video blocks when matching the output tail.
- **0.1.6:** stored options in each queued task so UI state could not be lost before generation.
- **0.1.5:** captured checkbox and file values in Wan2GP's native form event.
- **0.1.4:** added per-session option snapshots.
- **0.1.3:** translated prototype model identifiers for native AdaLN LoRA preprocessing.
- **0.1.2:** normalized line endings in the former source hash check.
- **0.1.1:** added required model discovery paths and robust Gradio State handling.
- **0.1.0:** initial video/audio latent continuation prototype.

## Related projects

These projects informed the feasibility study. Their code is not bundled in this plugin:

- [ComfyUI H3 Motion Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)
- [ComfyUI MiniMax H3 Context Loop](https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop)
- [ComfyUI MiniMax H3 Inpaint Tools](https://github.com/panghea/ComfyUI-MiniMax-H3-Inpaint-Tools)

See `THIRD_PARTY.md` and the included license files for provenance and licensing information.
