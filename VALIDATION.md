# 0.6.5 — WanGP 19.02

Reported: "no reachable native method contains the required latent insertion points" after the 19.02 update.
Anchor check on the supplied 19.02 source: 16 of 18 edits matched; the audio length (audio_tail_latents + Pika) and
the seeded audio noise (Pika branch) did not. Both re-anchored as 19.02 variants.

- All suites on 19.02 (supplied zip), 18.19 and 18.18: real generate() harness (incl. 18.18 audio noise, NaN stop,
  exact join length), with H3 Audio Refinement Controls (in-place rewrite verified), part2 export with ffmpeg,
  Gradio UI with the real plugin manager, settings transport, real H3 VAE decoder/encoder checks, unit tests.
- New on 19.02: Pika H3 Flex with save, baseline/context/audio_prefix/pinned continuations: pinned latents and frozen
  audio bit-exact. PTD only changes text-embedding refinement (euler path): no interaction.

Not tested: GPU render on 19.02.

# 0.6.4 — 18.18 audio noise for plain renders

- Real generate() harness: option alone -> Text Only noise identical to the validated 18.18 draw, video unchanged;
  Start with Image: channel 0 equal, channel 1 = 18.18 rule, video unchanged; window_no 3 accepted.
- Generation wrapper: option alone -> noise-only job (hooks installed, no preflight: discard frames accepted, no
  checkpoint required); no option -> native call without a job. All other checks pass (18.19, 18.17, AR plugin).

# 0.6.3 — exact join cut at the end

Supplied: 03h23m31s (exact join, overlap 18, 14.4 s): manifest target_frames 349, pinned 5 frames from source frame
238, audio 681 latents = ceil(362/24*40) + 72 prefix + 5 tail -> the model generated 362 frames (normalize_frame_count
rounds 349 up) and 13 were cut; speech cut at the end.

- Real generate() harness: 124-frame window, overlap 18 -> 111 requested, 107 generated and delivered, window output
  120 frames, audio matching; message logged; successive join from a 107-frame segment (5 pinned frames); 35-frame
  overlap (22 pinned + seamless end) unchanged. 18.19, 18.17 and with H3 Audio Refinement Controls: all pass.
- WanGP 18.19 accepts a window shorter than requested (stop_current_sample; audio trimmed to the frames).

Not tested: GPU render.

# 0.6.2 — NaN diagnosis

Supplied log: continuation of a 0.6.1 checkpoint (context mode, 2 phases, 141 target frames, 18.18 audio noise):
checkpoint loaded (finite, verified on load), then "invalid value encountered in cast" in the phase-1 preview at step
2, blank decoded frames, "Ambiguous output tail" at export. The NaN appeared during that render's denoising.

- Harness: a model stand-in returning NaN in audio latents 3..5 stops the render at the end of denoising (one phase)
  and at the end of phase 1 (two phases) with the latents named; native er_sde NaN (known stand-in artefact) is now
  reported the same way. All other checks unchanged and passing (18.19, 18.17, with H3 Audio Refinement Controls).

# 0.6.1 — same seed as WanGP 18.18

Supplied: continuation 04h20m55s (WanGP 18.18, plugin 0.4.0) and its re-render 01h23m23s (18.19, 0.6.0) with
identical settings (WanGP metadata and checkpoint manifests compared). Second parts: same image start, audio
envelope correlation 1.00/0.97/0.88/0.87 over the first 4 s, then the dialogue diverges. Cause in the 18.19 source:
audio_t + H3_AUDIO_TAIL_LATENTS and channel-major unpack_audio -> channel 1 noise shifted by 5 latents.

- Real generate() harness, 18.19: initial audio noise with the option equals the pre-18.19 draw on the first 207
  latents (draw order checked against native first), channel 0 equal and channel 1 different without it, tail
  finite noise, video output identical, setting recorded; refinement and er_sde edited lines executed.
  18.17: no tail, noise identical to native and logged. With H3 Audio Refinement Controls: combined, all pass.
- UI: checkbox off by default, captured by save_inputs, restored, 9 plugin controls. Unit, part2, settings pass.

Not tested: GPU re-render of 04h20m55s with the option.

# 0.6.0 — two-phase background morph

Supplied: two-phase source (243 frames, checkpoint 0.4.0 with phase-1 latents), continuation (overlap 35, exact join,
seamless end applied: switch difference 1.31) and log. Ceiling drift from frame 242: 1.76 / 3.76 / 5.03 / 5.50 / 5.58
after 6/12/24/48/72 frames vs 1.27-1.83 inside part 1; left wall similar. Saved draft vs final reduced to phase 1:
0.63 rms (0.55 low frequencies) against 0.39-0.47 latent-to-latent motion.

- Pipeline harness: phase 1 encodes exactly the last pinned source frames resized to the phase-1 size and pins that
  encode; option off keeps the exact saved draft with no encode; single-phase checkpoint continued in two phases;
  9-frame pin (successive join) uses keep_all_latents. All checks pass on 18.19 and 18.17, with and without
  H3 Audio Refinement Controls; unit, part2, settings, UI and both real-VAE checks pass.

Not tested: GPU render of the morph after the change.

# 0.5.2 — seamless end

User media after 0.5.1 (overlap 35): seam 2.11 vs median 0.91 (x2.3, was x7.7); RGB step ~0; residual evenly spread
(2-3x per region), consistent with decoder look-ahead rather than the references.

- Real H3 decoder code (random weights, active residual branches, SDPA attention): source vs exact-join window decode
  identical on frames 226..238, different on 221..225 (dropped overlap) and 239..242 (replaced). Before activating
  the zero-initialised branches the decoder showed no temporal mixing; the check now asserts the mixing.
- Real generate(): with a 35-frame overlap the 5 re-decoded frames + switch frame equal the window output; skipped
  with an 18-frame overlap; off when unticked.
- Export with real save_video/ffmpeg: last 5 source frames replaced in the complete video (switch difference 0),
  second part = 5 + new frames with its audio aligned; inconsistent re-decode refused, source kept.
- Gradio: checkbox default on, captured by save_inputs (unticked -> False), restored, 7 plugin controls.
- All unit, pipeline (18.19/18.17, with and without H3 Audio Refinement Controls), part2, settings, UI checks pass.

Not tested: GPU render with the real weights.

# 0.5.1 — audio references

Reported: "audio-prefix / exact join does not support additional audio guides" with Ref2VA voice references.
- Ref2VA with an image reference and a voice audio guide: baseline, context, audio-prefix and exact join run; packed
  layouts with references checked at every step; pinned video and frozen audio bit-exact.
- FL2VA audio guide driving the soundtrack: refused with the explicit message.
- Harness correction: waveforms are now sample-major (samples, channels) like WanGP's; previous runs passed 2-sample
  audio. All checks pass again with realistic audio on 18.19 and 18.17, with and without H3 Audio Refinement Controls.

# 0.5.0 — Exact join

Reported media: seam difference 6.83 vs median 0.89 (x7.7), global shift 0.5 px, RGB mean -1.9/-1.5/-0.4, unchanged
sharpness; audio seam in a silence, no click. Checkpoint: one phase, 832x1088, 243 frames, 72 video latents.

- Pipeline harness on 18.19 and 18.17, each with and without H3 Audio Refinement Controls: pinned latents bit-exact
  with euler, res_multistep and ralston_2s, in one phase and in two phases (phase-1 and full-resolution latents), with
  form masking 0.5 / denoising 0.7; frozen audio prefix bit-exact with its origin moved by (pinned - 1) frames; 22-frame
  pin with a 35-frame overlap; successive exact joins (111-frame segment -> 9 pinned frames); WanGP overlap/part2
  bookkeeping; refusals (overlap too short, audio context too short); exact join + audio refinement; no AV re-encode.
  The native packed-layout check passes at every step.
- Mutation: without the phase-1 pin the pinned latents differ by ~2.0 and the test fails.
- er_sde not checked: native er_sde already returns NaN from sigma 1 in this harness (division by 1 - sigma).
- All other unit (48), pipeline, part2, Gradio UI and settings checks pass on both versions.

Not tested: GPU quality of the join (decoder look-ahead residual), 2 Phases Plus.

# 0.4.2 — Wan2GP 18.19 (and 18.17)

Source: WanGP-main_1.zip (WanGP_version 18.19). With 0.4.1 one of 12 generate() anchors (audio length) had no match,
which is the reported failure in one and two phases. Transformer/packing anchors unchanged.

- Pipeline harness on 18.19 and 18.17, each with and without H3 Audio Refinement Controls 1.1.0: all 11 checks
  (generation, three join modes, two phases, phase-1 resample, refinement, Ref2VA, wrappers, layout cache) pass;
  returned audio includes the native tail on 18.19; bytecode check passes with the refinement plugin.
- Second-part test on both: 100 ms audio tail kept (part2 audio - video = complete audio - video).
- Unit (44), Gradio UI and settings smokes pass on both versions.

Not tested: GPU render; the 2 Phases Plus plugin (source not supplied).

# 0.4.1 — phase-1 size change

- Reproduced the reported "no saved latents match the WxH generation stage" in the pipeline harness: two-phase source
  at 128x128 (phase 1 64x64), continuation after H3_TWO_PHASE_SCALE 2 -> 1.5 (phase 1 96x96).
- Fixed: phase-1 latents resampled 4x4 -> 6x6 latent cells; the native packed layout check passes at every step;
  new checkpoint phase-1 latents at 96x96; manifest records the resample. All other checks pass, with and without
  H3 Audio Refinement Controls. Line-number mapping checked on a decorated native-style function.

Not determined: which setting changed the phase-1 size in the reported case (2 Phases Plus source not supplied).

# 0.4.0 — Audio Refinement Extra Phase, Wan2GP 18.17

Reported error reproduced from the code path: preflight refused any `audio_refinement` other than `none`,
including the refinement plugin's disabled value `disabled:6:0.5`.

- tests/pipeline_smoke.py run twice on the real 18.17 generate(): native refinement, and with the supplied
  H3 Audio Refinement Controls 1.1.0 installed first (as at WanGP start). With the plugin, the bytecode check confirms
  the installed body equals WanGP source + its two rewrites; the latent version applies them (enabled:2:0.3 ->
  2 steps from audio sigma 0.5625, exact schedule). Native: 6 steps from 0.75.
- Latent continuation + refinement in the three modes: join audio conditions kept, frozen prefix bit-identical after
  refinement, prefix offset kept, no AV re-encode, refined audio saved; two phases + refinement; renders without a
  join refine natively. All previous checks pass in both configurations.
- Mutations: without the composition the latent path falls back to WanGP's grid (0.5455 instead of 0.5625) and fails;
  without the prefix restore the frozen-prefix check fails.
- tests/test_refinement.py: join rows before kept reference rows, prefix restore, unknown rewrite warning, settings.

Not tested: GPU listening tests; Keep Audio References with real audio references combined with a latent join
(row order unit-tested only); other plugins that may rewrite generate in place (2 Phases Plus source not supplied).

# 0.3.9 — second part export, Wan2GP 18.17

- tests/part2_smoke.py uses the actual native save_video and combine_video_with_audio_tracks with ffmpeg:
  40 source frames + 18-frame overlap + 7 trimmed tail frames. The second part has exactly the 99 new frames, its
  first frame is a new frame, its audio starts on the new part (silent source / tone new part) and lasts 99/24 s.
  Complete video byte-identical after export; no temporary file left. Also: no audio track, altered tail (skipped
  with warning), render without Continue Video (no second part), chunk slicing.
- A deliberate mutation (start 12 frames early) fails this test.
- Pipeline harness: real generate() records overlap 18 / history 17 and tail fingerprints for a continuation, nothing
  for a plain render. All previous unit, pipeline, Gradio UI and settings checks pass.

Not tested: a real WanGP GPU render; the gallery is not updated for the second part.

# 0.3.8 — two phases, Wan2GP 18.17

- 38 CPU unit tests pass (two phases accepted, three phases and tiled phase 2 rejected).
- Native pipeline smoke on 18.17, two phases at 64x64 (phase 1 32x32): native upscaler used; phase-1 and final latents
  captured in float32; checkpoint round trip; continuation in baseline/context/audio-prefix modes with no AV encoder call;
  phase 1 uses phase-1 latents and phase 2 full-resolution latents; prompt frame 32x32 then 64x64; frozen audio prefix
  intact after phase 2; successive two-phase continuation; Ref2VA two-phase save/continue; two-phase checkpoint continued
  with one phase; single-phase checkpoint refused for two phases; inconsistent phase-1 manifest rejected; native
  two-phase render with options off.
- The transformer stand-in builds the actual native packed layout at every denoising step and checks condition video/audio
  row counts. A deliberate mutation (full-resolution latents in phase 1) fails this check.
- All single-phase checks, Gradio UI/queue/sidecar and settings smokes pass unchanged.

Not tested: GPU render quality, the "2 Phases Plus" plugin (its source was not supplied), tiled phase 2.

# 0.3.7 — Wan2GP 18.17

Source: WanGP-main.zip (wgp.py WanGP_version 18.17).

Reproduced with 0.3.6 in the native pipeline harness: the same RuntimeError as the report. Two of eight
generate() anchors (audio_t, audio decode) and one transformer _layout anchor no longer matched
because 18.x added `audio_time_offset`.

- 38 CPU unit tests pass.
- Native pipeline smoke passes on 18.17: original generation, baseline/context/audio-prefix continuation,
  overlap=1 minimum context, frozen audio prefix, sample counts, Ref2VA save/continue, ControlNet rejection,
  float32 capture before the FP16 VAE cast, wrapper chains, layout cache, loader identity.
- New: latent continuation with a native audio repair offset keeps the saved-latent audio geometry and logs it;
  a disabled job with the same input still uses the original native path.
- Gradio 5.29.0 UI/queue/sidecar smoke and settings-validation/import/edit smoke pass against 18.17.

CPU stand-ins replace the model weights. No GPU render or quality comparison was run. The 13.141 source was
not available for a backward check; 0.3.7 targets 18.17.

# 0.3.6 — Wan2GP 13.141

Source: Wan2GP-main(4).zip (wgp.py WanGP_version 13.141).

The old native_edits capture pattern has zero matches in this source, while the other seven generate edits still match once. The cleanup now contains control_source and an additional control-anchor assignment. The replacement anchors only to the unique decode branch and leaves those assignments unchanged.

- 38 CPU unit tests pass.
- Native pipeline smoke passes on supplied 13.141 and prior 13.14: original generation, baseline/context/audio-prefix continuation, minimum context with overlap=1, frozen audio prefix/sample counts, cache/layout hooks and existing closure wrappers.
- Added Ref2VA-mode capture and continuation regression with no AV re-encoding.
- Added float32 inference capture with an FP16 decoder: saved video latents remain float32 and decoded output is FP16.
- Active ControlNet layers explicitly fail for a latent operation. Native jobs with options disabled retain the original dispatch path.
- Gradio 5.29.0 UI/queue/sidecar smoke and native settings-validation/import/edit smoke pass against 13.141.

CPU stand-ins replace model weights/encoders for pipeline tests. No CUDA quality test or full third-party-plugin stack was run. ControlNet latent continuation is unsupported; checkpoint formats and inference continuation algorithms are unchanged.

# 0.3.5 — Wan2GP 13.14 compatibility

Supplied source: Wan2GP-main(3).zip, wgp.py WanGP_version 13.14, Gradio requirement 5.29.0.

- settings_smoke.py uses the actual supplied setting_names validator and native validate_settings body: reproduces the old unknown-key failure, then confirms migration reaches native model validation. Unrelated unknown keys remain rejected.
- Tests enabled/disabled snapshots with the reported fused Turbo model ID resolved to its native base. Native add_video_task and _build_runtime_task bodies are executed; task edits, imported legacy payloads, option immutability and the worker's explicit plugin_data plus filtered-params calling convention are checked.
- 38 CPU unit tests pass (checkpoint, options, abort, frame export, seam geometry and refresh bridge).
- Actual Gradio UI smoke passes against supplied 13.14: form insertion, cyclic-state refresh regression, queue snapshots and safetensors sidecar with a fake encoder.
- Native pipeline smoke passes against supplied 13.14: generation, three latent continuation modes, minimum 18-frame history for overlap=1, frozen prefix preservation, sample counts, no AV re-encoding, existing wrapper chains, packing/layout caching and native loader class retention. The harness supplies the newly imported native inference_checkpoint and excerpts helpers; it does not replace production code.
- Kept soundtrack guard is tested enabled/disabled. No full GPU render, full browser session, actual third-party plugins, or exhaustive new H3 reference combinations were tested.

# 0.3.4 shared plugin-data regression

- Compared the reporter's plugin ZIP against released 0.3.3: all 25 release files match after CRLF normalization. Obsolete extra model files are not referenced by the active extension.
- Read native Wan2GP 13.0 import/form refresh output wiring and Gradio 5.29.0 get_state_ids_to_track. A State change listener causes recursive hashing before the callback executes; adding a try/except inside the restore callback would be too late.
- Reproduced RecursionError in Gradio utils.deep_hash using the unchanged 0.3.3 plugin and a cyclic foreign dictionary in shared plugin_data. The previous 0.3.3 test covered only the global task state and missed this second state object.
- The identical cyclic payload succeeds with the new scalar notification bridge. Repeated refreshes restore the controls, and foreign cycles remain intact. Neither shared State has a plugin change listener.
- 38 CPU unit tests pass, including callback styles, skips, component-keyed output dictionaries, multiple outputs, idempotence and form isolation.
- Actual Gradio/native plugin API smoke test passes on supplied Wan2GP 13.0: form insertion, option capture, queue snapshots, global cyclic state, cyclic plugin_data restoration, and CPU sidecar writing with a fake video encoder.

The reporter's exported archive, exact Wan2GP/Gradio versions and other extensions were unavailable. This confirms a remaining plugin defect and its mechanism-level fix, not the exact origin of the reporter's circular data. No full browser or GPU generation test was performed for this UI-only fix. No inference algorithm changed. Other plugins independently watching cyclic State objects remain outside this fix.

# 0.3.3 settings import regression

The added Gradio regression fails on 0.3.2 with RecursionError in utils.deep_hash. It creates a task using the supplied native add_video_task body, asserts task.params.state points to the owning state, and invokes a Gradio callback returning that state twice. Removing the global state.change listener allows both calls to complete. This reproduces the cyclic-state failure mechanism, not an end-to-end import of the reporter's unavailable ZIP.

Generation logic and minimum context are unchanged. Existing UI/queue/sidecar checks are retained.

# 0.3.2 validation

32 CPU unit tests pass. The native pipeline smoke test passes on the supplied Wan2GP 13.0 source, including overlap=1: six exact source video blocks are selected, the audio context contains 31 tokens, assembly duration remains unchanged, and neither AV encoder is called. The Gradio form/queue/export test passes on the same source.

No real H3 GPU quality test was possible. Minimum history is a targeted comparative correction, not proof that the reported visual regression is fully resolved. See DIAGNOSIS-0.3.2.md for the supplied media findings and the unresolved final-mux frame-count discrepancy.

# 0.3.1 regression

Native pipeline smoke test includes a closure wrapper without `functools.wraps`; both its before/after calls survive hook installation and generation. The actual third-party plugins have not been supplied or tested on GPU.

# Validation — 0.3.0

CPU tests cover task options, abort handling, checkpoint writing and native passthrough with options disabled. Obsolete custom-model LoRA alias tests were removed.

The integration harness uses the supplied native H3 generate body, packing module and transformer layout from Wan2GP 12.72 / 13.0, with tiny CPU DiT/VAE stand-ins. It checks all three continuation modes, frozen audio prefix, wrapper preservation, hook idempotence, layout cache, native loader identity and unchanged pipeline class. The Gradio harness checks extension-only discovery (zero extra model declarations), native form/queue propagation and checkpoint output.

Third-party wrappers are simulated. No full GPU validation with RefMods, Face Refiner, Image Mode or the user's other plugins has been performed. Real output-altering plugins may still cause an explicit latent/output mismatch.

Run from the plugin directory with the testing dependencies available:

```bash
python -m unittest discover -s tests -q
python tests/pipeline_smoke.py /path/to/Wan2GP
python tests/ui_smoke.py /path/to/Wan2GP
```
