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
