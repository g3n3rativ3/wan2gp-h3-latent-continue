# H3 Latent Continue for Wan2GP

Version **0.2.2**. MiniMax H3 latent continuation for Wan2GP. **Single-phase generation only.**

## Purpose

Wan2GP's standard Continue Video workflow encodes the supplied MP4 again through the video VAE. Repeating that operation can progressively damage fine details, contrast, texture, and identity. H3 Latent Continue can save the original video and audio latents produced during inference and reuse them in a later continuation. The source context therefore does not undergo another VAE encode cycle.

User testing with five successive continuations showed a clear visual improvement over the standard pixel-based workflow on the tested material. A slight visual join and a more noticeable audio join can still remain. Version 0.2.0 added experimental context modes aimed at those joins.

This plugin extends the existing **Media Generator** form. It does not create a separate generation tab or duplicate Wan2GP's prompt, model, LoRA, sampler, resolution, seed, or video settings.

## Installation

1. Close Wan2GP.
2. Delete the former `Wan2GP/plugins/wan2gp-h3-latent-prototype` directory if it exists. Keeping both folders would load two copies of the plugin.
3. Extract the ZIP. It contains one root directory named `wan2gp-h3-latent-continue`.
4. Copy that directory to `Wan2GP/plugins/`. The resulting path must be:

   `Wan2GP/plugins/wan2gp-h3-latent-continue/plugin.py`

5. Start Wan2GP, open **Plugins**, enable **H3 Latent Continue**, save, and restart Wan2GP if requested.
6. Reload the browser page completely with `Ctrl+F5`.
7. In the normal model selector, choose one of the models carrying the **Latent Continue** suffix.

Do not replace or edit any Wan2GP core file.

## Available models

The plugin adds four model entries and delegates weight loading to Wan2GP's native MiniMax H3 handler:

- MiniMax H3 FL2VA 33B — Latent Continue
- MiniMax H3 FL2VA Pruned 20B — Latent Continue
- MiniMax H3 Ref2VA 33B — Latent Continue
- MiniMax H3 Ref2VA Pruned 20B — Latent Continue

The internal architecture identifiers remain `h3_latent_*`. This preserves compatibility with existing presets, queued tasks, and latent checkpoints made with the earlier H3 Latent Prototype releases.

## User interface

When a Latent Continue model is selected, the existing Media Generator form receives these controls:

- **Save latent checkpoint next to output video**: saves the inference latents. Disabled by default.
- **Continue with latent**: expandable panel shown with Continue Video.
- **Enable Continue with latent (experimental)**: enables latent continuation. Disabled by default.
- **Matching H3 latent checkpoint (.safetensors)**: loads the checkpoint associated with the source MP4.
- **Join experiment**: selects the continuation method.
- **Video latent context (frames)**: 18, 35, or 52 frames for the extended modes.
- **Audio latent context (seconds)**: 0.5, 1, or 2 seconds for the extended modes.

The source MP4 remains in Wan2GP's normal **Video to Continue** field.

## Creating a video and its latent checkpoint

1. Select a model with the **Latent Continue** suffix.
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

1. Select the same Latent Continue model variant used for the source.
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
| **Reference (0.1.7)** | Uses the original latent continuation path. The extended duration controls are ignored. | Initial reference and visual-quality baseline. |
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

1. Reference (0.1.7).
2. Longer video/audio context.
3. Longer context + frozen audio prefix.

Use **No Skipping** for all three branches when comparing the join methods. Keep the prompt, seed, model, LoRAs, resolution, FPS, sampler, step count, flow shift, and native overlap identical. Change only one context setting at a time.

Inspect several seconds before and after the join:

- image texture, skin, hair, identity, contrast, colour, framing, and motion;
- audio rhythm, timbre, level, clicks, silence, repeated syllables, and stereo ambience.

Then repeat the comparison across several sources and seeds and build separate chains of five continuations. Each branch must continue its own previous result and use the matching checkpoint.

If Turbo and Spectrum are normally combined, first compare the three branches with the same Turbo LoRA and Spectrum disabled. The frozen-audio mode explicitly rejects Spectrum and other skipped-step caches because they can alter conditioned target rows. A separate native-speed test without Turbo can help identify its effect on audio continuity.

## Checkpoint format

Version 0.2.2 reads checkpoint formats v1 and v2 and writes v2. Safetensors metadata identifies the format as:

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

The plugin still carries an adapted H3 `generate()` pipeline because it needs latent capture and injection points unavailable through the standard plugin API. A future removal of required functions or a change to H3 latent conventions may therefore require an update. Future changes made only inside Wan2GP's native `generate()` body are not automatically merged into the adapted copy.

`check_compatibility.py` is an optional installation diagnostic. It checks only that three Wan2GP/H3 entry files exist. It performs no hash or version comparison, does not certify runtime compatibility, and is never called during generation:

```bash
python plugins/wan2gp-h3-latent-continue/check_compatibility.py .
```

Official latent capture, injection, and final-export hooks in Wan2GP would be the best long-term way to reduce this dependency on internal functions.

## Architecture

- `handler.py`: declares four distinct H3 model entries, delegates native model loading, and upgrades only the selected pipeline instance.
- `pipeline.py`: adapted H3 pipeline with latent preparation, temporal injection, audio-prefix handling, and capture before decode.
- `latent_runtime.py`: per-task state, checkpoint selection, temporal context, validation, and CPU capture.
- `packing.py` and `layout.py`: explicit placement of video and audio context tokens.
- `checkpoint.py`: safetensors format, validation, video binding, and atomic output.
- `plugin.py`: adds controls to the existing Media Generator form.
- `form_bridge.py` and `integration.py`: preserve the selected options through form saving, queueing, generation, video export, metadata writing, and final checkpoint publication.
- `progress_bridge.py`: optional integration with Wan2GP 13.0 progress reporting.

The lifecycle bridge is required because the normal metadata hook cannot access raw inference latents and does not by itself associate a capture with the final muxed filename. The bridge patches functions in memory through Wan2GP's plugin API; it does not modify core files on disk.

Options are also copied into each queued task. This prevents later UI edits or loss of shared `plugin_data` from changing an already queued job.

The saved file represents a completed video segment. It is not a sampler-state checkpoint and cannot resume an interrupted denoising step.

## Validation

Version 0.2.2 passes 31 CPU tests. `pipeline_smoke.py` and `ui_smoke.py` also pass independently against both supplied Wan2GP 12.72 and 13.0 source trees.

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
- **Controls are missing:** enable H3 Latent Continue, restart Wan2GP, use `Ctrl+F5`, and select a model carrying the **Latent Continue** suffix. Check for `Inline controls connected to N native form event(s)` in the console.
- **The console reports `save=False`:** create a new task after enabling the checkbox. Existing queued tasks keep their original snapshot. Check the `Form captured`, `Queued`, and `Job options` lines.
- **Video exists but no checkpoint was written:** inspect the final save error and verify that no safetensors file already uses the same name.
- **Source mismatch:** use the exact original MP4 associated with the checkpoint. Do not remux or edit it.
- **Resolution/FPS mismatch:** restore the original settings; latent tensors cannot be resized safely.
- **Model/VAE mismatch:** use the same model variant and configured VAE files as the source generation.
- **`Output tail no longer matches`:** disable incompatible processing or trimming. The plugin refuses to publish a checkpoint associated with a different output.
- **Old `source compatibility check failed` message:** an earlier plugin version is still installed. Replace its complete directory with version 0.2.2 and restart Wan2GP.
- **Frozen audio mode is rejected:** choose **No Skipping**, disable Spectrum or other skipped-step caches, and remove additional audio guides.

## Version history

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
