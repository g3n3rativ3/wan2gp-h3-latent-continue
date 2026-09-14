# Supplied-pair diagnosis — 0.3.2

## Confirmed from the supplied files

- Source: 124 decoded video frames at 24 FPS, 704x704. Continuation: 245 decoded frames, while its checkpoint records 247 exported frames.
- Both checkpoint SHA-256 values match their corresponding MP4 bytes. Both files use format v2 and plugin 0.3.1.
- Continuation mode is `latent`, join mode `baseline`. The source was captured from a normal initial generation.
- Both raw video tensors have shape [1,24,37,44,44], dtype FP32. Audio tensors are [1,32,2,207], FP32. All values are finite. Global video latent standard deviations are about 1.020 and 1.019; no gross numerical explosion is present. This does not prove perceptual correctness.
- Both renders use the same declared H3 pruned weights, VAEs, prompt, seed, ten Euler steps and shift 12.
- MP4 metadata declares native overlap=1, Spectrum (multiplier 0.08), and a Turbo LoRA. The older checkpoint's resolved LoRA fields are null, but that is not evidence of no LoRA: the MP4's `activated_loras` field is populated. New checkpoints also retain those native UI fields.
- With that source length and overlap=1, the previous selector returned video block 36 at frame position -3. With 18 frames of history it returns blocks 31–36 at positions -20, -16, -12, -8, -4, -3. Original tensor values are reused unchanged.
- A contact-sheet inspection shows brighter/glossier facial rendering and altered skin texture in the generated continuation.

## Interpretation and change

The uploaded continuation did not silently fall back to standard pixel encoding. Insufficient temporal context is a concrete weakness in the previous Reference path for overlap=1. Version 0.3.2 introduces a minimum 18-frame history, bounded by the available source segment, while preserving the user's assembly overlap. This is a hypothesis-driven comparative fix. These files alone cannot establish that history length is the only cause, nor quantify the effects of Spectrum, Turbo or other runtime plugins.

CPU tests verify exact selected tensors, positions, audio interval, unchanged assembly duration, native wrapper execution and no AV encoder calls. They cannot establish the perceptual quality of a real H3 result.

## Separate output timeline issue

The continued MP4 contains 245 frames, while the pre-mux export recorded 247 in the checkpoint. The source MP4 and checkpoint both record 124. The native mux path uses FFmpeg `-shortest`, which is a plausible cause of an end truncation; the supplied files do not prove at which export stage frames disappeared. The checkpoint hashes validate byte identity, not this temporal discrepancy.

The continued MP4's decoded last frame differs more from its checkpoint's auxiliary frame than the source pair does (mean absolute RGB difference roughly 5.45 vs 1.64 on a 0–255 scale). Compression and motion also affect these numbers. They are not a quality score.

Do not use the already degraded continuation as the next reference for this comparison. Retry from the intact original source pair. Version 0.3.2 does not rewrite the uploaded checkpoints or claim to correct post-mux trimming. Matching saved latent endpoints to final muxed frame counts remains a separate unresolved limitation for repeated continuation.

## Suggested comparison

First retry Reference in 0.3.2 from the original source pair with the same prompt, seed, LoRA and Spectrum settings. The console should report six video latent blocks, context=18 frames, assembly overlap=1. Compare the new segment directly with the supplied failed continuation. If quality remains poor, compare another run with No Skipping while keeping the remaining settings fixed. A GPU test is necessary before calling this regression resolved.
