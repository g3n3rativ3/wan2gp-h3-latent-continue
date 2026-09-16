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
