"""Seamless end premise, checked on WanGP's actual H3 video VAE decoder code (random weights, small sizes).
Usage: python tests/vae_seam_check.py /path/to/Wan2GP   (needs diffusers + accelerate; attention replaced by SDPA)
For a 243-frame segment and an exact join pinning 22 frames, the window's decode of the pinned latents must equal the
source decode on frames 226..238 and differ only on the source's last 4 frames (decoded without their future).
"""
import sys, types, importlib.util, torch
root = sys.argv[1]; sys.path.insert(0, root)
for name, path in (('models', 'models'), ('models.minimax_h3', 'models/minimax_h3'), ('models.minimax_h3.components', 'models/minimax_h3/components')):
    m = types.ModuleType(name); m.__path__ = [f'{root}/{path}']; sys.modules[name] = m
attention = types.ModuleType('shared.attention')
def pay_attention(qkv_list, causal=False, recycle_q=False, **kw):
    q, k, v = (x.float().transpose(1, 2) for x in qkv_list)   # (B, L, H, D) -> (B, H, L, D)
    qkv_list.clear()
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2)
attention.pay_attention = pay_attention; sys.modules['shared.attention'] = attention
# Wan2GP 19.x: the VAE goes through shared.attention_kit; no SageAttention staging here (SDPA path).
attention.sage_staged = lambda *args, **kwargs: None
attention.sage_attention_mask = lambda attention_mask, *args, **kwargs: attention_mask
spec = importlib.util.spec_from_file_location('models.minimax_h3.components.video_autoencoder', f'{root}/models/minimax_h3/components/video_autoencoder.py')
mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
torch.manual_seed(0)
vae = mod.AutoencoderKLMiniMaxH3(block_out_channels=(32,)*6, layers_per_block=1, norm_num_groups=8, decoder_num_layers=2,
                                 decoder_num_attention_heads=2, decoder_attention_head_dim=16).eval()
vae.use_tiling = False
with torch.no_grad():  # activate every residual branch (zero-initialised gains) so attention mixes all tokens
    for name, param in vae.decoder.named_parameters():
        if 'scale' in name or 'register' in name:
            param.copy_(torch.randn_like(param) * 0.5)
L = 72
src = torch.randn(1, 24, L, 2, 2)
window = torch.cat([src[:, :, L - 7:], torch.randn(1, 24, 30, 2, 2)], dim=2)   # 22 pinned frames + new latents
with torch.no_grad():
    base, other = vae._decode(src).float(), vae._decode(window).float()
diff = {f: float((base[:, :, f] - other[:, :, f - 221]).abs().max()) for f in range(221, 243)}
identical = [f for f, d in diff.items() if d < 1e-5]
assert identical == list(range(226, 239)), identical
assert all(diff[f] > 1e-3 for f in range(239, 243))
print('PASS: real H3 decoder: frames 226..238 identical in the source and exact-join decodes, 239..242 re-decoded '
      '-> the seamless end switches between identical frames (237 -> 238) and replaces the last 5.')

# Phase-1 re-encode premise: the encoder is causal inside a 17-frame chunk, so with keep_all_latents the first K latents
# of a pinned run depend only on the pinned frames, whatever pads the chunk after them.
with torch.no_grad():
    vae.use_tiling = False
    for frames, count in ((9, 3), (22, 7), (5, 2)):
        pinned = torch.randn(1, 3, frames, 32, 32)
        padded = torch.cat([pinned, pinned[:, :, -1:].repeat(1, 1, (-frames) % 17, 1, 1)], dim=2)        # native padding
        other = torch.cat([pinned, torch.randn(1, 3, (-frames) % 17, 32, 32)], dim=2)                       # real future instead
        enc = lambda x: torch.cat([vae._encode_clip(x[:, :, s:s + 17]) for s in range(0, x.shape[2], 17)], dim=2)
        a, b = enc(padded)[:, :, :count], enc(other)[:, :, :count]
        assert torch.allclose(a, b, atol=1e-5), (frames, float((a - b).abs().max()))
        if frames % 17 == 5:   # 5+17n: identical to the regular encode (token_drop) used by the source sequence
            assert torch.allclose(vae._encode(pinned)[:, :, :count], a, atol=1e-5)
print('PASS: real H3 encoder: the first latents of a pinned run do not depend on the chunk padding (keep_all_latents), '
      'and equal the regular encode for 5+17n frames.')
