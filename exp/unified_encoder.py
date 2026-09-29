"""
Rung L7: unified spatio-temporal encoder (R1-C15).

The decoupled design of CP-MAE runs self-attention over the C variables inside
the encoder and over the N patches inside the decoder. Reviewer 1 asks whether a
single encoder attending over all N*C tokens jointly would do better. This module
supplies exactly that control.

Everything except the attention factorisation is held fixed: the patching, the
masking, the channel-wise embedding, the decoder stack, the projection head and
the unpatching are inherited unchanged from `TimeDomainEncoder`. The encoder is
built from the project's own `AttentionLayer`, so the layer type is identical too.

Cost warning
------------
The unified encoder attends over N*C tokens, so its attention matrix is
(N*C) x (N*C) instead of C x C. On WADI at N = 8 and C = 127 that is 1016 tokens,
about 1.0M attention entries per sample per layer. Batch 512 will not fit in
32 GB. Use --batch 32 or 64 for L7 on WADI and WADI-sized inputs, and report the
batch size alongside the result.

Usage
-----
  import unified_encoder
  unified_encoder.install(model)          # swap every time-domain scale to L7
"""
from __future__ import annotations

# torch is imported lazily so that `python unified_encoder.py` can print the cost
# report on a machine without a deep-learning stack.


def build():
    """Import the project classes; the caller must have put the repo on sys.path."""
    from model.CPMAE import TimeDomainEncoder, Encoder
    from model.attn import AttentionLayer
    from model.embed import PositionalEncoding1D
    return TimeDomainEncoder, Encoder, AttentionLayer, PositionalEncoding1D


def make_class():
    import torch
    import torch.nn as nn
    from einops import rearrange
    TimeDomainEncoder, Encoder, AttentionLayer, PositionalEncoding1D = build()

    class UnifiedTimeDomainEncoder(TimeDomainEncoder):
        """Same masked autoencoder, one joint attention over N*C tokens."""

        def __init__(self, c_in, num_patches, d_model, e_layers, win_size, mask_ratio=0.75):
            super().__init__(c_in=c_in, num_patches=num_patches, d_model=d_model,
                             e_layers=e_layers, win_size=win_size, mask_ratio=mask_ratio)
            # replace the cross-channel encoder by a joint spatio-temporal one
            self.encoder = Encoder(
                [AttentionLayer(d_model) for _ in range(e_layers)],
                norm_layer=nn.LayerNorm(d_model),
            )
            # the joint sequence needs both coordinates, so a temporal code over the
            # N patches and a channel code over the C variables are summed in
            self.temporal_pos_embed = PositionalEncoding1D(d_model, max_len=num_patches)
            self.channel_pos_embed = nn.Parameter(torch.zeros(1, 1, c_in, d_model))
            nn.init.trunc_normal_(self.channel_pos_embed, std=0.02)

        def forward(self, x, force_mask=False, mask_ratio=None, visible_mask_override=None):
            batch_size, seq_len, num_channels = x.size()

            x_patches = x.unfold(dimension=1, size=self.patch_size, step=self.patch_size)
            x_patches = x_patches.reshape(batch_size, self.num_patches, num_channels,
                                          self.patch_size)
            mask_input = x_patches.reshape(batch_size, self.num_patches, -1)

            if visible_mask_override is not None:
                visible_mask = visible_mask_override.to(device=x.device, dtype=x.dtype)
            else:
                visible_mask = self.masker(mask_input, force_mask=force_mask,
                                           mask_ratio=mask_ratio)

            mask_tokens = self.mask_token.repeat(batch_size, self.num_patches, 1, 1)
            vis = visible_mask.view(batch_size, self.num_patches, 1, 1)
            masked_patches = x_patches * vis + mask_tokens * (1.0 - vis)

            flat = masked_patches.reshape(-1, num_channels, self.patch_size)
            emb = self.patch_embed(flat)                                   # [B*N, C, D]
            emb = emb.view(batch_size, self.num_patches, num_channels, -1)  # [B, N, C, D]

            # temporal code over N, broadcast across C; channel code over C
            t_code = self.temporal_pos_embed(
                torch.zeros(batch_size, self.num_patches, emb.shape[-1], device=x.device,
                            dtype=emb.dtype))
            emb = emb + t_code.unsqueeze(2) + self.channel_pos_embed.to(emb.dtype)

            joint = emb.reshape(batch_size, self.num_patches * num_channels, -1)
            joint = self.encoder(joint)                                    # [B, N*C, D]
            emb = joint.view(batch_size, self.num_patches, num_channels, -1)

            decoded_input = rearrange(emb, 'b n c d -> (b c) n d')
            decoded_input = self.decoder_pos_embed(decoded_input)
            reconstructed = self.decoder(decoded_input)
            reconstruction = rearrange(reconstructed, '(b c) n p -> b (n p) c',
                                       b=batch_size, c=num_channels)

            return {"reconstruction": reconstruction, "visible_mask": visible_mask}

    return UnifiedTimeDomainEncoder


def install(model, c_in, win_size, d_model, e_layers, mask_ratio):
    """Replace every time-domain scale of `model` with the unified variant."""
    import torch.nn as nn
    cls = make_class()
    device = next(model.parameters()).device
    new = nn.ModuleList()
    for branch in model.multi_time_branch.branches:
        new.append(cls(c_in=c_in, num_patches=branch.num_patches, d_model=d_model,
                       e_layers=e_layers, win_size=win_size,
                       mask_ratio=mask_ratio).to(device))
    model.multi_time_branch.branches = new
    return model


def token_budget(win_size, num_patches, channels):
    """Report the joint sequence length and the attention entries per sample."""
    n_tokens = num_patches * channels
    return n_tokens, n_tokens ** 2


if __name__ == "__main__":
    print("Joint attention cost of rung L7 (tokens, attention entries per sample per layer)")
    for ds, C in (("LTDB", 2), ("PSM", 25), ("SMD", 38), ("SWaT", 51), ("WADI", 127)):
        for N in (4, 8):
            n, a = token_budget(320, N, C)
            print(f"  {ds:<5} N={N}: {n:5d} tokens, {a/1e6:8.3f}M entries"
                  f"   (decoupled: {C*C/1e6:.3f}M + {N*N/1e6:.6f}M)")
