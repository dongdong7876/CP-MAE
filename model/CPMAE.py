import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .attn import AttentionLayer
from .embed import IndependentDataEmbedding, PositionalEncoding1D


def _to_scale_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


# ==============================================================================
# 1. Random Masking
# ==============================================================================

class RandomPatchMasker(nn.Module):
    """
    Random masking generator.
    Returned mask uses:
        1 -> visible
        0 -> masked
    """

    def __init__(self, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x, force_mask=False, mask_ratio=None):
        batch_size, num_tokens = x.shape[0], x.shape[1]
        device = x.device

        effective_mask_ratio = self.mask_ratio if mask_ratio is None else mask_ratio

        if ((not self.training) and (not force_mask)) or num_tokens <= 1:
            return torch.ones(batch_size, num_tokens, device=device)

        num_masked = int(round(num_tokens * effective_mask_ratio))
        num_masked = min(max(1, num_masked), num_tokens - 1)
        num_visible = num_tokens - num_masked

        noise = torch.rand(batch_size, num_tokens, device=device)
        shuffled_indices = torch.argsort(noise, dim=1)
        visible_indices = shuffled_indices[:, :num_visible]

        visible_mask = torch.zeros(batch_size, num_tokens, device=device)
        visible_mask.scatter_(
            dim=1,
            index=visible_indices,
            src=torch.ones(batch_size, num_visible, device=device)
        )

        return visible_mask


# ==============================================================================
# 2. Decoder
# ==============================================================================

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=None)
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.ffn(x))
        return x


class MAEDecoder(nn.Module):
    def __init__(self, d_model, out_dim, num_layers=2):
        super().__init__()
        self.blocks = nn.ModuleList([DecoderLayer(d_model) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# ==============================================================================
# 3. Base Transformer Encoder
# ==============================================================================

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        for attn_layer in self.attn_layers:
            x, _ = attn_layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


# ==============================================================================
# 4. Time-Domain Branch
# ==============================================================================

class TimeDomainEncoder(nn.Module):
    """
    Single-scale time-domain branch.
    """

    def __init__(
            self,
            c_in,
            num_patches,
            d_model,
            e_layers,
            win_size,
            mask_ratio=0.75
    ):
        super().__init__()

        if win_size % num_patches != 0:
            raise ValueError(f"win_size={win_size} must be divisible by num_patches={num_patches}")

        self.num_patches = num_patches
        self.patch_size = win_size // num_patches
        self.mask_token = nn.Parameter(torch.randn(1, 1, c_in, self.patch_size))

        self.patch_embed = IndependentDataEmbedding(c_in, self.patch_size, d_model)

        self.encoder = Encoder(
            [AttentionLayer(d_model) for _ in range(e_layers)],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.masker = RandomPatchMasker(mask_ratio=mask_ratio)

        self.decoder = MAEDecoder(
            d_model=d_model,
            out_dim=self.patch_size,
            num_layers=2
        )

        self.decoder_pos_embed = PositionalEncoding1D(d_model, max_len=num_patches)

    def forward(self, x, force_mask=False, mask_ratio=None, visible_mask_override=None):
        batch_size, seq_len, num_channels = x.size()

        x_patches = x.unfold(dimension=1, size=self.patch_size, step=self.patch_size)
        x_patches = x_patches.reshape(batch_size, self.num_patches, num_channels, self.patch_size)

        mask_input = x_patches.reshape(batch_size, self.num_patches, -1)

        # Override mask for coverage-aware inference
        if visible_mask_override is not None:
            visible_mask = visible_mask_override.to(device=x.device, dtype=x.dtype)
        else:
            visible_mask = self.masker(
                mask_input,
                force_mask=force_mask,
                mask_ratio=mask_ratio
            )

        mask_tokens = self.mask_token.repeat(batch_size, self.num_patches, 1, 1)
        visible_mask_expanded = visible_mask.view(batch_size, self.num_patches, 1, 1)
        masked_patches = x_patches * visible_mask_expanded + mask_tokens * (1.0 - visible_mask_expanded)

        masked_patches = masked_patches.reshape(-1, num_channels, self.patch_size)
        embedded_tokens = self.patch_embed(masked_patches)
        encoded_tokens = self.encoder(embedded_tokens)

        decoded_input = rearrange(
            encoded_tokens,
            '(b n) c d -> (b c) n d',
            b=batch_size,
            n=self.num_patches
        )
        decoded_input = self.decoder_pos_embed(decoded_input)
        reconstructed_patches = self.decoder(decoded_input)

        reconstruction = rearrange(
            reconstructed_patches,
            '(b c) n p -> b (n p) c',
            b=batch_size,
            c=num_channels
        )

        return {
            "reconstruction": reconstruction,
            "visible_mask": visible_mask
        }


# ==============================================================================
# 5. Time-Frequency Branch
# ==============================================================================

class TimeFrequencyEncoder(nn.Module):
    """
    Single-scale time-frequency branch.
    """

    def __init__(
            self,
            c_in,
            num_patches,
            d_model,
            e_layers,
            win_size,
            mask_ratio=0.5
    ):
        super().__init__()

        if win_size % num_patches != 0:
            raise ValueError(f"win_size={win_size} must be divisible by num_patches={num_patches}")

        self.patch_size = win_size // num_patches
        self.num_patches = win_size // self.patch_size
        self.hop_length = self.patch_size // 2
        self.freq_bins = self.patch_size // 2 + 1

        self.feature_dim = c_in * self.freq_bins
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.feature_dim))

        self.token_embed = nn.Linear(self.feature_dim, d_model)

        self.encoder = Encoder(
            [AttentionLayer(d_model) for _ in range(e_layers)],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.masker = RandomPatchMasker(mask_ratio=mask_ratio)

        self.decoder = MAEDecoder(
            d_model=d_model,
            out_dim=self.feature_dim,
            num_layers=2
        )

        expected_frames = win_size // self.hop_length + 1
        self.encoder_pos_embed = PositionalEncoding1D(d_model, max_len=expected_frames)
        # self.decoder_pos_embed = PositionalEncoding1D(d_model, max_len=expected_frames)

        self.register_buffer(
            "stft_window",
            torch.hann_window(self.patch_size),
            persistent=False
        )

    def compute_spectrogram_tokens(self, x):
        batch_size, seq_len, num_channels = x.shape
        x_reshaped = rearrange(x, 'b t c -> (b c) t')

        window = self.stft_window.to(device=x.device, dtype=x.dtype)

        stft_output = torch.stft(
            x_reshaped,
            n_fft=self.patch_size,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
            center=True
        )

        stft_output = rearrange(stft_output, '(b c) f t -> b c f t', b=batch_size)
        magnitude = stft_output.abs()
        magnitude_tokens = rearrange(magnitude, 'b c f t -> b t (c f)')

        return magnitude_tokens, magnitude

    def forward(self, x, force_mask=False, mask_ratio=None, visible_mask_override=None):
        batch_size, seq_len, num_channels = x.shape

        magnitude_tokens, magnitude = self.compute_spectrogram_tokens(x)
        num_tf_tokens = magnitude_tokens.shape[1]

        # Override mask for coverage-aware inference
        if visible_mask_override is not None:
            visible_mask = visible_mask_override.to(device=x.device, dtype=x.dtype)
        else:
            visible_mask = self.masker(
                magnitude_tokens,
                force_mask=force_mask,
                mask_ratio=mask_ratio
            )

        visible_mask_expanded = visible_mask.unsqueeze(-1)
        mask_tokens = self.mask_token.repeat(batch_size, num_tf_tokens, 1)

        masked_magnitude_tokens = (
                magnitude_tokens * visible_mask_expanded +
                mask_tokens * (1.0 - visible_mask_expanded)
        )
        magnitude_target = magnitude_tokens.detach()

        embedded_tokens = self.token_embed(masked_magnitude_tokens)
        embedded_tokens = self.encoder_pos_embed(embedded_tokens)
        encoded_tokens = self.encoder(embedded_tokens)

        # encoded_tokens = self.decoder_pos_embed(encoded_tokens)
        reconstruction = self.decoder(encoded_tokens)

        return {
            "reconstruction": reconstruction,
            "target": magnitude_target,
            "visible_mask": visible_mask
        }


# ==============================================================================
# 6. Multi-Scale Wrappers
# ==============================================================================

class MultiScaleTimeDomainEncoder(nn.Module):
    def __init__(self, c_in, num_patches_list, d_model, e_layers, win_size, mask_ratio=0.75):
        super().__init__()
        self.num_patches_list = _to_scale_list(num_patches_list)
        self.branches = nn.ModuleList([
            TimeDomainEncoder(
                c_in=c_in, num_patches=n_patch, d_model=d_model,
                e_layers=e_layers, win_size=win_size, mask_ratio=mask_ratio
            )
            for n_patch in self.num_patches_list
        ])

    def forward(self, x, force_mask=False, mask_ratio=None, visible_masks_override=None):
        outputs = []
        for i, branch in enumerate(self.branches):
            mask_override = None if visible_masks_override is None else visible_masks_override[i]
            out = branch(x, force_mask=force_mask, mask_ratio=mask_ratio, visible_mask_override=mask_override)
            outputs.append(out)
        return outputs


class MultiScaleTimeFrequencyEncoder(nn.Module):
    def __init__(self, c_in, num_patches_list, d_model, e_layers, win_size, mask_ratio=0.5):
        super().__init__()
        self.num_patches_list = _to_scale_list(num_patches_list)
        self.branches = nn.ModuleList([
            TimeFrequencyEncoder(
                c_in=c_in, num_patches=n_patch, d_model=d_model,
                e_layers=e_layers, win_size=win_size, mask_ratio=mask_ratio
            )
            for n_patch in self.num_patches_list
        ])

    def forward(self, x, force_mask=False, mask_ratio=None, visible_masks_override=None):
        outputs = []
        for i, branch in enumerate(self.branches):
            mask_override = None if visible_masks_override is None else visible_masks_override[i]
            out = branch(x, force_mask=force_mask, mask_ratio=mask_ratio, visible_mask_override=mask_override)
            outputs.append(out)
        return outputs


# ==============================================================================
# 7. Main Model: Strict Informational Bottleneck & Consistency-aware Inference
# ==============================================================================

class CPMAE(nn.Module):
    """
    Multi-Scale Dual-Domain Contextual Predictability Masked Autoencoder.
    """

    def __init__(
            self,
            win_size,
            n_features,
            num_patches,
            num_patches_tf=None,
            d_model=64,
            e_layers=3,
            alpha=1.0,
            beta=1.0,
            dev=None,
            st_mask_ratio=0.75,
            tf_mask_ratio=0.5,
            mc_samples=8,
            mc_mask_ratio_time=1.0,
            mc_mask_ratio_freq=1.0,
            uncertainty_weight=1.0
    ):
        super().__init__()
        self.device = dev
        self.alpha = alpha
        self.beta = beta

        self.default_mc_samples = mc_samples
        self.default_mc_mask_ratio_time = mc_mask_ratio_time
        self.default_mc_mask_ratio_freq = mc_mask_ratio_freq
        self.uncertainty_weight = uncertainty_weight

        self.num_patches_list = _to_scale_list(num_patches)
        if num_patches_tf is None:
            self.num_patches_tf_list = self.num_patches_list
        else:
            self.num_patches_tf_list = _to_scale_list(num_patches_tf)

        self.multi_time_branch = MultiScaleTimeDomainEncoder(
            c_in=n_features, num_patches_list=self.num_patches_list,
            d_model=d_model, e_layers=e_layers, win_size=win_size, mask_ratio=st_mask_ratio
        )

        self.multi_freq_branch = MultiScaleTimeFrequencyEncoder(
            c_in=n_features, num_patches_list=self.num_patches_tf_list,
            d_model=d_model, e_layers=e_layers, win_size=win_size, mask_ratio=tf_mask_ratio
        )

    # --------------------------------------------------------------------------
    # Coverage-aware MC visible-mask generation
    # Ensures no temporal region is consistently left unmasked across trials
    # --------------------------------------------------------------------------
    def _generate_coverage_aware_visible_masks(self, batch_size, num_tokens, mc_samples, mask_ratio, device):
        if mc_samples < 1:
            raise ValueError(f"mc_samples must be >= 1, got {mc_samples}.")

        if num_tokens <= 1:
            ones = torch.ones(batch_size, num_tokens, device=device)
            return [ones for _ in range(mc_samples)]

        num_masked = int(round(num_tokens * mask_ratio))
        num_masked = min(max(1, num_masked), num_tokens - 1)
        total_mask_slots = mc_samples * num_masked
        coverage_possible = total_mask_slots >= num_tokens

        visible_masks = torch.ones(batch_size, mc_samples, num_tokens, device=device)

        for b in range(batch_size):
            if coverage_possible:
                perm = torch.randperm(num_tokens, device=device)
                assigned = [perm[k::mc_samples] for k in range(mc_samples)]

                for k in range(mc_samples):
                    forced_mask_idx = assigned[k]
                    if forced_mask_idx.numel() > 0:
                        visible_masks[b, k, forced_mask_idx] = 0.0

                    remaining_to_mask = num_masked - forced_mask_idx.numel()
                    if remaining_to_mask > 0:
                        candidates = torch.nonzero(visible_masks[b, k] > 0.5, as_tuple=False).flatten()
                        rand_order = torch.randperm(candidates.numel(), device=device)
                        extra_mask_idx = candidates[rand_order[:remaining_to_mask]]
                        visible_masks[b, k, extra_mask_idx] = 0.0
            else:
                for k in range(mc_samples):
                    perm = torch.randperm(num_tokens, device=device)
                    mask_idx = perm[:num_masked]
                    visible_masks[b, k, mask_idx] = 0.0

        return [visible_masks[:, k, :] for k in range(mc_samples)]

    # --------------------------------------------------------------------------
    # Training losses (Strict Informational Bottleneck using MSE & L1)
    # --------------------------------------------------------------------------
    def _compute_time_branch_loss(self, x, time_results):
        time_visible_mask = time_results["visible_mask"]
        time_loss_mask = 1.0 - time_visible_mask

        time_recon_error = (time_results["reconstruction"] - x) ** 2
        time_recon_error = rearrange(
            time_recon_error,
            'b (n p) c -> b n (p c)',
            n=time_visible_mask.shape[1]
        )
        time_recon_error = time_recon_error.mean(dim=-1)

        time_recon_loss = (
                (time_recon_error * time_loss_mask).sum() /
                (time_loss_mask.sum() + 1e-8)
        )
        return time_recon_loss

    def _compute_freq_branch_loss(self, freq_results):
        freq_visible_mask = freq_results["visible_mask"]
        freq_loss_mask = 1.0 - freq_visible_mask

        freq_recon_error = torch.abs(freq_results["reconstruction"] - freq_results["target"])
        freq_recon_error = freq_recon_error.mean(dim=-1)

        freq_recon_loss = (
                (freq_recon_error * freq_loss_mask).sum() /
                (freq_loss_mask.sum() + 1e-8)
        )
        return freq_recon_loss

    def forward_train(self, x):
        time_results_list = self.multi_time_branch(x, force_mask=False, mask_ratio=None)
        freq_results_list = self.multi_freq_branch(x, force_mask=False, mask_ratio=None)

        time_losses = []
        for time_results in time_results_list:
            time_losses.append(self._compute_time_branch_loss(x, time_results))
        time_recon_loss = torch.stack(time_losses).mean()

        freq_losses = []
        for freq_results in freq_results_list:
            freq_losses.append(self._compute_freq_branch_loss(freq_results))
        freq_recon_loss = torch.stack(freq_losses).mean()

        total_loss = self.alpha * time_recon_loss + self.beta * freq_recon_loss

        return {
            "loss": total_loss,
            "loss_time_recon": time_recon_loss,
            "loss_freq_recon": freq_recon_loss,
        }

    # --------------------------------------------------------------------------
    # Inference: Consistency-aware Anomaly Scoring
    # Computes Reconstruction Inconsistency (Mean) & Predictive Instability (Var)
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def _mc_time_branch_error(self, x, branch, mc_samples, mc_mask_ratio):
        batch_size, seq_len, num_channels = x.shape
        patch_size = branch.patch_size
        num_time_patches = branch.num_patches

        visible_masks = self._generate_coverage_aware_visible_masks(
            batch_size=batch_size, num_tokens=num_time_patches, mc_samples=mc_samples,
            mask_ratio=mc_mask_ratio, device=x.device
        )

        # Initialize Point-wise tensors
        time_sum = torch.zeros(batch_size, seq_len, device=x.device)
        time_sum2 = torch.zeros(batch_size, seq_len, device=x.device)
        time_count = torch.zeros(batch_size, seq_len, device=x.device)

        for visible_mask in visible_masks:
            results = branch(
                x,
                force_mask=True,
                mask_ratio=mc_mask_ratio,
                visible_mask_override=visible_mask
            )

            # Map Patch mask to Point mask
            time_visible_mask = results["visible_mask"]
            time_masked_patch = 1.0 - time_visible_mask
            time_masked_point = time_masked_patch.repeat_interleave(patch_size, dim=1)

            # Expected Error (Point-wise MSE over Channels)
            time_recon_error = (results["reconstruction"] - x) ** 2
            time_recon_error = time_recon_error.mean(dim=-1)

            time_sum += time_recon_error * time_masked_point
            time_sum2 += (time_recon_error ** 2) * time_masked_point
            time_count += time_masked_point

        time_count_safe = torch.clamp(time_count, min=1.0)
        time_mean = time_sum / time_count_safe

        # Calculate Epistemic Uncertainty (Variance)
        time_var = torch.clamp(time_sum2 / time_count_safe - time_mean ** 2, min=0.0)
        time_std = torch.sqrt(time_var + 1e-8)

        return time_mean, time_std

    @torch.no_grad()
    def _mc_freq_branch_error(self, x, branch, mc_samples, mc_mask_ratio, target_len):
        batch_size = x.shape[0]
        target_tokens, _ = branch.compute_spectrogram_tokens(x)
        num_tokens = target_tokens.shape[1]

        visible_masks = self._generate_coverage_aware_visible_masks(
            batch_size=batch_size, num_tokens=num_tokens, mc_samples=mc_samples,
            mask_ratio=mc_mask_ratio, device=x.device
        )

        freq_sum = torch.zeros(batch_size, target_len, device=x.device)
        freq_sum2 = torch.zeros(batch_size, target_len, device=x.device)
        freq_count = torch.zeros(batch_size, target_len, device=x.device)

        for visible_mask in visible_masks:
            results = branch(
                x,
                force_mask=True,
                mask_ratio=mc_mask_ratio,
                visible_mask_override=visible_mask
            )
            freq_masked = 1.0 - results["visible_mask"]

            # Token-level absolute error
            freq_recon_error = torch.abs(results["reconstruction"] - results["target"]).mean(dim=-1)

            # Interpolate error and mask to match original sequence length
            error_interp = F.interpolate(
                freq_recon_error.unsqueeze(1),
                size=target_len,
                mode='linear',
                align_corners=False
            ).squeeze(1)

            mask_interp = F.interpolate(
                freq_masked.unsqueeze(1),
                size=target_len,
                mode='linear',
                align_corners=False
            ).squeeze(1)

            freq_sum += error_interp * mask_interp
            freq_sum2 += (error_interp ** 2) * mask_interp
            freq_count += mask_interp

        freq_count_safe = torch.clamp(freq_count, min=1.0)
        freq_mean = freq_sum / freq_count_safe

        # Calculate Epistemic Uncertainty (Variance)
        freq_var = torch.clamp(freq_sum2 / freq_count_safe - freq_mean ** 2, min=0.0)
        freq_std = torch.sqrt(freq_var + 1e-8)

        return freq_mean, freq_std

    @torch.no_grad()
    def predict_anomaly_score_mc(
            self,
            x,
            mc_samples=None,
            mc_mask_ratio_time=None,
            mc_mask_ratio_freq=None,
            uncertainty_weight=None
    ):
        mc_samples = self.default_mc_samples if mc_samples is None else mc_samples
        mc_mask_ratio_time = self.default_mc_mask_ratio_time if mc_mask_ratio_time is None else mc_mask_ratio_time
        mc_mask_ratio_freq = self.default_mc_mask_ratio_freq if mc_mask_ratio_freq is None else mc_mask_ratio_freq
        uncertainty_weight = self.uncertainty_weight if uncertainty_weight is None else uncertainty_weight

        # ----- Time domain -----
        time_err_list, time_std_list = [], []

        for branch in self.multi_time_branch.branches:
            time_mean, time_std = self._mc_time_branch_error(
                x=x,
                branch=branch,
                mc_samples=mc_samples,
                mc_mask_ratio=mc_mask_ratio_time
            )
            time_err_list.append(time_mean)
            time_std_list.append(time_std)

        # ----- Frequency domain -----
        freq_err_list, freq_std_list = [], []

        target_len = x.shape[1]
        for branch in self.multi_freq_branch.branches:
            freq_mean, freq_std = self._mc_freq_branch_error(
                x=x,
                branch=branch,
                mc_samples=mc_samples,
                mc_mask_ratio=mc_mask_ratio_freq,
                target_len=target_len
            )
            freq_err_list.append(freq_mean)
            freq_std_list.append(freq_std)

        # ----- Multi-scale aggregation -----
        time_mean = torch.stack(time_err_list, dim=0).mean(dim=0)
        time_std = torch.stack(time_std_list, dim=0).mean(dim=0)

        freq_mean = torch.stack(freq_err_list, dim=0).mean(dim=0)
        freq_std = torch.stack(freq_std_list, dim=0).mean(dim=0)

        # ----- Cross-domain Sum Fusion -----
        norm_time_mean = torch.log1p(time_mean)
        norm_freq_mean = torch.log1p(freq_mean)
        fused_mean = self.alpha * norm_time_mean + self.beta * norm_freq_mean

        norm_time_std = torch.log1p(time_std)
        norm_freq_std = torch.log1p(freq_std)
        fused_std = self.alpha * norm_time_std + self.beta * norm_freq_std

        # Consistency-aware Score = Expected Error + w * Epistemic Standard Deviation
        fused_error = fused_mean
        score = fused_mean + uncertainty_weight * fused_std

        return {
            "err_recon": fused_error,
            "err_time_recon": norm_time_mean,
            "err_freq_recon": norm_freq_mean,
            'norm_time_std': norm_time_std,
            'norm_freq_std': norm_freq_std,
            "err_uncertainty": fused_std,
            "score": score
        }

    def forward(
            self,
            x,
            mc_samples=None,
            mc_mask_ratio_time=None,
            mc_mask_ratio_freq=None,
            uncertainty_weight=None
    ):
        if self.training:
            return self.forward_train(x)

        return self.predict_anomaly_score_mc(
            x,
            mc_samples=mc_samples,
            mc_mask_ratio_time=mc_mask_ratio_time,
            mc_mask_ratio_freq=mc_mask_ratio_freq,
            uncertainty_weight=uncertainty_weight
        )