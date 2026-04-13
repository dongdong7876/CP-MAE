import torch
import torch.nn as nn

# ==============================================================================
# 1. RevIN
# ==============================================================================

class RevIN(nn.Module):
    """
    Reversible Instance Normalization.
    Here it is applied ONLY to the seasonal branch.

    x: [B, T, C]
    """
    def __init__(self, num_features, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

        self._cached_mean = None
        self._cached_stdev = None
        self._cached_last = None

    def _get_statistics(self, x):
        if self.subtract_last:
            last = x[:, -1:, :]
            x_centered = x - last
            mean = torch.zeros_like(last)
        else:
            mean = x.mean(dim=1, keepdim=True)
            x_centered = x - mean
            last = None

        stdev = torch.sqrt(torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + self.eps)
        return mean, stdev, last

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            mean, stdev, last = self._get_statistics(x)
            self._cached_mean = mean
            self._cached_stdev = stdev
            self._cached_last = last

            if self.subtract_last:
                x = x - last
            else:
                x = x - mean

            x = x / stdev

            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x

        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)

            x = x * self._cached_stdev

            if self.subtract_last:
                x = x + self._cached_last
            else:
                x = x + self._cached_mean
            return x

        else:
            raise ValueError(f"Unsupported RevIN mode: {mode}")

# ==============================================================================
# 2. Positional Encoding
# ==============================================================================

class PositionalEncoding1D(nn.Module):
    """
    1D Learnable Positional Encoding for temporal sequences.
    """
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        x: [Batch, Sequence_Length, d_model]
        """
        seq_len = x.size(1)
        return x + self.pos_embed[:, :seq_len, :]

# ==============================================================================
# 3. Independent Data Embedding
# ==============================================================================

class IndependentPatchEmbedding(nn.Module):
    """
    Project each patch of length 'patch_len' to a feature vector of 'd_model'.
    """

    def __init__(self, num_features, patch_len, d_model):
        super(IndependentPatchEmbedding, self).__init__()
        self.num_features = num_features

        # Use nn.Linear to project the patch_len to d_model for each feature independently
        self.embeddings = nn.ModuleList([
            nn.Linear(patch_len, d_model) for _ in range(num_features)
        ])

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [Batch, N_features, Patch_len]
        Returns:
            Output tensor of shape [Batch, N_features, d_model]
        """
        outputs = []
        for i in range(self.num_features):
            feat_data = x[:, i, :]  # [Batch, Patch_len]
            embed_feat = self.embeddings[i](feat_data)  # [Batch, d_model]
            outputs.append(embed_feat.unsqueeze(1))  # [Batch, 1, d_model]

        return torch.cat(outputs, dim=1)  # [Batch, N_features, d_model]


class IndependentDataEmbedding(nn.Module):
    """
    Wrapper for IndependentPatchEmbedding with Dropout.
    """

    def __init__(self, c_in, patch_len, d_model, dropout=0.05):
        super(IndependentDataEmbedding, self).__init__()

        self.independent_embedding = IndependentPatchEmbedding(
            num_features=c_in,
            patch_len=patch_len,
            d_model=d_model
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [Batch, c_in, patch_len]
        x = self.independent_embedding(x)  # -> [Batch, c_in, d_model]
        return self.dropout(x)


# ==============================================================================
# 4. Multi-kernel decomposition
# ==============================================================================

class MovingAvg(nn.Module):
    """
    Moving average block for time series trend extraction.
    """
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd."
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        """
        x: [B, T, C]
        """
        pad_len = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad_len, 1)
        end = x[:, -1:, :].repeat(1, pad_len, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class MultiKernelSeriesDecomp(nn.Module):
    """
    Multi-kernel decomposition with learnable sample-wise kernel weighting.

    x -> trend + seasonal
    """
    def __init__(self, c_in, kernel_sizes=(5, 13, 25), gate_hidden=64):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.moving_avgs = nn.ModuleList([MovingAvg(k, stride=1) for k in kernel_sizes])

        self.gate = nn.Sequential(
            nn.Linear(c_in * 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, len(kernel_sizes))
        )

    def forward(self, x):
        """
        x: [B, T, C]
        """
        # Multi-kernel trend candidates
        trend_candidates = [ma(x) for ma in self.moving_avgs]  # list of [B, T, C]
        trend_candidates = torch.stack(trend_candidates, dim=-1)  # [B, T, C, K]

        # Sample-wise kernel selection weights
        pooled_mean = x.mean(dim=1)                             # [B, C]
        pooled_std = x.std(dim=1, unbiased=False)               # [B, C]
        gate_in = torch.cat([pooled_mean, pooled_std], dim=-1)  # [B, 2C]

        weights = torch.softmax(self.gate(gate_in), dim=-1)     # [B, K]
        weights_expanded = weights[:, None, None, :]            # [B, 1, 1, K]

        trend = (trend_candidates * weights_expanded).sum(dim=-1)  # [B, T, C]
        seasonal = x - trend

        return seasonal, trend, weights


# ==============================================================================
# 5. Masking
# ==============================================================================

class RandomPatchMasker(nn.Module):
    """
    Random patch masking.
    Returned mask:
        1 -> visible
        0 -> masked
    """
    def __init__(self, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        """
        x: [B, N, ...]
        """
        batch_size, num_tokens = x.shape[0], x.shape[1]

        if (not self.training) or num_tokens <= 1:
            return torch.ones(batch_size, num_tokens, device=x.device)

        num_visible = int(round(num_tokens * (1.0 - self.mask_ratio)))
        num_visible = min(max(1, num_visible), num_tokens - 1)

        noise = torch.rand(batch_size, num_tokens, device=x.device)
        shuffled_indices = torch.argsort(noise, dim=1)
        visible_indices = shuffled_indices[:, :num_visible]

        visible_mask = torch.zeros(batch_size, num_tokens, device=x.device)
        visible_mask.scatter_(
            dim=1,
            index=visible_indices,
            src=torch.ones(batch_size, num_visible, device=x.device)
        )
        return visible_mask
