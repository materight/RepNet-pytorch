"""PyTorch implementation of RepNet."""
import torch
from torch import nn
from typing import Tuple


# List of ResNet50V2 conv layers that uses bias in the tensorflow implementation
CONVS_WITH_BIAS = [
    'stem.conv',
    'stages.0.blocks.0.downsample.conv', 'stages.0.blocks.0.conv3', 'stages.0.blocks.1.conv3', 'stages.0.blocks.2.conv3',
    'stages.1.blocks.0.downsample.conv', 'stages.1.blocks.0.conv3', 'stages.1.blocks.1.conv3', 'stages.1.blocks.2.conv3', 'stages.1.blocks.3.conv3',
    'stages.2.blocks.0.downsample.conv', 'stages.2.blocks.0.conv3', 'stages.2.blocks.1.conv3', 'stages.2.blocks.2.conv3', 
]

# List of ResNet50V2 conv layers that uses stride 1 in the tensorflow implementation
CONVS_WITHOUT_STRIDE = [
    'stages.1.blocks.0.downsample.conv', 'stages.1.blocks.0.conv2',
    'stages.2.blocks.0.downsample.conv', 'stages.2.blocks.0.conv2',
]

# List of ResNet50V2 conv layers that use max pooling instead of stride 2 in the tensorflow implementation
FINAL_BLOCKS_WITH_MAX_POOL = [
    'stages.0.blocks.2', 'stages.1.blocks.3',
]


class RepNet(nn.Module):
    """RepNet model."""
    def __init__(self, num_frames: int = 64, temperature: float = 13.544):
        super().__init__()
        self.num_frames = num_frames
        self.temperature = temperature
        self.encoder = self._init_encoder()
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(1024, 512, kernel_size=3, dilation=(3, 1, 1), padding=(3, 1, 1)),
            nn.BatchNorm3d(512, eps=0.001),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool3d((None, 1, 1)),
            nn.Flatten(2, 4),
        )
        self.tsm_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.period_length_head = self._init_transformer_head(num_frames, 2048, 4, 512, num_frames // 2)
        self.periodicity_head = self._init_transformer_head(num_frames, 2048, 4, 512, 1)


    @staticmethod
    def _init_encoder() -> nn.Module:
        """Initialize the encoder network using ResNet50 V2."""
        encoder = torch.hub.load('huggingface/pytorch-image-models', 'resnetv2_50')
        # Remove unused layers
        del encoder.stages[2].blocks[3:6], encoder.stages[3]
        encoder.norm = nn.Identity()
        encoder.head.global_pool = nn.Identity()
        encoder.head.fc = nn.Identity()
        encoder.head.flatten = nn.Identity()
        # Change padding from -inf to 0 on max pool to have the same behavior as tensorflow
        encoder.stem.pool.padding = 0
        encoder.stem.pool = nn.Sequential(nn.ZeroPad2d((1, 1, 1, 1)), encoder.stem.pool)
        # Change properties of existing layers
        for name, module in encoder.named_modules():
            # Add missing bias to conv layers
            if name in CONVS_WITH_BIAS:
                module.bias = nn.Parameter(torch.zeros(module.out_channels))
            # Remove stride from the first block in the later stages
            if name in CONVS_WITHOUT_STRIDE:
                module.stride = (1, 1)
            # Change stride and add max pooling to final block
            if name in FINAL_BLOCKS_WITH_MAX_POOL:
                module.conv2.stride = (2, 2)
                module.downsample = nn.MaxPool2d(1, stride=2)
                # Change the forward function so that the input of max pooling is the raw `x` instead of the pre-activation result
                bound_method = _max_pool_block_forward.__get__(module, module.__class__)
                setattr(module, 'forward', bound_method)
            # Change eps in batchnorm layers
            if isinstance(module, nn.BatchNorm2d):
                module.eps = 1.001e-5
        return encoder


    @staticmethod
    def _init_transformer_head(num_frames: int, in_features: int, n_head: int, hidden_features: int, out_features: int) -> nn.Module:
        """Initialize the fully-connected head for the final output."""
        return nn.Sequential(
            TranformerLayer(in_features, n_head, hidden_features, num_frames),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, out_features),
        )


    def extract_feat(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the encoder network to extract per-frame embeddings. Expected input shape: N x C x D x H x W."""
        batch_size, _, seq_len, _, _ = x.shape
        torch._assert(seq_len == self.num_frames, f'Expected {self.num_frames} frames, got {seq_len}')
        # Extract features frame-by-frame
        x = x.movedim(1, 2).flatten(0, 1)
        x = self.encoder(x)
        x = x.unflatten(0, (batch_size, seq_len)).movedim(1, 2)
        # Temporal convolution
        x = self.temporal_conv(x)
        x = x.movedim(1, 2) # Convert to N x D x C
        return x


    def period_predictor(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the period predictor network from the extracted embeddings. Expected input shape: N x D x C."""
        batch_size, seq_len, _ = x.shape
        torch._assert(seq_len == self.num_frames, f'Expected {self.num_frames} frames, got {seq_len}')
        # Compute temporal self-similarity matrix
        x = torch.cdist(x, x)**2 # N x D x D
        x = -x / self.temperature
        x = x.softmax(dim=-1)
        # Conv layer on top of the TSM
        x = self.tsm_conv(x.unsqueeze(1))
        x = x.movedim(1, 3).reshape(batch_size, seq_len, -1) # Flatten channels into N x D x C
        # Final prediction heads
        period_length = self.period_length_head(x)
        periodicity = self.periodicity_head(x)
        return period_length, periodicity


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass. Expected input shape: N x C x D x H x W."""
        embeddings = self.extract_feat(x)
        period_length, periodicity = self.period_predictor(embeddings)
        return period_length, periodicity, embeddings


    @staticmethod
    def get_counts(raw_period_length: torch.Tensor, raw_periodicity: torch.Tensor, stride: int,
                   periodicity_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the final scores from the period length and periodicity predictions."""
        # Repeat the input to account for the stride
        raw_period_length = raw_period_length.repeat_interleave(stride, dim=0)
        raw_periodicity = raw_periodicity.repeat_interleave(stride, dim=0)
        # Compute the final scores in [0, 1]
        periodicity_score = torch.sigmoid(raw_periodicity).squeeze(-1)
        period_length_confidence, period_length = torch.max(torch.softmax(raw_period_length, dim=-1), dim=-1)
        # Remove the confidence for short periods and convert to the correct stride
        period_length_confidence[period_length < 2] = 0
        period_length = (period_length + 1) * stride
        periodicity_score = torch.sqrt(periodicity_score * period_length_confidence)
        # Generate the final counts and set them to 0 if the periodicity is too low
        period_count = 1 / period_length
        period_count[periodicity_score < periodicity_threshold] = 0
        period_length = 1 / (torch.mean(period_count) + 1e-6)
        period_count = torch.cumsum(period_count, dim=0)
        confidence = torch.mean(periodicity_score)
        return confidence, period_length, period_count, periodicity_score



class TranformerLayer(nn.Module):
    """A single transformer layer with self-attention and positional encoding."""

    def __init__(self, in_features: int, n_head: int, out_features: int, num_frames: int):
        super().__init__()
        self.input_projection = nn.Linear(in_features, out_features)
        self.pos_encoding = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, num_frames, 1)))
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=out_features, nhead=n_head, dim_feedforward=out_features, activation='relu',
            layer_norm_eps=1e-6, batch_first=True, norm_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, expected input shape: N x C x D."""
        x = self.input_projection(x)
        x = x + self.pos_encoding
        x = self.transformer_layer(x)
        return x



def _max_pool_block_forward(self, x):
    """ 
    Custom `forward` function for the last block of each stage in ResNetV2, to have the same behavior as tensorflow.
    Original implementation: https://github.com/huggingface/pytorch-image-models/blob/4b8cfa6c0a355a9b3cb2a77298b240213fb3b921/timm/models/resnetv2.py#L197
    """
    x_preact = self.norm1(x)
    shortcut = x
    if self.downsample is not None:
        shortcut = self.downsample(x) # Changed here from `x_preact` to `x`
    x = self.conv1(x_preact)
    x = self.conv2(self.norm2(x))
    x = self.conv3(self.norm3(x))
    x = self.drop_path(x)
    return x + shortcut
