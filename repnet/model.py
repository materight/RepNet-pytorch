"""PyTorch implementation of RepNet."""
import torch
from torch import nn
from typing import Tuple


# List of ResNet50V2 conv layers that uses bias in the tensorflow implementation
RESNET_CONVS_WITH_BIAS = [
    'stem.conv',
    'stages.0.blocks.0.downsample.conv', 'stages.0.blocks.0.conv3', 'stages.0.blocks.1.conv3', 'stages.0.blocks.2.conv3',
    'stages.1.blocks.0.downsample.conv', 'stages.1.blocks.0.conv3', 'stages.1.blocks.1.conv3', 'stages.1.blocks.2.conv3', 'stages.1.blocks.3.conv3',
    'stages.2.blocks.0.downsample.conv', 'stages.2.blocks.0.conv3', 'stages.2.blocks.1.conv3', 'stages.2.blocks.2.conv3', 
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
            nn.BatchNorm3d(512),
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
        del encoder.stages[2].blocks[3:6]
        del encoder.stages[3]
        encoder.norm = nn.Identity()
        encoder.head.global_pool = nn.Identity()
        encoder.head.fc = nn.Identity()
        encoder.head.flatten = nn.Identity()
        # Change padding from -inf to 0 to have same beahvior as tensorflow
        encoder.stem.pool.padding = 0
        encoder.stem.pool = nn.Sequential(nn.ZeroPad2d((1, 1, 1, 1)), encoder.stem.pool)
        # Change properties of existing layers
        for name, module in encoder.named_modules():
            # Add missing bias to conv layers
            if name in RESNET_CONVS_WITH_BIAS:
                module.bias = nn.Parameter(torch.zeros(module.out_channels))
            # Change eps in batchnorm layers
            if isinstance(module, nn.BatchNorm2d):
                module.eps = 1.001e-5
        # Chage stride and add max pooling to final block to have same beahvior as tensorflow
        for stage in encoder.stages:
            stage.blocks[-1].conv2.stride = (2, 2)
            stage.blocks[-1].downsample = nn.MaxPool2d(1, stride=2)
            # Change the input of max pooling to the raw input, before the pre-act block
            graph = torch.fx.symbolic_trace(stage.blocks[-1]).graph
            raw_input = next(iter(graph.nodes))
            for node in graph.nodes:
                if node.target == 'downsample':
                    node.replace_input_with(node.all_input_nodes[0], raw_input)
            stage.blocks[-1] = torch.fx.GraphModule(stage.blocks[-1], graph)
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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Expected input shape: N x C x D x H x W."""
        batch_size, _, seq_len, _, _ = x.shape
        assert seq_len == self.num_frames, f'Expected {self.num_frames} frames, got {seq_len}'
        # Extract features frame-by-frame
        x = x.movedim(1, 2).flatten(0, 1)
        x = self.encoder(x)
        x = x.unflatten(0, (batch_size, seq_len)).movedim(1, 2)
        # Temporal convolution
        x = self.temporal_conv(x)
        x = x.movedim(1, 2) # Convert to N x D x C
        embeddings = x
        # Compute temporal self-similarity matrix
        x = torch.cdist(x, x) # N x D x D
        x = -x / self.temperature
        x = x.softmax(dim=1)
        # Conv layer on top of the TSM
        x = self.tsm_conv(x.unsqueeze(1))
        x = x.reshape(batch_size, seq_len, -1) # Flatten channels into N x D x C
        # Final prediction heads
        period_length = self.period_length_head(x)
        periodicity = self.periodicity_head(x)
        return period_length, periodicity, embeddings


    @staticmethod
    def get_scores(raw_period_length: torch.Tensor, raw_periodicity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the final scores from the period length and periodicity predictions."""
        periodicity = torch.sigmoid(raw_periodicity).squeeze(-1)
        raw_period_length = torch.softmax(raw_period_length, dim=-1)
        period_length_conf, period_length = torch.max(raw_period_length, dim=-1)
        period_length += 1
        return period_length, period_length_conf, periodicity




class TranformerLayer(nn.Module):
    """A single transformer layer with self-attention and positional encoding."""

    def __init__(self, in_features: int, n_head: int, out_features: int, num_frames: int):
        super().__init__()
        self.input_projection = nn.Linear(in_features, out_features)
        self.pos_encoding = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, num_frames, 1)))
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=out_features, nhead=n_head, dim_feedforward=out_features,
            batch_first=True, norm_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, expected input shape: N x C x D."""
        x = self.input_projection(x)
        x = x + self.pos_encoding
        x = self.transformer_layer(x)
        return x
