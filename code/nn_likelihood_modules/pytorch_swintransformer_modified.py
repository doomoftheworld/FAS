from common_imports import *
from typing import Any, Callable, List, Optional
from torchvision.models.swin_transformer import PatchMerging, SwinTransformerBlock, PatchMergingV2, SwinTransformerBlockV2
from torchvision.ops import Permute
from torchvision.utils import _log_api_usage_once
from functools import partial

class SwinTransformer_Modified(nn.Module):
    """
    Implements Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows" <https://arxiv.org/abs/2103.14030>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
    """

    def __init__(
        self,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        downsample_layer: Callable[..., nn.Module] = PatchMerging,
        hidden_size: int = 256,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.num_classes = num_classes

        if block is None:
            block = SwinTransformerBlock
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        layers: List[nn.Module] = []
        # split image into non-overlapping patches
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    3, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
                ),
                Permute([0, 2, 3, 1]),
                norm_layer(embed_dim),
            )
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.Sequential(*layers)

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)
        self.permute = Permute([0, 3, 1, 2])  # B H W C -> B C H W
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        # Old code outputting directy the predicted class
        # self.head = nn.Linear(num_features, num_classes)
        self.nb_hidden_layers = 1
        self.hidden_size = hidden_size
        self.head_hidden = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
        )
        self.head_out = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
        )
        # Parameter for registering activation levels
        self.regist_actLevel = False

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def activate_registration(self):
        self.regist_actLevel = True
        
    def deactivate_registration(self):
        self.regist_actLevel = False

    def forward(self, x):
        if self.regist_actLevel:
            actLevel = []
            x = self.features(x)
            x = self.norm(x)
            x = self.permute(x)
            x = self.avgpool(x)
            x = self.flatten(x)
            x = self.head_hidden(x)
            # Registration of the activation levels before the last prediction layer
            actLevel.append(x)
            x = self.head_out(x)
            return x, actLevel
        else:
            x = self.features(x)
            x = self.norm(x)
            x = self.permute(x)
            x = self.avgpool(x)
            x = self.flatten(x)
            x = self.head_hidden(x)
            x = self.head_out(x)
            return x
        
def _swin_transformer_modified(
    patch_size: List[int],
    embed_dim: int,
    depths: List[int],
    num_heads: List[int],
    window_size: List[int],
    stochastic_depth_prob: float,
    **kwargs: Any,
) -> SwinTransformer_Modified:
    model = SwinTransformer_Modified(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs,
    )

    return model
    
def get_swin_v2_t_modified(**kwargs: Any) -> SwinTransformer_Modified:
    """
    Constructs a swin_v2_tiny architecture from
    `Swin Transformer V2: Scaling Up Capacity and 
    Resolution <https://arxiv.org/abs/2111.09883>` 
    with the modified head block.

    Args:
        **kwargs: parameters passed to the ``SwinTransformer_Modified``

    Note: Generally you provide the ``hidden_size`` to configure the number of neurons in the hidden layer.
    """

    return _swin_transformer_modified(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.2,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )

def get_swin_v2_s_modified(**kwargs: Any) -> SwinTransformer_Modified:
    """
    Constructs a swin_v2_small architecture from
    `Swin Transformer V2: Scaling Up Capacity and 
    Resolution <https://arxiv.org/abs/2111.09883>` 
    with the modified head block.

    Args:
        **kwargs: parameters passed to the ``SwinTransformer_Modified``

    Note: Generally you provide the ``hidden_size`` to configure the number of neurons in the hidden layer.
    """

    return _swin_transformer_modified(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.3,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )

def get_swin_v2_b_modified(**kwargs: Any) -> SwinTransformer_Modified:
    """
    Constructs a swin_v2_base architecture from
    `Swin Transformer V2: Scaling Up Capacity and 
    Resolution <https://arxiv.org/abs/2111.09883>`
    with the modified head block.

    Args:
        **kwargs: parameters passed to the ``SwinTransformer_Modified``

    Note: Generally you provide the ``hidden_size`` to configure the number of neurons in the hidden layer.
    """
    
    return _swin_transformer_modified(
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8],
        stochastic_depth_prob=0.5,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )

def get_swin_v2_l_modified(**kwargs: Any) -> SwinTransformer_Modified:
    """
    Constructs a swin_v2_large architecture from
    `Swin Transformer V2: Scaling Up Capacity and 
    Resolution <https://arxiv.org/abs/2111.09883>`
    with the modified head block.

    Args:
        **kwargs: parameters passed to the ``SwinTransformer_Modified``

    Note: Generally you provide the ``hidden_size`` to configure the number of neurons in the hidden layer.
    """
    
    return _swin_transformer_modified(
        patch_size=[4, 4],
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=[8, 8],
        stochastic_depth_prob=0.5,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )