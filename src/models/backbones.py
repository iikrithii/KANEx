"""
Model Definitions for KANEx.

ResNet-50 variants (frozen backbone, head only trains):
  a) ResNet50Baseline     — FC head (2048 → num_classes)
  b) ResNet50VanillaKAN   — KANLinear head (B-splines, efficient-kan)
  c) ResNet50RationalKAN  — JacobiRKAN head (rational polynomial)
  d) ResNet50GroupKAN     — Grouped KAN head
  e) ResNet50ConvKAN      — KAN conv layer after layer4

ViT-Small/16 variants (frozen transformer blocks, head only trains):
  f) ViTSmallBaseline     — FC head (384 → num_classes)
  g) ViTSmallVanillaKAN   — two-layer KANLinear head
  h) ViTSmallGroupKAN     — GroupKANHead
  i) ViTSmallRationalKAN  — RationalKANLinear head

All models expose:
  - forward(x)               → (B, num_classes) logits
  - get_features(x)          → (B, C, H, W) spatial feature map
  - get_cam_target_layer()   → nn.Module to hook for Grad-CAM / Rollout
  - get_attention_blocks()   → nn.ModuleList  [ViT only, for Rollout]
  - is_kan_model()           → bool
  - is_vit_model()           → bool

ViT get_features() reshapes patch tokens to (B, 384, 14, 14) so KAN-CAM
works identically for ResNet and ViT KAN variants.
"""

import torch
import torch.nn as nn
import torchvision.models as models

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False

from ..layers.efficient_kan import KANLinear
from ..layers.rational_kan import RationalKANLinear
from ..layers.group_kan import GroupKANHead
from ..layers.conv_kan import KANConvNDLayer


# ── Shared hyperparameters ────────────────────────────────────────────────────
KAN_HIDDEN_DIM   = 512
KAN_GRID_SIZE    = 5
KAN_SPLINE_ORDER = 3
KAN_NUM_GROUPS   = 8
RKAN_DEGREE      = 3

# ViT-Small/16 constants
VIT_HIDDEN_DIM  = 384
VIT_PATCH_SIZE  = 16
VIT_GRID_SIZE   = 224 // VIT_PATCH_SIZE   # 14
VIT_N_PATCHES   = VIT_GRID_SIZE ** 2      # 196


# ── Backbone builders ─────────────────────────────────────────────────────────

def _build_resnet_backbone(freeze_backbone: bool = True):
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    feature_dim = resnet.fc.in_features  # 2048
    backbone = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
    )
    avgpool = resnet.avgpool
    if freeze_backbone:
        for p in backbone.parameters(): p.requires_grad = False
        for p in avgpool.parameters():  p.requires_grad = False
    last_conv = resnet.layer4[-1]
    return backbone, avgpool, feature_dim, last_conv


def _build_vit_backbone(freeze_backbone: bool = True):
    if not _TIMM_AVAILABLE:
        raise ImportError("timm is required for ViT models. pip install timm")
    vit = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
    if freeze_backbone:
        for p in vit.parameters(): p.requires_grad = False
    return vit


# ─────────────────────────────────────────────────────────────────────────────
# ResNet-50 variants
# ─────────────────────────────────────────────────────────────────────────────

class ResNet50Baseline(nn.Module):
    """ResNet-50 + FC head. Non-KAN CNN baseline."""

    def __init__(self, num_classes: int = 14, freeze_backbone: bool = True):
        super().__init__()
        self.model_name = "ResNet50_Baseline"
        self.backbone, self.avgpool, feature_dim, self.last_conv = \
            _build_resnet_backbone(freeze_backbone)
        self.head = nn.Linear(feature_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        return self.head(self.avgpool(self.backbone(x)).flatten(1))

    def get_cam_target_layer(self):
        return self.last_conv.conv3

    def get_features(self, x):
        return self.backbone(x)  # (B, 2048, 7, 7)

    def is_kan_model(self): return False
    def is_vit_model(self):  return False


class ResNet50VanillaKAN(nn.Module):
    """ResNet-50 + two-layer KANLinear head (B-splines)."""

    def __init__(self, num_classes: int = 14, freeze_backbone: bool = True):
        super().__init__()
        self.model_name = "ResNet50_VanillaKAN"
        self.backbone, self.avgpool, feature_dim, self.last_conv = \
            _build_resnet_backbone(freeze_backbone)
        self.kan1 = KANLinear(feature_dim,    KAN_HIDDEN_DIM, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
        self.kan2 = KANLinear(KAN_HIDDEN_DIM, num_classes,    grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
        self.num_classes = num_classes

    def forward(self, x):
        return self.kan2(self.kan1(self.avgpool(self.backbone(x)).flatten(1)))

    def get_cam_target_layer(self): return self.last_conv.conv3
    def get_features(self, x):      return self.backbone(x)

    def get_kan_feature_importance(self, x, class_idx=None):
        pooled = self.avgpool(self.backbone(x)).flatten(1).detach().requires_grad_(True)
        out = self.kan2(self.kan1(pooled))
        target = out[:, class_idx].sum() if class_idx is not None else out.abs().sum()
        return torch.autograd.grad(target, pooled)[0].abs()

    def get_kan_layers(self): return [self.kan1, self.kan2]
    def is_kan_model(self):   return True
    def is_vit_model(self):   return False


class ResNet50RationalKAN(nn.Module):
    """ResNet-50 + Rational KAN head (Jacobi polynomials)."""

    def __init__(self, num_classes: int = 14, freeze_backbone: bool = True):
        super().__init__()
        self.model_name = "ResNet50_RationalKAN"
        self.backbone, self.avgpool, feature_dim, self.last_conv = \
            _build_resnet_backbone(freeze_backbone)
        self.head = RationalKANLinear(
            in_features=feature_dim, hidden_dim=KAN_HIDDEN_DIM,
            num_classes=num_classes, degree=RKAN_DEGREE)
        self.num_classes = num_classes

    def forward(self, x):
        return self.head(self.avgpool(self.backbone(x)).flatten(1))

    def get_cam_target_layer(self): return self.last_conv.conv3
    def get_features(self, x):      return self.backbone(x)

    def get_kan_feature_importance(self, x, class_idx=None):
        pooled = self.avgpool(self.backbone(x)).flatten(1).detach().requires_grad_(True)
        out = self.head(pooled)
        target = out[:, class_idx].sum() if class_idx is not None else out.abs().sum()
        return torch.autograd.grad(target, pooled)[0].abs()

    def is_kan_model(self): return True
    def is_vit_model(self): return False


class ResNet50GroupKAN(nn.Module):
    """ResNet-50 + Grouped KAN head."""

    def __init__(self, num_classes: int = 14, freeze_backbone: bool = True):
        super().__init__()
        self.model_name = "ResNet50_GroupKAN"
        self.backbone, self.avgpool, feature_dim, self.last_conv = \
            _build_resnet_backbone(freeze_backbone)
        self.head = GroupKANHead(
            in_features=feature_dim, hidden_dim=KAN_HIDDEN_DIM,
            num_classes=num_classes, num_groups=KAN_NUM_GROUPS,
            grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
        self.num_classes = num_classes

    def forward(self, x):
        return self.head(self.avgpool(self.backbone(x)).flatten(1))

    def get_cam_target_layer(self): return self.last_conv.conv3
    def get_features(self, x):      return self.backbone(x)

    def get_kan_feature_importance(self, x, class_idx=None):
        pooled = self.avgpool(self.backbone(x)).flatten(1).detach().requires_grad_(True)
        out = self.head(pooled)
        target = out[:, class_idx].sum() if class_idx is not None else out.abs().sum()
        return torch.autograd.grad(target, pooled)[0].abs()

    def is_kan_model(self): return True
    def is_vit_model(self): return False


class ResNet50ConvKAN(nn.Module):
    """ResNet-50 + KAN conv layer after layer4."""

    def __init__(self, num_classes: int = 14, freeze_backbone: bool = True):
        super().__init__()
        self.model_name = "ResNet50_ConvKAN"
        self.backbone, self.avgpool, feature_dim, self.last_conv = \
            _build_resnet_backbone(freeze_backbone)
        self.kan_conv = KANConvNDLayer(
            in_channels=feature_dim, out_channels=KAN_HIDDEN_DIM,
            kernel_size=3, stride=1, padding=1,
            grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER, groups=8)
        self.kan_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(KAN_HIDDEN_DIM, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        kan_feats = self.kan_conv(self.backbone(x))
        return self.head(self.kan_pool(kan_feats).flatten(1))

    def get_cam_target_layer(self): return self.last_conv.conv3
    def get_features(self, x):      return self.backbone(x)

    def get_kan_spatial_importance(self, x):
        return self.kan_conv.get_spatial_spline_importance(self.backbone(x).detach())

    def is_kan_model(self):  return True
    def is_conv_kan(self):   return True
    def is_vit_model(self):  return False


# ─────────────────────────────────────────────────────────────────────────────
# ViT-Small/16 variants
# ─────────────────────────────────────────────────────────────────────────────

def _vit_patch_spatial(backbone, x, grid_size):
    """
    Helper: run timm ViT forward_features, drop CLS, reshape to (B, D, G, G).
    Used by get_features() in all ViT variants.
    """
    tokens  = backbone.forward_features(x)          # (B, 197, 384)
    patches = tokens[:, 1:, :]                        # (B, 196, 384)
    B, N, D = patches.shape
    return patches.permute(0, 2, 1).reshape(B, D, grid_size, grid_size)


class ViTSmallBaseline(nn.Module):
    """
    ViT-Small/16 + FC head. Non-KAN ViT baseline.

    Backbone frozen. Head: Linear(384 → num_classes).
    get_features() → (B, 384, 14, 14) for compatibility with Grad-CAM + KAN-CAM infra.
    get_attention_blocks() → all 12 transformer blocks for AttentionRollout.
    """

    PATCH_SIZE = VIT_PATCH_SIZE
    GRID_SIZE  = VIT_GRID_SIZE
    N_PATCHES  = VIT_N_PATCHES
    HIDDEN_DIM = VIT_HIDDEN_DIM

    def __init__(self, num_classes: int = 14, freeze_backbone: bool = True):
        super().__init__()
        self.model_name  = "ViT_Small_Baseline"
        self.num_classes = num_classes
        self.backbone  = _build_vit_backbone(freeze_backbone)
        self.last_block = self.backbone.blocks[-1]
        self.head = nn.Linear(self.HIDDEN_DIM, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))   # vit(x) returns CLS token (B, 384)

    def get_features(self, x):
        return _vit_patch_spatial(self.backbone, x, self.GRID_SIZE)

    def get_cam_target_layer(self):
        return self.backbone.blocks[-1]

    def get_attention_blocks(self):
        return self.backbone.blocks

    def is_kan_model(self): return False
    def is_vit_model(self):  return True


class ViTSmallVanillaKAN(nn.Module):
    """ViT-Small/16 + two-layer KANLinear head (B-splines)."""

    PATCH_SIZE = VIT_PATCH_SIZE
    GRID_SIZE  = VIT_GRID_SIZE
    N_PATCHES  = VIT_N_PATCHES
    HIDDEN_DIM = VIT_HIDDEN_DIM

    def __init__(self, num_classes: int = 14, freeze_backbone: bool = True):
        super().__init__()
        self.model_name  = "ViT_Small_VanillaKAN"
        self.num_classes = num_classes
        self.backbone  = _build_vit_backbone(freeze_backbone)
        self.last_block = self.backbone.blocks[-1]
        self.kan1 = KANLinear(self.HIDDEN_DIM, KAN_HIDDEN_DIM, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
        self.kan2 = KANLinear(KAN_HIDDEN_DIM,  num_classes,    grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)

    def forward(self, x):
        return self.kan2(self.kan1(self.backbone(x)))

    def get_features(self, x):
        return _vit_patch_spatial(self.backbone, x, self.GRID_SIZE)

    def get_cam_target_layer(self): return self.backbone.blocks[-1]
    def get_attention_blocks(self): return self.backbone.blocks

    def get_kan_feature_importance(self, x, class_idx=None):
        cls = self.backbone(x).detach().requires_grad_(True)
        out = self.kan2(self.kan1(cls))
        target = out[:, class_idx].sum() if class_idx is not None else out.abs().sum()
        return torch.autograd.grad(target, cls)[0].abs()

    def get_kan_layers(self): return [self.kan1, self.kan2]
    def is_kan_model(self):   return True
    def is_vit_model(self):   return True


class ViTSmallGroupKAN(nn.Module):
    """ViT-Small/16 + GroupKAN head."""

    PATCH_SIZE = VIT_PATCH_SIZE
    GRID_SIZE  = VIT_GRID_SIZE
    N_PATCHES  = VIT_N_PATCHES
    HIDDEN_DIM = VIT_HIDDEN_DIM

    def __init__(self, num_classes: int = 14, freeze_backbone: bool = True):
        super().__init__()
        self.model_name  = "ViT_Small_GroupKAN"
        self.num_classes = num_classes
        self.backbone  = _build_vit_backbone(freeze_backbone)
        self.last_block = self.backbone.blocks[-1]
        self.head = GroupKANHead(
            in_features=self.HIDDEN_DIM, hidden_dim=KAN_HIDDEN_DIM,
            num_classes=num_classes, num_groups=KAN_NUM_GROUPS,
            grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)

    def forward(self, x):
        return self.head(self.backbone(x))

    def get_features(self, x):
        return _vit_patch_spatial(self.backbone, x, self.GRID_SIZE)

    def get_cam_target_layer(self): return self.backbone.blocks[-1]
    def get_attention_blocks(self): return self.backbone.blocks

    def get_kan_feature_importance(self, x, class_idx=None):
        cls = self.backbone(x).detach().requires_grad_(True)
        out = self.head(cls)
        target = out[:, class_idx].sum() if class_idx is not None else out.abs().sum()
        return torch.autograd.grad(target, cls)[0].abs()

    def is_kan_model(self): return True
    def is_vit_model(self):  return True




class ViTSmallRationalKAN(nn.Module):
    """ViT-Small/16 + Rational KAN head (Jacobi polynomials)."""

    PATCH_SIZE = VIT_PATCH_SIZE
    GRID_SIZE  = VIT_GRID_SIZE
    N_PATCHES  = VIT_N_PATCHES
    HIDDEN_DIM = VIT_HIDDEN_DIM

    def __init__(self, num_classes: int = 14, freeze_backbone: bool = True):
        super().__init__()
        self.model_name  = "ViT_Small_RationalKAN"
        self.num_classes = num_classes
        self.backbone  = _build_vit_backbone(freeze_backbone)
        self.last_block = self.backbone.blocks[-1]
        self.head = RationalKANLinear(
            in_features=self.HIDDEN_DIM, hidden_dim=KAN_HIDDEN_DIM,
            num_classes=num_classes, degree=RKAN_DEGREE)

    def forward(self, x):
        return self.head(self.backbone(x))

    def get_features(self, x):
        return _vit_patch_spatial(self.backbone, x, self.GRID_SIZE)

    def get_cam_target_layer(self): return self.backbone.blocks[-1]
    def get_attention_blocks(self): return self.backbone.blocks

    def get_kan_feature_importance(self, x, class_idx=None):
        cls = self.backbone(x).detach().requires_grad_(True)
        out = self.head(cls)
        target = out[:, class_idx].sum() if class_idx is not None else out.abs().sum()
        return torch.autograd.grad(target, cls)[0].abs()

    def is_kan_model(self): return True
    def is_vit_model(self):  return True


# ── Model registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    # ResNet-50 variants
    "resnet_baseline":       ResNet50Baseline,
    "resnet_vanilla_kan":    ResNet50VanillaKAN,
    "resnet_rational_kan":   ResNet50RationalKAN,
    "resnet_group_kan":      ResNet50GroupKAN,
    "resnet_conv_kan":       ResNet50ConvKAN,
    # ViT-Small/16 variants
    "vit_baseline":          ViTSmallBaseline,
    "vit_vanilla_kan":       ViTSmallVanillaKAN,
    "vit_group_kan":         ViTSmallGroupKAN,
    "vit_rational_kan":      ViTSmallRationalKAN,
}



def build_model(model_name: str, num_classes: int = 14, freeze_backbone: bool = True):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    model = MODEL_REGISTRY[model_name](num_classes=num_classes, freeze_backbone=freeze_backbone)
    print(f"[Model] Built {model.model_name}")
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"        Total params:     {total:,}")
    print(f"        Trainable params: {trainable:,}")
    return model