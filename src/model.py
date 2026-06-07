import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class ChannelAttention(nn.Module):
    """
    Channel attention module used inside CBAM.
    It helps the model focus on important feature channels.
    """

    def __init__(self, in_planes, ratio=16):
        super().__init__()

        hidden_planes = max(in_planes // ratio, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_planes, in_planes, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))

        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    Spatial attention module used inside CBAM.
    It helps the model focus on important spatial regions.
    """

    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]

        x_cat = torch.cat([avg_out, max_out], dim=1)

        return self.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Used before the classification branch so the classifier can focus on
    more meaningful encoder features.
    """

    def __init__(self, in_planes):
        super().__init__()

        self.channel_attention = ChannelAttention(in_planes)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)

        return x


class MTL_EfficientUNetPlusPlus(nn.Module):
    """
    Multi-task model for breast tumor segmentation and classification.

    Outputs:
        seg_mask: segmentation logits, shape [B, 1, H, W]
        class_score: classification logits, shape [B, 1]

    Segmentation:
        U-Net++ with EfficientNet-B0 encoder.

    Classification:
        Uses the deepest encoder feature map with CBAM attention.
    """

    def __init__(
        self,
        encoder_name="tu-tf_efficientnet_b0",
        encoder_weights="imagenet",
        in_channels=1,
        num_classes=1,
    ):
        super().__init__()

        self.smp_base = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )

        bottleneck_channels = self.smp_base.encoder.out_channels[-1]

        self.classification_branch = nn.Sequential(
            CBAM(in_planes=bottleneck_channels),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        features = self.smp_base.encoder(x)

        decoder_output = self.smp_base.decoder(features)
        seg_mask = self.smp_base.segmentation_head(decoder_output)

        class_score = self.classification_branch(features[-1])

        return seg_mask, class_score


class Run85UNetPlusPlus(nn.Module):
    """
    Notebook Run 8.5 compatible architecture.

    - Unet++ with EfficientNet-B0 encoder
    - 3-channel fused input
    - segmentation head (1 channel) + 2-class auxiliary classifier

    Kept as an additional model option so current deployed GUI weights
    (`models/best_model.pth`) remain fully compatible with
    `MTL_EfficientUNetPlusPlus`.
    """

    def __init__(self, encoder_weights=None):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b0",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            aux_params=dict(classes=2, dropout=0.5, pooling="avg"),
        )

    def forward(self, x):
        return self.model(x)


class BinaryFocalLoss(nn.Module):
    """
    Binary focal loss for benign/malignant classification.

    Useful when classes are imbalanced.
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()


def calc_dice(pred_logits, target, threshold=0.5, smooth=1e-5):
    """
    Calculates Dice score for segmentation.

    Args:
        pred_logits: raw model segmentation output
        target: ground truth binary mask
        threshold: sigmoid threshold
        smooth: small value to avoid division by zero
    """

    pred = (torch.sigmoid(pred_logits) > threshold).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice.mean().item()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MTL_EfficientUNetPlusPlus(
        encoder_weights=None  # test için internet/pretrained indirme gerekmesin
    ).to(device)

    dummy_input = torch.randn(2, 1, 512, 512).to(device)

    seg_mask, class_score = model(dummy_input)

    print("Model test successful.")
    print("Input shape:", dummy_input.shape)
    print("Segmentation output shape:", seg_mask.shape)
    print("Classification output shape:", class_score.shape)