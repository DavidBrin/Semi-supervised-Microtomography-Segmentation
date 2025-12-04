import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tifffile import imread
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
#  ViT segmentation head
# ---------------------------------------------------------------------
class ViTSegmentationHead(nn.Module):
    """Segmentation head for Vision Transformer"""
    def __init__(self, embed_dim=768, num_classes=2, img_size=224, patch_size=16):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.img_size = img_size

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # x: [B, num_patches + 1, embed_dim]
        x = x[:, 1:, :]  # drop CLS
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        x = self.decoder(x)
        return x


# ---------------------------------------------------------------------
#  ViT backbone + head
# ---------------------------------------------------------------------
class ViTSegmentation(nn.Module):
    """Vision Transformer for Segmentation with timm pretrained backbone"""
    def __init__(
        self,
        num_classes: int = 2,
        img_size: int = 224,
        freeze_backbone: bool = False,
        use_pretrained: bool = True,
        vit_npz_path: Optional[str] = None,  # optional, not needed in practice
    ):
        super().__init__()

        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=use_pretrained,
            num_classes=0,      # remove classifier head
            img_size=img_size
        )

        # Optional: override with .npz weights (not really needed)
        if vit_npz_path:
            self.load_vit_weights(vit_npz_path)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.seg_head = ViTSegmentationHead(
            embed_dim=self.backbone.embed_dim,
            num_classes=num_classes,
            img_size=img_size,
            patch_size=self.backbone.patch_embed.patch_size[0]
        )

    def load_vit_weights(self, npz_path: str):
        logger.info(f"Loading ViT weights from {npz_path}")
        try:
            weights = np.load(npz_path)
            state_dict = {}
            for key in weights.files:
                if 'Transformer/encoderblock' in key:
                    new_key = key.replace('Transformer/encoderblock_', 'blocks.')
                    new_key = new_key.replace('/LayerNorm_0/', '.norm1.')
                    new_key = new_key.replace('/LayerNorm_2/', '.norm2.')
                    new_key = new_key.replace('/MlpBlock_3/Dense_0/', '.mlp.fc1.')
                    new_key = new_key.replace('/MlpBlock_3/Dense_1/', '.mlp.fc2.')
                    state_dict[new_key] = torch.from_numpy(weights[key])
                elif 'embedding' in key:
                    new_key = key.replace('embedding/', 'patch_embed.')
                    state_dict[new_key] = torch.from_numpy(weights[key])
            self.backbone.load_state_dict(state_dict, strict=False)
            logger.info("ViT .npz weights loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load .npz weights: {e}. Using timm weights instead.")

    def forward(self, x):
        feats = self.backbone.forward_features(x)   # [B, num_patches+1, C]
        seg_map = self.seg_head(feats)
        return seg_map


# ---------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------
class SegmentationDataset(Dataset):
    """Dataset for segmentation with labeled and unlabeled data"""
    def __init__(
        self,
        image_paths,
        image_dir,
        mask_paths=None,
        mask_dir=None,
        transform=None,
        img_size=512,
    ):
        self.image_paths = image_paths
        self.image_dir = Path(image_dir)
        self.mask_paths = mask_paths
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform
        self.img_size = img_size
        self.is_labeled = mask_paths is not None

        self.basic_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # => [0,1] float32
        ])

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path: Path):
        img = imread(path).astype(np.float32)

        # Normalize safely to [0,1] without casting to uint8
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # to H,W,C (3 channels)
        if img.ndim == 2:  # grayscale
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]

        return img

    def _load_mask(self, path: Path):
        mask = imread(path)

        # Binarize: pores=1, background=0 (matching ground truth)
        mask = (mask > 0).astype(np.uint8)

        mask_pil = transforms.ToPILImage()(mask)
        mask_resized = transforms.Resize(
            (self.img_size, self.img_size),
            interpolation=transforms.InterpolationMode.NEAREST
        )(mask_pil)

        mask_resized = np.array(mask_resized).astype(np.int64)
        return mask_resized

    def __getitem__(self, idx):
        img_path = self.image_dir / self.image_paths[idx]
        image = self._load_image(img_path)
        image = self.basic_transform(image)  # [3, H, W] float32

        if self.transform:
            image = self.transform(image)

        if self.is_labeled:
            mask_path = self.mask_dir / self.mask_paths[idx]
            mask = self._load_mask(mask_path)
            return image, torch.from_numpy(mask)
        else:
            return image


# ---------------------------------------------------------------------
#  Cross-teaching trainer
# ---------------------------------------------------------------------
class CrossTeachingTrainer:
    """Cross-teaching trainer for semi-supervised segmentation"""
    def __init__(
        self,
        unet_model: nn.Module,
        vit_model: nn.Module,
        device: str = "cuda",
        lr: float = 1e-4,
        consistency_weight: float = 0.5,
        confidence_threshold: float = 0.9,
        unet_size: int = 512,
        vit_size: int = 224,
    ):
        self.unet = unet_model.to(device)
        self.vit = vit_model.to(device)
        self.device = device

        self.unet_size = unet_size
        self.vit_size = vit_size

        self.unet_optimizer = torch.optim.Adam(self.unet.parameters(), lr=lr)
        self.vit_optimizer = torch.optim.Adam(self.vit.parameters(), lr=lr)

        self.consistency_weight = consistency_weight
        self.confidence_threshold = confidence_threshold

        self.supervised_loss = nn.CrossEntropyLoss()

    # -------- utilities --------
    def get_confidence_mask(self, pred, threshold):
        probs = F.softmax(pred, dim=1)
        max_probs, _ = probs.max(dim=1)
        return (max_probs > threshold).float()

    # -------- labeled step --------
    def train_step_labeled(self, images, masks):
        images = images.to(self.device)          # [B,3,H,W] from dataset
        masks = masks.to(self.device)           # [B,H,W]

        masks = (masks > 0).long()

        # U-Net expects 1 channel, 512x512
        unet_images = images.mean(dim=1, keepdim=True)  # [B,1,H,W]
        if unet_images.shape[-1] != self.unet_size:
            unet_images = F.interpolate(
                unet_images, size=(self.unet_size, self.unet_size),
                mode="bilinear", align_corners=False
            )
            masks_unet = F.interpolate(
                masks.float().unsqueeze(1),
                size=(self.unet_size, self.unet_size),
                mode="nearest"
            ).squeeze(1).long()
        else:
            masks_unet = masks

        # ViT expects 3 channels, 224x224
        vit_images = images
        if vit_images.shape[-1] != self.vit_size:
            vit_images = F.interpolate(
                vit_images, size=(self.vit_size, self.vit_size),
                mode="bilinear", align_corners=False
            )
            masks_vit = F.interpolate(
                masks.unsqueeze(1).float(),
                size=(self.vit_size, self.vit_size),
                mode="nearest"
            ).squeeze(1).round().long()
        else:
            masks_vit = masks

        # forward
        unet_pred = self.unet(unet_images)   # [B,C,512,512]
        vit_pred = self.vit(vit_images)      # [B,C,224,224]

        unet_loss = self.supervised_loss(unet_pred, masks_unet)
        vit_loss = self.supervised_loss(vit_pred, masks_vit)

        # backward
        self.unet_optimizer.zero_grad()
        unet_loss.backward()
        self.unet_optimizer.step()

        self.vit_optimizer.zero_grad()
        vit_loss.backward()
        self.vit_optimizer.step()

        return {"unet_loss": unet_loss.item(), "vit_loss": vit_loss.item()}

    # -------- unlabeled step (cross-teaching) --------
    def train_step_unlabeled(self, images):
        images = images.to(self.device)  # [B,3,H,W]

        # Prepare U-Net input
        unet_images = images.mean(dim=1, keepdim=True)
        if unet_images.shape[-1] != self.unet_size:
            unet_images = F.interpolate(
                unet_images, size=(self.unet_size, self.unet_size),
                mode="bilinear", align_corners=False
            )

        # Prepare ViT input
        vit_images = images
        if vit_images.shape[-1] != self.vit_size:
            vit_images = F.interpolate(
                vit_images, size=(self.vit_size, self.vit_size),
                mode="bilinear", align_corners=False
            )

        with torch.no_grad():
            # teacher predictions
            unet_teacher = self.unet(unet_images)        # [B,C,512,512]
            vit_teacher = self.vit(vit_images)          # [B,C,224,224]

            # confidence (for logging)
            unet_conf = self.get_confidence_mask(unet_teacher, self.confidence_threshold)
            vit_conf = self.get_confidence_mask(vit_teacher, self.confidence_threshold)

            # create pseudo-labels in opposite model's resolution
            # ViT -> U-Net (upsample to 512)
            vit_teacher_512 = F.interpolate(
                vit_teacher, size=(self.unet_size, self.unet_size),
                mode="bilinear", align_corners=False
            )
            vit_pseudo_labels_512 = vit_teacher_512.argmax(dim=1)   # [B,512,512]

            # U-Net -> ViT (downsample to 224)
            unet_teacher_224 = F.interpolate(
                unet_teacher, size=(self.vit_size, self.vit_size),
                mode="bilinear", align_corners=False
            )
            unet_pseudo_labels_224 = unet_teacher_224.argmax(dim=1) # [B,224,224]

        # ------- Train U-Net with ViT pseudo-labels -------
        unet_pred = self.unet(unet_images)  # [B,C,512,512]
        unet_consistency = self.supervised_loss(unet_pred, vit_pseudo_labels_512)

        self.unet_optimizer.zero_grad()
        (self.consistency_weight * unet_consistency).backward()
        self.unet_optimizer.step()

        # ------- Train ViT with U-Net pseudo-labels -------
        vit_pred = self.vit(vit_images)     # [B,C,224,224]
        vit_consistency = self.supervised_loss(vit_pred, unet_pseudo_labels_224)

        self.vit_optimizer.zero_grad()
        (self.consistency_weight * vit_consistency).backward()
        self.vit_optimizer.step()

        return {
            "unet_consistency_loss": unet_consistency.item(),
            "vit_consistency_loss": vit_consistency.item(),
            "unet_confidence": unet_conf.mean().item(),
            "vit_confidence": vit_conf.mean().item(),
        }

    # -------- epoch loop --------
    def train_epoch(self, labeled_loader, unlabeled_loader, epoch: int):
        self.unet.train()
        self.vit.train()

        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        num_batches = max(len(labeled_loader), len(unlabeled_loader))

        metrics = {
            "unet_loss": 0.0,
            "vit_loss": 0.0,
            "unet_consistency": 0.0,
            "vit_consistency": 0.0,
        }

        for batch_idx in range(num_batches):
            # labeled step
            try:
                images_l, masks_l = next(labeled_iter)
                labeled_metrics = self.train_step_labeled(images_l, masks_l)
                metrics["unet_loss"] += labeled_metrics["unet_loss"]
                metrics["vit_loss"] += labeled_metrics["vit_loss"]
            except StopIteration:
                labeled_iter = iter(labeled_loader)

            # unlabeled step
            try:
                images_u = next(unlabeled_iter)
                if isinstance(images_u, (list, tuple)):
                    images_u = images_u[0]
                unlabeled_metrics = self.train_step_unlabeled(images_u)
                metrics["unet_consistency"] += unlabeled_metrics["unet_consistency_loss"]
                metrics["vit_consistency"] += unlabeled_metrics["vit_consistency_loss"]
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}")

        for k in metrics:
            metrics[k] /= num_batches

        logger.info(f"Epoch {epoch} - Metrics: {metrics}")
        return metrics

    # -------- ensemble prediction (for evaluation / visualization) --------
    def ensemble_predict(self, images):
        self.unet.eval()
        self.vit.eval()
        with torch.no_grad():
            images = images.to(self.device)  # [B,3,H,W]

            # U-Net input
            unet_images = images.mean(dim=1, keepdim=True)
            unet_images = F.interpolate(
                unet_images, size=(self.unet_size, self.unet_size),
                mode="bilinear", align_corners=False
            )
            unet_logits = self.unet(unet_images)  # [B,C,512,512]
            unet_probs = F.softmax(unet_logits, dim=1)

            # ViT input
            vit_images = images
            vit_images = F.interpolate(
                vit_images, size=(self.vit_size, self.vit_size),
                mode="bilinear", align_corners=False
            )
            vit_logits = self.vit(vit_images)      # [B,C,224,224]
            vit_logits_512 = F.interpolate(
                vit_logits, size=(self.unet_size, self.unet_size),
                mode="bilinear", align_corners=False
            )
            vit_probs = F.softmax(vit_logits_512, dim=1)

            ensemble_probs = (unet_probs + vit_probs) / 2
            return ensemble_probs  # [B,C,512,512]


# ---------------------------------------------------------------------
#  Dataset helpers
# ---------------------------------------------------------------------
def load_dataset_info(IMAGE_DIR, MASK_DIR):
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".tif")])
    mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith(".tif")])

    print("Dataset Overview:")
    print(f"- Total images: {len(image_files)}")
    print(f"- Total masks: {len(mask_files)}")
    print(f"- Dataset size: ~3.6 GB")
    print(f"- Resolution: 22 nm in all directions\n")

    print("File naming pattern:")
    print(f"- Images: {image_files[:3]}...")
    print(f"- Masks:  {mask_files[:3]}...\n")

    return image_files, mask_files


def load_unlabeled_data(UIMAGE_DIR):
    image_files = []
    for subdir in os.listdir(UIMAGE_DIR):
        subdir_path = os.path.join(UIMAGE_DIR, subdir)
        if os.path.isdir(subdir_path):
            subdir_files = [f for f in os.listdir(subdir_path) if f.endswith(".tif")]
            image_files.extend(subdir_files)
        else:
            # Files directly in UIMAGE_DIR
            if subdir.endswith(".tif"):
                image_files.append(subdir)
    
    image_files = sorted(image_files)

    print("Unlabeled dataset Overview:")
    print(f"- Total images: {len(image_files)}\n")
    print("File naming pattern:")
    print(f"- Images: {image_files[:3]}...\n")

    return image_files


# ---------------------------------------------------------------------
#  U-Net loader
# ---------------------------------------------------------------------
def load_unet_model(unet_path: str, device: str = "cuda"):
    logger.info(f"Loading U-Net model from {unet_path}")

    if not Path(unet_path).exists():
        raise FileNotFoundError(f"U-Net model file not found at: {unet_path}")

    checkpoint = torch.load(unet_path, map_location=device)

    # Try loading with segmentation_models_pytorch first (if that's what was used for training)
    try:
        import segmentation_models_pytorch as smp
        unet_model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=2
        )
        logger.info("Using segmentation_models_pytorch U-Net")
    except ImportError:
        # Fallback to custom U-Net
        import sys
        sys.path.append("..")
        from unet_pytorch import create_unet_for_porosity
        unet_model = create_unet_for_porosity(input_size=(512, 512), device=device)
        logger.info("Using custom U-Net from unet_pytorch")

    # Extract and load state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Unexpected checkpoint format")

    unet_model.load_state_dict(state_dict, strict=False)
    unet_model.to(device)
    unet_model.eval()
    logger.info("U-Net model loaded successfully")
    return unet_model

# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------
def main():
    # ===== User config =====
    unet_path = "../checkpoints/unet_porosity_model_pytorch.pth"
    vit_npz_path = "../checkpoints/vit_porosity_model_pytorch.pth"  # not needed; using timm weights

    DATA_DIR = Path("../Data")
    IMAGE_DIR = DATA_DIR / "Original Images"
    MASK_DIR = DATA_DIR / "Original Masks"
    UIMAGE_DIR = DATA_DIR / "Original UImages"

    num_classes = 2
    unet_img_size = 512
    vit_img_size = 224

    batch_size = 8
    num_epochs = 30
    learning_rate = 1e-4
    consistency_weight = 0.5
    confidence_threshold = 0.9

    # ===== Device =====
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")

    # ===== Data =====
    logger.info("Loading dataset information...")
    image_files, mask_files = load_dataset_info(IMAGE_DIR, MASK_DIR)
    unlabeled_image_files = load_unlabeled_data(UIMAGE_DIR)

    logger.info("Creating datasets...")
    labeled_dataset = SegmentationDataset(
        image_paths=image_files,
        image_dir=IMAGE_DIR,
        mask_paths=mask_files,
        mask_dir=MASK_DIR,
        img_size=unet_img_size,
    )

    unlabeled_dataset = SegmentationDataset(
        image_paths=unlabeled_image_files,
        image_dir=UIMAGE_DIR,
        img_size=unet_img_size,
    )

    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    logger.info(f"Labeled batches: {len(labeled_loader)}, Unlabeled batches: {len(unlabeled_loader)}")

    # ===== Models =====
    unet_model = load_unet_model(unet_path, device)

    vit_model = ViTSegmentation(
        num_classes=num_classes,
        img_size=vit_img_size,
        freeze_backbone=False,
        use_pretrained=True,
        vit_npz_path=vit_npz_path if vit_npz_path and Path(vit_npz_path).exists() else None,
    ).to(device)

    # ===== Trainer =====
    trainer = CrossTeachingTrainer(
        unet_model=unet_model,
        vit_model=vit_model,
        device=device,
        lr=learning_rate,
        consistency_weight=consistency_weight,
        confidence_threshold=confidence_threshold,
        unet_size=unet_img_size,
        vit_size=vit_img_size,
    )

    # ===== Training loop =====
    for epoch in range(num_epochs):
        metrics = trainer.train_epoch(labeled_loader, unlabeled_loader, epoch)

        if (epoch + 1) % 10 == 0:
            torch.save(trainer.unet.state_dict(), f"unet_epoch_{epoch+1}.pth")
            torch.save(trainer.vit.state_dict(), f"vit_epoch_{epoch+1}.pth")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
