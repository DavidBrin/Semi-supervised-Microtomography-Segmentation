import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image
from torchvision import transforms
from skimage.filters import threshold_otsu
import timm
import logging

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Configuration ---
class Config:
    """Configuration class for ViT training."""
   
    image_height = 224  # ViT works best with 224x224
    image_width = 224
    num_classes = 2      # 0: Background, 1: Pore
    batch_size = 8
    epochs = 30
    validation_split = 0.2
    test_split = 0.1
    learning_rate = 1e-4
    freeze_backbone = True  # Freeze ViT backbone for transfer learning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()
CHECKPOINTS_DIR = Path("../checkpoints")
MODEL_SAVE_PATH = CHECKPOINTS_DIR / "vit_porosity_model_pytorch.pth"

print(f"Using device: {config.device}")

# --- 2. ViT Segmentation Head ---
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
        x = x[:, 1:, :]  # drop CLS token
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        x = self.decoder(x)
        return x


# --- 3. ViT Model with Transfer Learning ---
class ViTSegmentation(nn.Module):
    """Vision Transformer for Segmentation with timm pretrained backbone"""
    def __init__(
        self,
        num_classes: int = 2,
        img_size: int = 224,
        freeze_backbone: bool = True,
        use_pretrained: bool = True,
    ):
        super().__init__()

        # Load pretrained ViT backbone from timm
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=use_pretrained,
            num_classes=0,      # remove classifier head
            img_size=img_size
        )

        # Freeze backbone for transfer learning
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("ViT backbone frozen for transfer learning")

        # Segmentation head (decoder) - this will be trained
        self.seg_head = ViTSegmentationHead(
            embed_dim=self.backbone.embed_dim,
            num_classes=num_classes,
            img_size=img_size,
            patch_size=self.backbone.patch_embed.patch_size[0]
        )

        logger.info(f"ViT model initialized with pretrained weights: {use_pretrained}")

    def forward(self, x):
        feats = self.backbone.forward_features(x)   # [B, num_patches+1, C]
        seg_map = self.seg_head(feats)
        return seg_map


# --- 4. Dataset Class  ---
class XrayPorosityDataset(Dataset):
    """
    Custom dataset for X-ray tomography porosity segmentation
    Adapted from Unet_TransferLearn.py
    Uses Otsu's automatic thresholding for better pore segmentation
    """
    
    def __init__(self, image_paths, mask_paths, transform=None, target_size=(224, 224), 
                 augment=False, use_otsu=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_size = target_size
        self.augment = augment
        self.use_otsu = use_otsu
        
        # Preprocessing transforms for ViT (needs 3 channels)
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size, transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        
        # Mask preprocessing
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size, transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        
        # Augmentation pipeline
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90, 
                                        interpolation=transforms.InterpolationMode.BILINEAR),
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load as numpy arrays
        image = np.array(Image.open(image_path)).astype(np.float32)
        mask = np.array(Image.open(mask_path)).astype(np.float32)
        
        # Normalize image to [0, 1]
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
        
        # Convert grayscale to 3 channels for ViT
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        
        # Convert mask to binary using Otsu thresholding
        if self.use_otsu:
            threshold = threshold_otsu(mask)
            mask_binary = (mask > threshold).astype(np.float32)
        else:
            mask_binary = (mask > np.median(mask)).astype(np.float32)
        
        # Apply preprocessing transforms
        image = self.base_transform(image)
        mask_binary = self.mask_transform(mask_binary)
        
        # Apply augmentations if enabled
        if self.augment:
            seed = torch.randint(0, 2**32, (1,)).item()
            
            torch.manual_seed(seed)
            image = self.augment_transform(image)
            
            torch.manual_seed(seed)
            mask_binary = self.augment_transform(mask_binary)
        
        # Convert mask to long tensor for CrossEntropyLoss
        mask_binary = (mask_binary.squeeze(0) > 0.5).long()
        
        return image, mask_binary


# --- 5. Data Loading Functions ---
def load_dataset_info(IMAGE_DIR, MASK_DIR):
    """Load and analyze the dataset structure"""
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.tif')])
    mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('.tif')])
    
    print(f"Dataset Overview:")
    print(f"- Total images: {len(image_files)}")
    print(f"- Total masks: {len(mask_files)}")
    print(f"- Resolution: 22 nm in all directions")
    
    return image_files, mask_files


def create_datasets(image_files, mask_files, IMAGE_DIR, MASK_DIR, 
                   test_size=0.1, val_size=0.2, target_size=(224, 224), 
                   batch_size=8, use_otsu=True):
    """Create train, validation, and test datasets with proper splits"""
    
    # Get all file paths
    image_paths = [str(IMAGE_DIR / f) for f in image_files]
    mask_paths = [str(MASK_DIR / f) for f in mask_files]
    
    # Split into train+val and test
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_paths, mask_paths, test_size=test_size, random_state=42, shuffle=True
    )
    
    # Split train into train and validation
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        train_imgs, train_masks, test_size=val_size/(1-test_size), 
        random_state=42, shuffle=True
    )
    
    threshold_method = "Otsu (improved)" if use_otsu else "Median (baseline)"
    print(f"\nDataset splits:")
    print(f"- Training: {len(train_imgs)} samples")
    print(f"- Validation: {len(val_imgs)} samples")
    print(f"- Testing: {len(test_imgs)} samples")
    print(f"- Threshold method: {threshold_method}")
    
    # Create datasets
    train_dataset = XrayPorosityDataset(
        train_imgs, train_masks, target_size=target_size, 
        augment=True, use_otsu=use_otsu
    )
    val_dataset = XrayPorosityDataset(
        val_imgs, val_masks, target_size=target_size, 
        augment=False, use_otsu=use_otsu
    )
    test_dataset = XrayPorosityDataset(
        test_imgs, test_masks, target_size=target_size, 
        augment=False, use_otsu=use_otsu
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )
    
    return train_loader, val_loader, test_loader


# --- 6. Metrics and Loss Functions ---
def dice_loss(y_pred, y_true, smooth=1e-6):
    """Dice loss for segmentation"""
    # Convert y_true to one-hot encoding
    y_true_one_hot = F.one_hot(y_true, num_classes=config.num_classes)
    y_true_one_hot = y_true_one_hot.permute(0, 3, 1, 2).float()
    
    # Apply softmax to predictions
    y_pred_soft = F.softmax(y_pred, dim=1)
    
    # Focus on the Pore class (index 1)
    y_true_f = y_true_one_hot[:, 1, ...].contiguous().view(-1)
    y_pred_f = y_pred_soft[:, 1, ...].contiguous().view(-1)

    intersection = (y_pred_f * y_true_f).sum()
    total = y_pred_f.sum() + y_true_f.sum()
    dice = (2. * intersection + smooth) / (total + smooth)

    return 1. - dice


def dice_coeff(y_pred, y_true, smooth=1e-6):
    """Dice coefficient (F1-score) metric"""
    # Convert y_true to one-hot encoding
    y_true_one_hot = F.one_hot(y_true, num_classes=config.num_classes)
    y_true_one_hot = y_true_one_hot.permute(0, 3, 1, 2).float()
    
    # Apply softmax and threshold
    y_pred_soft = F.softmax(y_pred, dim=1)
    
    y_true_f = y_true_one_hot[:, 1, ...].contiguous().view(-1)
    y_pred_f = (y_pred_soft[:, 1, ...] > 0.5).float().contiguous().view(-1)

    intersection = (y_pred_f * y_true_f).sum()
    total = y_pred_f.sum() + y_true_f.sum()
    
    return (2. * intersection + smooth) / (total + smooth)


# --- 7. Training and Evaluation ---
def train_epoch(model, dataloader, criterion, optimizer):
    """Performs a single training epoch."""
    model.train()
    total_loss = 0
    total_dice = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(config.device), targets.to(config.device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_dice += dice_coeff(outputs, targets).item()

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    return avg_loss, avg_dice


def evaluate_model(model, dataloader, criterion):
    """Evaluates the model on validation or test set."""
    model.eval()
    total_loss = 0
    total_dice = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_dice += dice_coeff(outputs, targets).item()

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    return avg_loss, avg_dice


# --- 8. Main Execution ---
def main():
    """Loads data, trains the ViT model with transfer learning, and saves the checkpoint."""
    print("--- PyTorch ViT Porosity Segmentation Training Script (Transfer Learning) ---")

    # Dataset paths
    DATA_DIR = Path(".")
    IMAGE_DIR = DATA_DIR / "Original Images"
    MASK_DIR = DATA_DIR / "Original Masks"
    
    # Load dataset
    print("\nLoading dataset...")
    image_files, mask_files = load_dataset_info(IMAGE_DIR, MASK_DIR)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_datasets(
        image_files, mask_files, IMAGE_DIR, MASK_DIR,
        target_size=(config.image_height, config.image_width),
        batch_size=config.batch_size,
        use_otsu=True
    )
    
    # Create model with transfer learning
    print("\nCreating ViT model with transfer learning...")
    model = ViTSegmentation(
        num_classes=config.num_classes,
        img_size=config.image_height,
        freeze_backbone=config.freeze_backbone,
        use_pretrained=True
    )
    model.to(config.device)
    
    # Define Loss, Optimizer
    criterion = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Print trainable parameters (should mainly be the segmentation head)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in model.parameters())
    print(f"\n Model Parameters: {num_total_params:,}")
    print(f"Trainable Parameters (Segmentation Head only): {num_trainable_params:,} "
          f"({(num_trainable_params/num_total_params)*100:.2f}%)")

    # Training Loop
    print(f"\nStarting training for {config.epochs} epochs...")
    best_val_dice = 0.0
    
    for epoch in range(config.epochs):
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_dice = evaluate_model(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        
        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            print(f"  New best validation Dice: {best_val_dice:.4f}")
              
    print(" Training complete.")

    # Evaluation
    print("\n Evaluating model on Test Set...")
    test_loss, test_dice = evaluate_model(model, test_loader, criterion)

    print(f"\n--- Test Results ---")
    print(f"  - Test Loss (Dice Loss): {test_loss:.4f}")
    print(f"  - Test Dice Coeff (F1-score): {test_dice:.4f}")

    # Model Saving
    print("\n Saving model checkpoint...")
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    
    # Save model state dict
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_dice': test_dice,
        'test_loss': test_loss,
        'config': {
            'num_classes': config.num_classes,
            'img_size': config.image_height,
            'freeze_backbone': config.freeze_backbone,
        }
    }, MODEL_SAVE_PATH)
    
    print(f" ViT Model successfully saved to: {MODEL_SAVE_PATH}")
    print("\n--- Script Finished ---")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
