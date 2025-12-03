import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# External library for segmentation models (install with: pip install segmentation-models-pytorch)
try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("Warning: 'segmentation_models_pytorch' not found. Please install it with 'pip install segmentation-models-pytorch'.")
    exit()

# --- 1. Configuration ---
class Config:
    """Configuration class for U-Net training."""
   
    image_height = 512
    image_width = 512
    num_classes = 2      # 0: Background, 1: Pore
    batch_size = 4
    epochs = 5
    validation_split = 0.2
    test_split = 0.1
    learning_rate = 1e-4
    encoder_name = "resnet34"  # A robust, pre-trained backbone
    weights = "imagenet"       # Use pre-trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()
CHECKPOINTS_DIR = os.path.join("..", "checkpoints")
MODEL_SAVE_PATH = os.path.join(CHECKPOINTS_DIR, "unet_porosity_model_pytorch.pth")

print(f"Using device: {config.device}")

# --- 2. Custom Metrics and Loss Functions (PyTorch) ---

def dice_loss(y_pred, y_true, smooth=1e-6):
    """Dice loss based on Jaccard (IoU) coefficient."""
    # y_pred and y_true are (N, C, H, W). We focus on the Pore class (index 1).
    y_true_f = y_true[:, 1, ...].contiguous().view(-1)
    y_pred_f = y_pred[:, 1, ...].contiguous().view(-1)

    intersection = (y_pred_f * y_true_f).sum()
    total = y_pred_f.sum() + y_true_f.sum()
    dice = (2. * intersection + smooth) / (total + smooth)

    return 1. - dice

def dice_coeff(y_pred, y_true, smooth=1e-6):
    """Dice coefficient (F1-score) metric."""
    # Same logic as loss, but returns the coefficient.
    y_true_f = y_true[:, 1, ...].contiguous().view(-1)
    y_pred_f = (y_pred[:, 1, ...] > 0.5).float().contiguous().view(-1) # Apply threshold for metric

    intersection = (y_pred_f * y_true_f).sum()
    total = y_pred_f.sum() + y_true_f.sum()
    
    return (2. * intersection + smooth) / (total + smooth)

# --- 3. U-Net Model Definition with Transfer Learning ---

def create_unet_tl():
    """
    Constructs the U-Net architecture using a pre-trained encoder and freezes
    its weights for transfer learning on the segmentation head (decoder).
    [Image of U-Net architecture]
    """
    # Initialize U-Net with a pre-trained ResNet encoder
    model = smp.Unet(
        encoder_name=config.encoder_name,
        encoder_weights=config.weights,  # Use ImageNet pre-trained weights
        in_channels=1,                   # Grayscale input (1 channel)
        classes=config.num_classes,      # 2 classes (Background, Pore)
        activation='softmax'             # Use softmax for multi-class/one-hot output
    )

    print(f"Frozen encoder: {config.encoder_name} weights loaded from {config.weights}.")
    
    # FREEZE ENCODER WEIGHTS (Transfer Learning)
    # This prevents the pre-trained weights from changing during training,
    # focusing learning on the decoder/segmentation head.
    for param in model.encoder.parameters():
        param.requires_grad = False

    return model

# --- 4. Mock Data Loading and Preprocessing ---

def load_and_prepare_data():
    """
    Creates !mock! data and converts it to PyTorch tensors in NCHW format.
    """
    print("🔬 Creating mock dataset (100 samples)...")
    N = 100 # Number of samples
    H, W = config.image_height, config.image_width
    C_in = 1 # Grayscale input

    # Mock Input Images: Random grayscale images (NHWC format initially)
    images_nhwc = np.random.rand(N, H, W, C_in).astype(np.float32)

    # Mock Labels: Random binary masks (N H W format)
    masks_nhw = np.random.randint(0, config.num_classes, size=(N, H, W)).astype(np.uint8)

    # Convert to PyTorch Tensor format (NCHW for images, NHW for labels/masks)
    # PyTorch NCHW: (N, C, H, W)
    images_nchw = np.transpose(images_nhwc, (0, 3, 1, 2))

    # Convert masks to one-hot encoding (N, H, W) -> (N, H, W, C) -> (N, C, H, W)
    # Note: We use one-hot for Dice Loss calculation, but often raw long tensor is used for CrossEntropyLoss.
    # Sticking to one-hot (N, C, H, W) for consistency with the original TensorFlow approach.
    masks_one_hot = np.moveaxis(tf.keras.utils.to_categorical(masks_nhw, num_classes=config.num_classes), -1, 1).astype(np.float32)

    X = torch.from_numpy(images_nchw)
    y = torch.from_numpy(masks_one_hot)

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.test_split, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=config.validation_split / (1 - config.test_split), random_state=42
    )

    print(f"Dataset Size: {N}")
    print(f"Train samples: {len(X_train)} (Shape: {X_train.shape})")
    print(f"Validation samples: {len(X_val)} (Shape: {X_val.shape})")
    print(f"Test samples: {len(X_test)} (Shape: {X_test.shape})")

    # Create PyTorch DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    return train_loader, val_loader, test_loader

# Dataset paths
DATA_DIR = Path("../Data")
IMAGE_DIR = DATA_DIR / "Original Images"
MASK_DIR = DATA_DIR / "Original Masks"

def load_dataset_info():
    """Load and analyze the dataset structure"""
    
    # Get all image and mask files
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.tif')])
    mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('.tif')])
    
    print(f"Dataset Overview:")
    print(f"- Total images: {len(image_files)}")
    print(f"- Total masks: {len(mask_files)}")
    print(f"- Dataset size: ~3.6 GB")
    print(f"- Resolution: 22 nm in all directions")
    
    # Analyze file naming convention for temporal information
    print(f"\nFile naming pattern:")
    print(f"- Images: {image_files[:3]}...")
    print(f"- Masks: {mask_files[:3]}...")
    
    return image_files, mask_files

def analyze_sample_image(image_path, mask_path):
    """Analyze a single image-mask pair"""
    
    # Load image and mask
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    
    image_array = np.array(image)
    mask_array = np.array(mask)
    
    print(f"\nImage Analysis:")
    print(f"- Image shape: {image_array.shape}")
    print(f"- Image dtype: {image_array.dtype}")
    print(f"- Image range: [{image_array.min():.6f}, {image_array.max():.6f}]")
    print(f"- Image mean: {image_array.mean():.6f}")
    print(f"- Image std: {image_array.std():.6f}")
    
    print(f"\nMask Analysis:")
    print(f"- Mask shape: {mask_array.shape}")
    print(f"- Mask dtype: {mask_array.dtype}")
    print(f"- Unique values: {len(np.unique(mask_array))} distinct values")
    print(f"- Mask range: [{mask_array.min():.6f}, {mask_array.max():.6f}]")
    
    # Calculate porosity statistics
    if len(np.unique(mask_array)) == 2:  # Binary mask
        pore_pixels = (mask_array > mask_array.min()).sum()
        total_pixels = mask_array.size
        porosity_ratio = pore_pixels / total_pixels
        print(f"- Porosity ratio: {porosity_ratio:.3f} ({porosity_ratio*100:.1f}%)")
    else:
        # Non-binary mask - threshold at median
        threshold = np.median(mask_array)
        pore_pixels = (mask_array > threshold).sum()
        total_pixels = mask_array.size
        porosity_ratio = pore_pixels / total_pixels
        print(f"- Estimated porosity ratio (thresh={threshold:.6f}): {porosity_ratio:.3f}")
    
    return image_array, mask_array

class XrayPorosityDataset(Dataset):
    """
    Custom dataset for X-ray tomography porosity segmentation
    Handles temporal sequences and applies appropriate preprocessing
    
    IMPROVED: Uses Otsu's automatic thresholding instead of median threshold
    for better pore segmentation quality
    """
    
    def __init__(self, image_paths, mask_paths, transform=None, target_size=(512, 512), 
                 augment=False, use_otsu=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_size = target_size
        self.augment = augment
        self.use_otsu = use_otsu  # Use Otsu thresholding (improved) vs median (baseline)
        
        # Preprocessing transforms
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size, transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        
        # Mask preprocessing (use nearest neighbor for masks to preserve labels)
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
        
        # Convert mask to binary using OTSU thresholding (IMPROVED!)
        if self.use_otsu:
            # Otsu's method automatically finds optimal threshold
            threshold = threshold_otsu(mask)
            mask_binary = (mask > threshold).astype(np.float32)
        else:
            # Baseline: median threshold (kept for comparison)
            mask_binary = (mask > np.median(mask)).astype(np.float32)
        
        # Apply preprocessing transforms
        image = self.base_transform(image)
        mask_binary = self.mask_transform(mask_binary)
        
        # Apply augmentations if enabled
        if self.augment:
            # Create seed for synchronized augmentation
            seed = torch.randint(0, 2**32, (1,)).item()
            
            # Apply same augmentation to both image and mask
            torch.manual_seed(seed)
            image = self.augment_transform(image)
            
            torch.manual_seed(seed)
            mask_binary = self.augment_transform(mask_binary)
        
        # Convert mask to long tensor for CrossEntropyLoss
        mask_binary = (mask_binary.squeeze(0) > 0.5).long()
        
        return image, mask_binary

def create_datasets(test_size=0.2, val_size=0.1, target_size=(512, 512), batch_size=4, use_otsu=True):
    """
    Create train, validation, and test datasets with proper splits
    
    Args:
        use_otsu: If True, uses Otsu thresholding (improved). If False, uses median (baseline)
    """
    
    # Get all file paths
    image_paths = [str(IMAGE_DIR / f) for f in image_files]
    mask_paths = [str(MASK_DIR / f) for f in mask_files]
    
    # Split into train+val and test
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_paths, mask_paths, test_size=test_size, random_state=42, shuffle=True
    )
    
    # Split train into train and validation
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        train_imgs, train_masks, test_size=val_size/(1-test_size), random_state=42, shuffle=True
    )
    
    threshold_method = "Otsu (improved)" if use_otsu else "Median (baseline)"
    print(f"Dataset splits:")
    print(f"- Training: {len(train_imgs)} samples")
    print(f"- Validation: {len(val_imgs)} samples")
    print(f"- Testing: {len(test_imgs)} samples")
    print(f"- Threshold method: {threshold_method}")
    
    # Create datasets
    train_dataset = XrayPorosityDataset(
        train_imgs, train_masks, target_size=target_size, augment=True, use_otsu=use_otsu
    )
    val_dataset = XrayPorosityDataset(
        val_imgs, val_masks, target_size=target_size, augment=False, use_otsu=use_otsu
    )
    test_dataset = XrayPorosityDataset(
        test_imgs, test_masks, target_size=target_size, augment=False, use_otsu=use_otsu
    )
    
    # Create data loaders (num_workers=0 for Jupyter compatibility)
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
    
    return train_loader, val_loader, test_loader, (train_dataset, val_dataset, test_dataset)


def get_dataset():
    '''
    Use code from previous scripts to load data with the same augmentation and splits
    '''
    # Load dataset information
    image_files, mask_files = load_dataset_info()
    
    # Analyze first sample
    sample_image_path = IMAGE_DIR / image_files[0]
    sample_mask_path = MASK_DIR / mask_files[0]
    
    sample_image, sample_mask = analyze_sample_image(sample_image_path, sample_mask_path)
    
    TARGET_SIZE = (512, 512)
    BATCH_SIZE = 4
    
    train_loader, val_loader, test_loader, datasets = create_datasets(
        target_size=TARGET_SIZE, batch_size=BATCH_SIZE, use_otsu=True
    )
    
    print(f"✅ Datasets created successfully with Otsu thresholding!")
    print(f"Target image size: {TARGET_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    
    return train_loader, val_loader, test_loader, datasets
# --- 5. Training and Evaluation Loops ---

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

# --- 6. Main Execution ---

def main():
    """Loads data, trains the U-Net model with transfer learning, and saves the checkpoint."""
    print("--- PyTorch U-Net Porosity Segmentation Training Script (Transfer Learning) ---")

    # Load and prepare data
    #train_loader, val_loader, test_loader = load_and_prepare_data() #<-- mock function for testing, doesn't use image data
    # Dataset paths given by (in data loader functions)
    '''
    DATA_DIR = Path(".")
    IMAGE_DIR = DATA_DIR / "Original Images"
    MASK_DIR = DATA_DIR / "Original Masks"
    '''
    train_loader, val_loader, test_loader, datasets = get_datasets()
    
    # Create model and set up transfer learning
    model = create_unet_tl()
    model.to(config.device)
    
    # Define Loss, Optimizer, and Metrics
    criterion = dice_loss # Use the custom Dice Loss
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Print trainable parameters (should mainly be the decoder)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model Parameters: {num_total_params:,}")
    print(f"✨ Trainable Parameters (Decoder/Head only): {num_trainable_params:,} ({(num_trainable_params/num_total_params)*100:.2f}%)")


    # Training Loop
    print(f"\n🚀 Starting training for {config.epochs} epochs...")
    for epoch in range(config.epochs):
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_dice = evaluate_model(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
              
    print("✅ Training complete.")

    # Evaluation
    print("\n📊 Evaluating model on Test Set...")
    test_loss, test_dice = evaluate_model(model, test_loader, criterion)

    print(f"\n--- Test Results ---")
    print(f"  - Test Loss (Dice Loss): {test_loss:.4f}")
    print(f"  - Test Dice Coeff (F1-score): {test_dice:.4f}")

    # Model Saving
    print("\n💾 Saving model checkpoint...")
    # 1. Ensure the target directory exists
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    # 2. Save the model's state dictionary
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"🎉 PyTorch Model state dictionary successfully saved to: {MODEL_SAVE_PATH}")

    print("\n--- Script Finished ---")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        print("Please ensure you have all required libraries installed, especially 'segmentation-models-pytorch'.")
