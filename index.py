import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import cv2

# --- XAI Libraries ---
# Install via: pip install lime grad-cam scikit-image opencv-python
try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    print("CRITICAL ERROR: Missing libraries.")
    print("Please run: pip install lime grad-cam scikit-image opencv-python")
    exit()

# ==========================================
# 1. GLOBAL CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
# UPDATE THIS PATH to where your 'chest_xray' folder is located
INPUT_PATH = 'chest_xray' 

# ==========================================
# 2. HELPER FUNCTIONS & MODEL SETUP
# ==========================================

def get_model():
    """
    Loads ResNet50, freezes layers, and adjusts the final layer for 2 classes.
    """
    # Load pre-trained weights
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Freeze initial layers to speed up training
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # 2 classes: Normal vs Pneumonia
    
    model = model.to(DEVICE)
    return model

def train_model(model, dataloaders, criterion, optimizer, dataset_sizes, num_epochs=1):
    """
    Standard PyTorch training loop.
    """
    print(f"Starting training on {DEVICE}...")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
    return model

# ==========================================
# 3. EXPLAINABILITY FUNCTIONS (Grad-CAM & LIME)
# ==========================================

def generate_gradcam(model, input_tensor, target_class=None):
    """
    Generates Grad-CAM heatmap using the last convolutional layer.
    """
    # Target Layer: Last convolutional block of ResNet
    target_layers = [model.layer4[-1]]
    
    # --- FIX FOR ATTRIBUTE ERROR ---
    # We must ensure the target layer's parameters require gradients,
    # otherwise PyTorch won't track them, and GradCAM returns None.
    for param in model.layer4[-1].parameters():
        param.requires_grad = True
    # -------------------------------

    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
    
    # Generate map
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :] # Take first sample
    
    return grayscale_cam

def lime_predict_fn(images):
    """
    LIME requires numpy arrays as input and outputs probabilities.
    """
    global model
    model.eval()
    
    batch_tensors = []
    for img in images:
        # HWC -> CHW, float conversion
        img_tensor = torch.tensor(img).float().permute(2, 0, 1)
        # Normalize
        img_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])(img_tensor)
        batch_tensors.append(img_tensor)
        
    batch_tensors = torch.stack(batch_tensors).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(batch_tensors)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
    return probs.cpu().numpy()

def generate_lime(image_path):
    explainer = lime_image.LimeImageExplainer()
    
    # Load and preprocess image for LIME (numpy)
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_np = np.array(img) / 255.0 # Scale to [0,1]
    
    # Generate explanation
    # reduced num_samples to 100 for speed, increase to 1000 for better quality
    explanation = explainer.explain_instance(
        img_np, 
        lime_predict_fn, 
        top_labels=2, 
        hide_color=0, 
        num_samples=100 
    )
    
    return explanation, img_np

# ==========================================
# 4. VISUALIZATION
# ==========================================

def visualize_results(image_paths, model, class_names, transforms_test):
    # Set plot size based on number of images
    fig, axes = plt.subplots(len(image_paths), 3, figsize=(15, 5 * len(image_paths)))
    
    # Fix for axes indexing if there is only 1 image (makes it a 2D array)
    if len(image_paths) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, img_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {img_path}")
        
        # 1. Prepare Inputs
        img_pil = Image.open(img_path).convert('RGB')
        input_tensor = transforms_test(img_pil).unsqueeze(0).to(DEVICE)
        
        # Get Prediction
        output = model(input_tensor)
        pred_idx = torch.argmax(output).item()
        pred_label = class_names[pred_idx]
        
        # 2. Grad-CAM
        grad_cam_mask = generate_gradcam(model, input_tensor, target_class=pred_idx)
        img_norm = np.array(img_pil.resize((224, 224))) / 255.0
        grad_cam_viz = show_cam_on_image(img_norm, grad_cam_mask, use_rgb=True)
        
        # 3. LIME
        lime_exp, lime_img = generate_lime(img_path)
        temp, mask = lime_exp.get_image_and_mask(
            lime_exp.top_labels[0], 
            positive_only=True, 
            num_features=5, 
            hide_rest=False
        )
        lime_viz = mark_boundaries(temp, mask)

        # 4. Plot
        # Original
        axes[i, 0].imshow(img_pil)
        axes[i, 0].set_title(f"Original\nPred: {pred_label}")
        axes[i, 0].axis('off')
        
        # Grad-CAM
        axes[i, 1].imshow(grad_cam_viz)
        axes[i, 1].set_title("Grad-CAM (Heatmap)")
        axes[i, 1].axis('off')
        
        # LIME
        axes[i, 2].imshow(lime_viz)
        axes[i, 2].set_title("LIME (Superpixels)")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

# ==========================================
# 5. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == '__main__':
    # --------------------------------------
    # A. Data Loading
    # --------------------------------------
    print("Initializing Data Loaders...")
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ]),
    }

    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Dataset path '{INPUT_PATH}' not found.")
        print("Please check where you extracted the 'chest_xray' folder.")
        exit()

    image_datasets = {x: datasets.ImageFolder(os.path.join(INPUT_PATH, x), data_transforms[x])
                      for x in ['train', 'test']}

    # num_workers=2 works on Windows only if inside 'if __name__ == "__main__":'
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x=='train'), num_workers=2)
                   for x in ['train', 'test']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    
    print(f"Classes found: {class_names}")

    # --------------------------------------
    # B. Model Training
    # --------------------------------------
    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # Train for 1 epoch
    model = train_model(model, dataloaders, criterion, optimizer, dataset_sizes, num_epochs=1)

    # --------------------------------------
    # C. Visualization & Explainability
    # --------------------------------------
    print("\nGenerating Explanations...")
    model.eval()
    
    import random
    test_pneumonia_dir = os.path.join(INPUT_PATH, 'test/PNEUMONIA')
    
    if os.path.exists(test_pneumonia_dir):
        # Filter for image files only
        all_files = os.listdir(test_pneumonia_dir)
        test_images = [os.path.join(test_pneumonia_dir, f) for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(test_images) > 0:
            # Select up to 3 images
            selected_images = random.sample(test_images, min(3, len(test_images)))
            visualize_results(selected_images, model, class_names, data_transforms['test'])
        else:
            print("No images found in test/PNEUMONIA folder.")
    else:
        print(f"Could not find folder: {test_pneumonia_dir}")