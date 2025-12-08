import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from PIL import Image
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets

# XAI Libraries
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic

# ==========================================
# 1. CONFIGURATION
# ==========================================
# UPDATE THIS PATH to your dataset location
INPUT_PATH = 'chest_xray' 
MODEL_SAVE_PATH = 'best_densenet_chestxray_final.pth'

# EXECUTION SWITCH:
# True  = Load existing model weights (Skip training)
# False = Train model from scratch (Overwrite weights)
LOAD_FROM_FILE = True

BATCH_SIZE = 16
NUM_EPOCHS = 8
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

# Target Layer for DenseNet121 (The last dense block)
TARGET_LAYER_NAME = "features.denseblock4" 

# ==========================================
# 2. ADVANCED PREPROCESSING (CLAHE)
# ==========================================
class CLAHE_Transform:
    """
    Contrast Limited Adaptive Histogram Equalization.
    Enhances local contrast, helping model see subtle lung opacities.
    """
    def __call__(self, img):
        img_np = np.array(img)
        
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(final)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        CLAHE_Transform(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        CLAHE_Transform(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# ==========================================
# 3. MODEL SETUP & UTILS
# ==========================================
def build_model(num_classes=2):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    in_f = model.classifier.in_features
    model.classifier = nn.Linear(in_f, num_classes)
    return model.to(DEVICE)

def evaluate_model(model, dataloader, criterion):
    """
    Calculates Loss and Accuracy on the given dataloader (Test set).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    print("\n[INFO] Evaluating model performance...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    final_loss = running_loss / total
    final_acc = 100 * correct / total
    
    print("-" * 40)
    print(f"TEST RESULTS:\nLoss: {final_loss:.4f}\nAccuracy: {final_acc:.2f}%")
    print("-" * 40)
    return final_acc

def train_model(model, dataloaders, criterion, optimizer, dataset_sizes, num_epochs=3):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print(f"[INFO] Starting training on {DEVICE}...")
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save dynamically during training
                torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f'Best Val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

# ==========================================
# 4. EXPLAINABILITY: Grad-CAM++
# ==========================================
class GradCAMPlusPlus:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        self.activations = None
        self.gradients = None
        
        self.target_layer = self._find_layer(target_layer_name)
        if self.target_layer is None:
            raise ValueError(f"Layer {target_layer_name} not found.")
            
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_backward_hook(self._backward_hook)

    def _find_layer(self, name):
        module = self.model
        for part in name.split('.'):
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        
        if target_class is None:
            target_class = torch.argmax(probs)
            
        score = output[0, target_class]
        score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]
        
        grads_power_2 = gradients.pow(2)
        grads_power_3 = gradients.pow(3)
        sum_activations = torch.sum(activations, dim=(1, 2))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 + sum_activations[:, None, None] * grads_power_3 + eps)
        aij = torch.where(gradients != 0, aij, torch.zeros_like(aij))
        
        weights = torch.sum(aij * torch.relu(gradients), dim=(1, 2))
        
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
            
        cam_np = F.interpolate(cam.unsqueeze(0).unsqueeze(0), 
                               size=(input_tensor.size(2), input_tensor.size(3)), 
                               mode='bilinear', align_corners=False)
        cam_np = cam_np.squeeze().cpu().numpy()
        
        return cam_np, probs.detach().cpu().numpy()

# ==========================================
# 5. PIPELINE & VISUALIZATION
# ==========================================

def calculate_faithfulness(model, img_tensor, cam_mask, predicted_class):
    model.eval()
    with torch.no_grad():
        out = model(img_tensor)
        orig_prob = F.softmax(out, dim=1)[0, predicted_class].item()
    
    threshold = np.percentile(cam_mask, 80)
    binary_mask = torch.tensor(cam_mask >= threshold).float().to(DEVICE)
    
    masked_input = img_tensor.clone()
    for c in range(3):
        masked_input[0, c, :, :] = masked_input[0, c, :, :] * (1 - binary_mask)

    with torch.no_grad():
        out_masked = model(masked_input)
        masked_prob = F.softmax(out_masked, dim=1)[0, predicted_class].item()
        
    return orig_prob - masked_prob

def run_pipeline(model, dataloaders, class_names):
    gradcam_pp = GradCAMPlusPlus(model, TARGET_LAYER_NAME)
    explainer = lime_image.LimeImageExplainer()
    
    def lime_predict_fn(images):
        model.eval()
        batch_tensors = []
        for img in images:
            img_pil = Image.fromarray(img.astype('uint8'))
            t = data_transforms['test'](img_pil)
            batch_tensors.append(t)
        batch = torch.stack(batch_tensors).to(DEVICE)
        with torch.no_grad():
            outputs = model(batch)
            probs = F.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    test_path = os.path.join(INPUT_PATH, 'test', 'PNEUMONIA')
    if not os.path.exists(test_path):
        print(f"Path not found: {test_path}")
        return

    images = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('.jpeg')]
    
    import random
    if len(images) > 0:
        selected_images = random.sample(images, min(3, len(images)))
    else:
        print("No images found.")
        return
    
    fig, axes = plt.subplots(len(selected_images), 4, figsize=(20, 5 * len(selected_images)))
    if len(selected_images) == 1: axes = np.expand_dims(axes, axis=0)

    for i, img_path in enumerate(selected_images):
        print(f"Processing XAI for: {os.path.basename(img_path)}...")
        
        img_pil = Image.open(img_path).convert('RGB')
        input_tensor = data_transforms['test'](img_pil).unsqueeze(0).to(DEVICE)
        
        # 1. Grad-CAM++
        cam_mask, probs = gradcam_pp(input_tensor)
        pred_idx = np.argmax(probs)
        conf = probs[0, pred_idx]
        
        # 2. Faithfulness
        faith_score = calculate_faithfulness(model, input_tensor, cam_mask, pred_idx)
        
        # 3. LIME
        img_np_lime = np.array(img_pil.resize((IMAGE_SIZE, IMAGE_SIZE)))
        explanation = explainer.explain_instance(
            img_np_lime, 
            lime_predict_fn, 
            top_labels=2, 
            hide_color=0, 
            num_samples=500, 
            segmentation_fn=partial(slic, n_segments=80, compactness=10, sigma=1)
        )
        
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], 
            positive_only=True, 
            num_features=5, 
            hide_rest=False
        )
        
        # FIX: Scaling for mark_boundaries
        lime_viz = mark_boundaries(temp / 255.0, mask)

        # Plot
        axes[i, 0].imshow(img_pil.resize((IMAGE_SIZE, IMAGE_SIZE)))
        axes[i, 0].set_title(f"Pred: {class_names[pred_idx]} ({conf:.2f})")
        
        axes[i, 1].imshow(cam_mask, cmap='jet')
        axes[i, 1].set_title("Grad-CAM++ Heatmap")
        
        img_norm = np.array(img_pil.resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0
        heatmap_colored = cm.jet(cam_mask)[:, :, :3]
        overlay = 0.6 * img_norm + 0.4 * heatmap_colored
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"Faithfulness: {faith_score:.3f}")
        
        axes[i, 3].imshow(lime_viz)
        axes[i, 3].set_title("LIME (SLIC Segments)")
        
        for j in range(4): axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    # A. Setup
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Dataset path '{INPUT_PATH}' not found.")
        exit()

    image_datasets = {x: datasets.ImageFolder(os.path.join(INPUT_PATH, x), data_transforms[x])
                      for x in ['train', 'test']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x=='train'), num_workers=2)
                   for x in ['train', 'test']}
    
    class_names = image_datasets['train'].classes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    
    print(f"[INFO] Classes detected: {class_names}")
    
    # B. Model Initialization
    model = build_model(len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # C. Execution Logic (Train vs Load)
    model_file_exists = os.path.exists(MODEL_SAVE_PATH)

    if LOAD_FROM_FILE and model_file_exists:
        print(f"\n[INFO] Loading pre-trained weights from '{MODEL_SAVE_PATH}'...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    else:
        if LOAD_FROM_FILE and not model_file_exists:
            print(f"\n[WARN] '{MODEL_SAVE_PATH}' not found. Switching to training mode.")
        else:
            print("\n[INFO] Training model from scratch (LOAD_FROM_FILE=False)...")
            
        model = train_model(model, dataloaders, criterion, optimizer, dataset_sizes, num_epochs=NUM_EPOCHS)
        # Weights are saved automatically in train_model
        print(f"[INFO] Training complete. Model saved to '{MODEL_SAVE_PATH}'.")

    # D. Evaluation (Always runs)
    evaluate_model(model, dataloaders['test'], criterion)

    # E. Explainability Pipeline
    print("\n[INFO] Generating Explanations (Grad-CAM++ & LIME)...")
    run_pipeline(model, dataloaders, class_names)