import os
import glob
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class_names = ["you_chalk_brood","you_foulbrood"]
base_model_path = "./model_save"
base_result_path = "./results_gradcam"
data_path = "./data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_target_layer(model, model_name):
    if model_name == "efficientnetv2_proposal":
        return [model.cnn_stem[3]]
    elif model_name == "convnextv2_proposal":
        return [model.backbone.stages[-1]]
    elif model_name == "mambavision_proposal":
        return [model.feature_extractor.levels[-1]]
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

def compute_gradcam(model, input_tensor, model_name):
    try:
        target_layers = get_target_layer(model, model_name)
        
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        
        # 실제 값 범위 유지하기 위해 직접 colormap 적용
        rgb_img = input_tensor.squeeze().permute(1,2,0).cpu().numpy()
        
        import matplotlib.cm as cm
        cmap = cm.get_cmap('jet')
        heatmap_colored = cmap(grayscale_cam)[:, :, :3]
        
        # 원본과 블렌딩
        alpha = 0.5
        visualization = alpha * rgb_img + (1 - alpha) * heatmap_colored
        visualization = np.clip(visualization, 0, 1)
        
        cam.activations_and_grads.release()
        
        return visualization
    except Exception as e:
        print(f"      GradCAM error: {str(e)}, using fallback visualization")
        rgb_img = input_tensor.squeeze().permute(1,2,0).cpu().numpy()
        return rgb_img

def get_top_confident_samples(model_name, target_class, img_paths, num_samples):
    print(f"  Calculating confidence scores for {len(img_paths)} images...")
    
    model_path = os.path.join(base_model_path, target_class, "fold1", f"{model_name}.pt")
    model = build_model(model_name, model_path)
    
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    
    scores = []
    for img_path in img_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                max_prob = probs.max().item()
                scores.append((img_path, max_prob))
        except Exception as e:
            print(f"    Error processing {img_path}: {str(e)}")
            scores.append((img_path, 0.0))
    
    del model
    torch.cuda.empty_cache()
    
    scores.sort(key=lambda x: x[1], reverse=True)
    top_samples = [path for path, score in scores[:num_samples]]
    
    print(f"  Selected top {len(top_samples)} confident samples")
    return top_samples

def save_fold_visualizations(image_path, gradcam_imgs, save_dir, model_name, subclass):
    base_name = os.path.basename(image_path).split('.')[0]
    img_dir = os.path.join(save_dir, subclass, base_name)
    os.makedirs(img_dir, exist_ok=True)

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_np = np.array(img)/255.0

    original_path = os.path.join(img_dir, f"{base_name}_original.png")
    img.save(original_path)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plt.suptitle(f"{model_name} GradCAM Comparison (Fold1~3)", fontsize=14)

    for i in range(3):
        axes[i].imshow(gradcam_imgs[i])
        axes[i].set_title(f"GradCAM Fold {i+1}")
        axes[i].axis('off')

    plt.tight_layout()
    compare_path = os.path.join(img_dir, f"{base_name}_{model_name}_gradcam_folds_compare.png")
    plt.savefig(compare_path, bbox_inches="tight")
    plt.close()

    avg_gradcam = np.mean(np.stack(gradcam_imgs), axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(avg_gradcam)
    ax.set_title("Average GradCAM")
    ax.axis('off')
    avg_path = os.path.join(img_dir, f"{base_name}_{model_name}_gradcam_average.png")
    plt.tight_layout()
    plt.savefig(avg_path, bbox_inches="tight")
    plt.close()

def build_model(model_name, model_path):
    if model_name == "efficientnetv2_proposal":
        from models.efficientnetv2 import EfficientNetV2ForImageClassification_v2
        model = EfficientNetV2ForImageClassification_v2(num_labels=2,img_size=224,patch_size=16,hidden_dim=512,model_variant='s')
    elif model_name == "convnextv2_proposal":
        from models.convnextv2 import ConvNeXtV2ForImageClassification_v2
        model = ConvNeXtV2ForImageClassification_v2(num_labels=2,img_size=224,patch_size=16,hidden_dim=512,model_variant='tiny')
    elif model_name == "mambavision_proposal":
        from models.mambavision import MambaVisionForImageClassification_v2
        model = MambaVisionForImageClassification_v2(num_labels=2,img_size=224,patch_size=16,hidden_dim=512,model_variant='tiny')
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    state_dict = torch.load(model_path, map_location=device,weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

def run_all_classes(model_name, num_samples_per_subclass=10):
    for target_class in class_names:
        print(f"Processing class: {target_class}")
        
        abnormal_paths = glob.glob(os.path.join(data_path, target_class, "abnormal", "*.jpg"))
        abnormal_paths += glob.glob(os.path.join(data_path, target_class, "abnormal", "*.png"))
        
        normal_paths = glob.glob(os.path.join(data_path, target_class, "normal", "*.jpg"))
        normal_paths += glob.glob(os.path.join(data_path, target_class, "normal", "*.png"))

        selected_paths = []
        
        if len(abnormal_paths) > 0:
            print(f"  Processing abnormal images ({len(abnormal_paths)} total)...")
            top_abnormal = get_top_confident_samples(model_name, target_class, abnormal_paths, num_samples_per_subclass)
            selected_paths.extend([(path, 'abnormal') for path in top_abnormal])
        
        if len(normal_paths) > 0:
            print(f"  Processing normal images ({len(normal_paths)} total)...")
            top_normal = get_top_confident_samples(model_name, target_class, normal_paths, num_samples_per_subclass)
            selected_paths.extend([(path, 'normal') for path in top_normal])

        if len(selected_paths) == 0:
            print(f"  No images found for class: {target_class}")
            continue

        print(f"  Processing {len(selected_paths)} selected images")

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])

        for idx, (img_path, subclass) in enumerate(selected_paths):
            print(f"  Processing image {idx+1}/{len(selected_paths)}: {os.path.basename(img_path)}")
            
            try:
                gradcam_imgs = []
                img = Image.open(img_path).convert("RGB")
                input_tensor = transform(img).unsqueeze(0).to(device)

                for fold_num in [0,1,2]:
                    model_path = os.path.join(base_model_path, target_class, f"fold{fold_num}", f"{model_name}.pt")
                    
                    if not os.path.exists(model_path):
                        print(f"    Model not found: {model_path}")
                        continue
                    
                    print(f"    Loading model from fold{fold_num}...")
                    model = build_model(model_name, model_path)

                    gradcam_img = compute_gradcam(model, input_tensor, model_name)
                    gradcam_imgs.append(gradcam_img)
                    
                    del model
                    torch.cuda.empty_cache()

                if len(gradcam_imgs) > 0:
                    result_path = os.path.join(base_result_path, target_class)
                    save_fold_visualizations(img_path, gradcam_imgs, result_path, model_name, subclass)
                    print(f"    Saved results")
                else:
                    print(f"    No valid folds found for this image")
                    
            except Exception as e:
                print(f"    Error processing {img_path}: {str(e)}")
                continue

if __name__ == "__main__":
    model_name = "efficientnetv2_proposal"
    num_samples_per_subclass = 10
    
    print(f"Starting GradCAM generation for {model_name}")
    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Samples per subclass (normal/abnormal): {num_samples_per_subclass}")
    print("-" * 50)
    
    run_all_classes(model_name, num_samples_per_subclass)
    
    print("\n" + "=" * 50)
    print("All processing complete!")

