
import torch

def build_model(model_name: str, model_path: str, img_size: int, device):
    name = model_name.lower()

    if "mamba" in name:
        from models.mambavision import MambaVisionForImageClassification_v2
        model = MambaVisionForImageClassification_v2(
            num_labels=2,
            img_size=img_size,
            patch_size=16,
            hidden_dim=512,
            model_variant="tiny",
        )
    elif "efficient" in name:
        from models.efficientnetv2 import EfficientNetV2ForImageClassification_v2
        model = EfficientNetV2ForImageClassification_v2(
            num_labels=2,
            img_size=img_size,
            patch_size=16,
            hidden_dim=512,
            model_variant="s",
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model