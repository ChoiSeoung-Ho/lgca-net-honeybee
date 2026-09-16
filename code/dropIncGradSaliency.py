import torch
from pytorch_grad_cam import GradCAM
from utils.utils import load_data

from DropInc_auc_util.seed import set_seed
from DropInc_auc_util.config import DropIncConfig
from DropInc_auc_util.eval import run_xai_drop_inc_cv
from DropInc_auc_util.load_model import build_model as model_loader

cfg = DropIncConfig()
set_seed(cfg.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cam_target_layers(model):
    if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        blocks = getattr(model.backbone, "blocks", None)
        if blocks is not None and len(blocks) > 0:
            return [blocks[-1]]

    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is not None:
        return [last_conv]

    mods = list(model.modules())
    return [mods[-1]] if len(mods) > 1 else [model]


def get_gradcam_map(model, input_tensor):
    target_layers = get_cam_target_layers(model)
    cam = GradCAM(model=model, target_layers=target_layers)
    return cam(input_tensor=input_tensor, targets=None)[0, :]


def get_saliency_map(model, input_tensor):
    model.eval()
    x = input_tensor.clone().detach().requires_grad_(True)
    output = model(x)
    idx = output.argmax(dim=1).item()

    model.zero_grad(set_to_none=True)
    output[0, idx].backward()

    return (
        torch.max(x.grad.data.abs(), dim=1)[0]
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )


if __name__ == "__main__":
    methods = {
        "GradCAM": get_gradcam_map,
        "Saliency": get_saliency_map,
    }

    run_xai_drop_inc_cv(
        cfg=cfg,
        load_data_fn=load_data,
        model_name="efficientnetv2_proposal",
        model_loader=model_loader,
        methods=methods,
        device=device,
        print_errors=True,
        out_suffix="grad_saliency_drop_inc.csv",
        write_cv_mean=True,
        write_all=True,
    )
