import numpy as np
import torch

from lime.lime_image import LimeImageExplainer
import shap

from utils.utils import load_data

from DropInc_auc_util.seed import set_seed
from DropInc_auc_util.config import DropIncConfig
from DropInc_auc_util.load_model import build_model as model_loader
from DropInc_auc_util.eval import run_xai_drop_inc_cv

cfg = DropIncConfig()
set_seed(cfg.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_lime_map(model, input_tensor):
    np_img = (
        input_tensor.squeeze(0)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    explainer = LimeImageExplainer()

    def classifier_fn(x):
        with torch.no_grad():
            batch = torch.tensor(x).permute(0, 3, 1, 2).float().to(device)
            outputs = model(batch)
            return torch.softmax(outputs, dim=1).detach().cpu().numpy()

    explanation = explainer.explain_instance(
        np_img, classifier_fn, top_labels=1, hide_color=0, num_samples=100
    )

    label = explanation.top_labels[0]
    dict_heatmap = dict(explanation.local_exp[label])
    heatmap = np.zeros(np_img.shape[:2], dtype=np.float32)
    for seg_id, weight in dict_heatmap.items():
        heatmap[explanation.segments == seg_id] = weight

    return np.abs(heatmap)


def get_shap_map(model, input_tensor):
    x = input_tensor.clone().detach().requires_grad_(True)
    background = torch.cat(
        [(x.detach() + torch.randn_like(x) * 0.1) for _ in range(2)],
        dim=0,
    ).to(x.device).requires_grad_(True)

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(x)

    with torch.no_grad():
        pred_class = model(x).argmax(dim=1).item()

    if isinstance(shap_values, list):
        vals = shap_values[pred_class]
        vals = vals.detach().cpu().numpy() if torch.is_tensor(vals) else np.array(vals)
    else:
        vals = shap_values
        vals = vals.detach().cpu().numpy() if torch.is_tensor(vals) else np.array(vals)

    if vals.ndim == 5:
        vals = vals[0, :, :, :, pred_class]
    elif vals.ndim == 4:
        vals = vals[0]

    if vals.ndim == 3 and vals.shape[0] == 3:
        heatmap = np.mean(np.abs(vals), axis=0)
    elif vals.ndim == 3 and vals.shape[2] == 3:
        heatmap = np.mean(np.abs(vals), axis=2)
    else:
        heatmap = np.abs(vals)

    return np.abs(heatmap)


if __name__ == "__main__":
    methods = {
        "LIME": get_lime_map,
        "SHAP": get_shap_map,
    }

    run_xai_drop_inc_cv(
        cfg=cfg,
        load_data_fn=load_data,
        model_name="efficientnetv2_proposal",
        model_loader=model_loader,
        methods=methods,
        device=device,
        print_errors=True,
        out_suffix="lime_shap_drop_inc.csv",
        write_cv_mean=True,
        write_all=True,
    )
