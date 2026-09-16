import numpy as np
import torch

from lime.lime_image import LimeImageExplainer
import shap

from utils.utils import load_data

from PosNeg_auc_util.seed import set_seed
from PosNeg_auc_util.config import AUCConfig
from PosNeg_auc_util.load_model import build_model as model_loader
from PosNeg_auc_util.eval import run_xai_pos_neg_auc_cv

# 공통 설정 로드
cfg = AUCConfig()
set_seed(cfg.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_lime_map(model, input_tensor):
    # [1, C, H, W] -> [H, W, C] numpy 변환 (LIME 입력 형식)
    np_img = (
        input_tensor.squeeze(0)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    explainer = LimeImageExplainer()

    # LIME이 요구하는 "배치 입력 -> 클래스 확률" 함수
    def classifier_fn(x):
        with torch.no_grad():
            batch = torch.tensor(x).permute(0, 3, 1, 2).float().to(device)
            outputs = model(batch)
            return torch.softmax(outputs, dim=1).detach().cpu().numpy()

    # 현재 이미지에 대한 LIME 설명 생성
    explanation = explainer.explain_instance(
        np_img, classifier_fn, top_labels=1, hide_color=0, num_samples=100
    )

    # 가장 높은 클래스의 superpixel 가중치를 픽셀맵으로 펼침
    label = explanation.top_labels[0]
    dict_heatmap = dict(explanation.local_exp[label])
    heatmap = np.zeros(np_img.shape[:2], dtype=np.float32)
    for seg_id, weight in dict_heatmap.items():
        heatmap[explanation.segments == seg_id] = weight

    # 부호보다 "강도"만 쓰기 위해 절댓값 처리
    return np.abs(heatmap)


def get_shap_map(model, input_tensor):
    # 설명 대상 샘플
    x = input_tensor.clone().detach().requires_grad_(True)
    # SHAP GradientExplainer용 배경 샘플(간단 노이즈 2개)
    background = torch.cat(
        [(x.detach() + torch.randn_like(x) * 0.1) for _ in range(2)],
        dim=0,
    ).to(x.device).requires_grad_(True)

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(x)

    # 모델이 가장 자신있는 클래스 인덱스
    with torch.no_grad():
        pred_class = model(x).argmax(dim=1).item()

    # SHAP 출력 형태가 라이브러리/모델에 따라 달라서 케이스 분기
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

    # 채널 방향을 평균내서 최종 2D 중요도 맵(H, W) 생성
    if vals.ndim == 3 and vals.shape[0] == 3:
        heatmap = np.mean(np.abs(vals), axis=0)
    elif vals.ndim == 3 and vals.shape[2] == 3:
        heatmap = np.mean(np.abs(vals), axis=2)
    else:
        heatmap = np.abs(vals)

    return np.abs(heatmap)


if __name__ == "__main__":
    # 비교할 설명 기법 목록
    methods = {
        "LIME": get_lime_map,
        "SHAP": get_shap_map,
    }

    # 논문식 Pos/Neg perturbation AUC 평가 실행
    run_xai_pos_neg_auc_cv(
        cfg=cfg,
        load_data_fn=load_data,
        model_name="efficientnetv2_proposal",
        model_loader=model_loader,
        methods=methods,
        device=device,
        print_errors=True,
        out_suffix="lime_shap_pos_neg_perturb_auc.csv",
        write_cv_mean=True,
        write_all=True,
    )
