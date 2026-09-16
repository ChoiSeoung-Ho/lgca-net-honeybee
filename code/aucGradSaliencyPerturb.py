import torch
from pytorch_grad_cam import GradCAM
from utils.utils import load_data

from PosNeg_auc_util.seed import set_seed
from PosNeg_auc_util.config import AUCConfig
from PosNeg_auc_util.eval import run_xai_pos_neg_auc_cv
from PosNeg_auc_util.load_model import build_model as model_loader

# 공통 설정 로드
cfg = AUCConfig()
set_seed(cfg.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cam_target_layers(model):
    # 1순위: ViT backbone의 마지막 block
    if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        blocks = getattr(model.backbone, "blocks", None)
        if blocks is not None and len(blocks) > 0:
            return [blocks[-1]]

    # 2순위: 마지막 Conv2d 레이어
    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is not None:
        return [last_conv]

    # 3순위: 마지막 모듈(최후의 안전장치)
    mods = list(model.modules())
    return [mods[-1]] if len(mods) > 1 else [model]


def get_gradcam_map(model, input_tensor):
    # Grad-CAM 설명맵 2D(H, W) 반환
    target_layers = get_cam_target_layers(model)
    cam = GradCAM(model=model, target_layers=target_layers)
    return cam(input_tensor=input_tensor, targets=None)[0, :]


def get_saliency_map(model, input_tensor):
    # Saliency: 입력 픽셀 변화가 출력에 미치는 영향(gradient) 계산
    model.eval()
    x = input_tensor.clone().detach().requires_grad_(True)
    output = model(x)
    # 현재 모델이 가장 자신있는 클래스
    idx = output.argmax(dim=1).item()

    model.zero_grad(set_to_none=True)
    # 선택된 클래스 점수를 기준으로 역전파
    output[0, idx].backward()

    # 채널별 gradient 절댓값 중 최대값을 픽셀 중요도로 사용
    return (
        torch.max(x.grad.data.abs(), dim=1)[0]
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )


if __name__ == "__main__":
    # 비교할 설명 기법 목록
    methods = {
        "GradCAM": get_gradcam_map,
        "Saliency": get_saliency_map,
    }

    # 논문식 Pos/Neg perturbation AUC 평가 실행
    # - POS: 중요한 픽셀부터 10~90% 제거
    # - NEG: 덜 중요한 픽셀부터 10~90% 제거
    # - 각 단계 Top-1 hit 곡선을 적분해 AUC 계산
    run_xai_pos_neg_auc_cv(
        cfg=cfg,
        load_data_fn=load_data,
        model_name="efficientnetv2_proposal",
        model_loader=model_loader,
        methods=methods,
        device=device,
        print_errors=True,
        out_suffix="pos_neg_perturb_auc.csv",
        write_cv_mean=True,
        write_all=True,
    )
