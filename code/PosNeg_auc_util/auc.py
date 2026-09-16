import numpy as np
import torch


# 이 파일은 "논문식 Pos/Neg Perturbation" 계산 핵심 코드입니다.
# 핵심 아이디어:
# 1) 설명맵에서 중요한 픽셀 순서를 만든다.
# 2) 픽셀을 10%~90%까지 점점 지운다(0으로 마스킹).
# 3) 지울 때마다 모델이 정답을 맞추는지(Top-1 hit) 본다.
# 4) 그 점들을 이어 만든 곡선을 적분해서 AUC를 구한다.

@torch.no_grad()
def top1_hit(model, x, target_class: int) -> float:
    # 모델이 예측한 클래스(가장 큰 점수의 클래스)
    pred = model(x).argmax(dim=1).item()
    # 정답 클래스와 같으면 1.0, 다르면 0.0
    return float(pred == int(target_class))


def _prepare_explanation_2d(explanation_map) -> np.ndarray:
    # 설명맵을 numpy 2차원(H, W)로 통일한다.
    if torch.is_tensor(explanation_map):
        exp = explanation_map.detach().cpu().numpy()
    else:
        exp = np.array(explanation_map)
    # 픽셀 중요도 맵은 반드시 2차원이어야 한다.
    if exp.ndim != 2:
        raise ValueError("explanation_map must be 2D (H,W)")
    return exp


def perturbation_hit_curve(
    model,
    img_tensor: torch.Tensor,
    explanation_map,
    target_class: int,
    erase_ratios=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    neg: bool = False,
):
    """
    논문식 perturbation hit 곡선 계산.
    - pos (neg=False): 중요한 픽셀부터 지우기
    - neg (neg=True) : 덜 중요한 픽셀부터 지우기
    """
    # 배치 1장짜리 이미지 [1, C, H, W]만 받도록 고정
    if img_tensor.ndim != 4 or img_tensor.shape[0] != 1:
        raise ValueError("img_tensor must be shape [1, C, H, W]")

    # 이미지 높이/너비와 총 픽셀 수
    _, _, h, w = img_tensor.shape
    total = h * w
    # 설명맵 준비
    exp = _prepare_explanation_2d(explanation_map)

    # 중요도 큰 픽셀 -> 작은 픽셀 순서 인덱스
    order = np.argsort(exp.reshape(-1))[::-1]  # high -> low
    # neg=True면 순서를 뒤집어서 작은 값부터 지운다.
    if neg:
        order = order[::-1]  # low -> high

    # 각 제거 비율(10%, 20%, ... 90%)에서의 정답 여부 저장
    curves = []
    for r in erase_ratios:
        # 이번 단계에서 지울 픽셀 개수
        k = int(total * float(r))
        # 중요도 순서대로 앞에서 k개 선택
        idxs = order[:k]
        # 원본 이미지를 복사해서 마스킹 적용
        masked = img_tensor.clone()
        # 1차원 인덱스를 (행, 열)로 바꾸기
        rows = idxs // w
        cols = idxs % w
        # 선택된 픽셀을 0으로 지운다(모든 채널 동일 위치)
        masked[:, :, rows, cols] = 0.0
        # 현재 단계에서 정답 맞췄는지 기록(1 또는 0)
        curves.append(top1_hit(model, masked, target_class))

    # x: 제거 비율, y: 단계별 정답률(여기선 샘플 1개라 0/1)
    x = np.asarray(erase_ratios, dtype=np.float32)
    y = np.asarray(curves, dtype=np.float32)
    # 10%~90% 구간 사다리꼴 적분 AUC
    auc_val = float(np.trapz(y, x))
    return x, y, auc_val
