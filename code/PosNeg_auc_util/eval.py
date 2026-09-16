import os
import csv
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from .auc import perturbation_hit_curve


# 이 함수는 "데이터 전체 -> 폴드별 평가 -> CSV 저장"을 한 번에 수행한다.
# 쉽게 말해:
# 1) 데이터 불러오기
# 2) 폴드별 테스트셋 순회
# 3) 샘플마다 설명맵 만들기
# 4) Pos/Neg perturbation AUC 계산
# 5) 평균값을 CSV로 저장
def run_xai_pos_neg_auc_cv(
    *,
    cfg,
    load_data_fn,
    model_name: str,
    model_loader,
    methods,
    device,
    print_errors: bool = True,
    out_suffix: str = "xai_pos_neg_auc.csv",
    write_cv_mean: bool = True,
    write_all: bool = True,
):
    # 결과 폴더가 없으면 만든다.
    os.makedirs(cfg.out_dir, exist_ok=True)

    # 논문과 동일하게 10%~90% 구간만 사용
    erase_ratios = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    # class_name 루프:
    # 현재 프로젝트는 class 단위로 모델 파일 경로가 나뉘어 있어 class별로 돈다.
    for class_name in cfg.class_names:
        print(f"=== Processing Class: {class_name} ===")

        # 데이터 로드 (tensor 형태)
        X_tensor, y_tensor = load_data_fn(class_name, img_size=(cfg.img_size, cfg.img_size), device="cpu")
        if len(X_tensor) == 0:
            print("  No samples found.\n")
            continue

        kf = KFold(n_splits=3,shuffle=True,random_state=cfg.seed)

        # fold_results 구조:
        # fold -> method -> pos/neg AUC 리스트, pos/neg 곡선 리스트
        fold_results = {
            fold: {
                m: {
                    "pos_auc": [],
                    "neg_auc": [],
                    "pos_curve": [],
                    "neg_curve": [],
                } for m in methods.keys()
            }
            for fold in cfg.folds
        }

        for i, (_, test_index) in enumerate(kf.split(X_tensor)):
            if i not in cfg.folds:
                continue

            # 현재 폴드 테스트셋
            X_test = X_tensor[test_index]
            y_test = y_tensor[test_index]

            # 논문식 설정: 특정 클래스 샘플만 고르지 않고 테스트 전체를 평가
            eval_local = np.arange(len(X_test))
            if len(eval_local) == 0:
                print(f"  Fold {i}: no test samples")
                continue

            # 폴드별 모델 경로
            model_path = os.path.join(cfg.base_model_path, class_name, f"fold{i}", f"{model_name}.pt")
            if not os.path.exists(model_path):
                print(f"  Fold {i}: model not found -> {model_path}")
                continue

            # 모델 로드
            model = model_loader(model_name, model_path, cfg.img_size, device)

            # 테스트 샘플 하나씩 평가
            for local_idx in tqdm(eval_local, desc=f"{class_name} fold{i}", leave=False):
                x = X_test[local_idx].unsqueeze(0).to(device)
                # Top-1 정답 비교에 사용할 GT 클래스
                y_true = int(y_test[local_idx].item())

                # 방법(GradCAM, Saliency, LIME, SHAP...)별로 반복
                for method_name, map_fn in methods.items():
                    try:
                        # 설명맵 생성
                        exp_map = map_fn(model, x)
                        # Pos perturbation AUC
                        _, pos_curve, pos_auc = perturbation_hit_curve(
                            model=model,
                            img_tensor=x,
                            explanation_map=exp_map,
                            target_class=y_true,
                            erase_ratios=erase_ratios,
                            neg=False,
                        )
                        # Neg perturbation AUC
                        _, neg_curve, neg_auc = perturbation_hit_curve(
                            model=model,
                            img_tensor=x,
                            explanation_map=exp_map,
                            target_class=y_true,
                            erase_ratios=erase_ratios,
                            neg=True,
                        )

                        # 샘플 단위 결과 저장
                        fold_results[i][method_name]["pos_auc"].append(pos_auc)
                        fold_results[i][method_name]["neg_auc"].append(neg_auc)
                        fold_results[i][method_name]["pos_curve"].append(pos_curve)
                        fold_results[i][method_name]["neg_curve"].append(neg_curve)
                    except Exception as e:
                        if print_errors:
                            print(f"[{method_name} ERROR] {class_name} fold{i} {type(e).__name__} {e}")

            counts = "  ".join(f"{m}={len(fold_results[i][m]['pos_auc'])}" for m in methods.keys())
            print(f"[COUNT] {class_name} fold{i}  {counts}")

        # class별 CSV 파일 경로
        out_csv = os.path.join(cfg.out_dir, f"{class_name}_{out_suffix}")

        # 전체 통합 평균용 버퍼
        all_buf = {m: {"pos_auc": [], "neg_auc": []} for m in methods.keys()}
        # fold 평균의 평균(CV_MEAN) 계산용 버퍼
        fold_means = {m: {"pos_auc": [], "neg_auc": []} for m in methods.keys()}

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # PosAUC_x100 / NegAUC_x100:
            # 논문 표처럼 퍼센트 스케일로 보기 쉽게 같이 저장
            writer.writerow([
                "Fold",
                "Method",
                "AUC_XRange",
                "PosAUC",
                "NegAUC",
                "PosAUC_x100",
                "NegAUC_x100",
                "PosCurveMean",
                "NegCurveMean",
            ])

            for fold in cfg.folds:
                for method_name in methods.keys():
                    pos_list = fold_results[fold][method_name]["pos_auc"]
                    neg_list = fold_results[fold][method_name]["neg_auc"]
                    # 참고용 곡선 평균도 저장(재시각화 쉽게 하려고)
                    pos_curve_list = fold_results[fold][method_name]["pos_curve"]
                    neg_curve_list = fold_results[fold][method_name]["neg_curve"]
                    if len(pos_list) == 0:
                        continue

                    # fold 내부 샘플 평균
                    pos_m = float(np.mean(pos_list))
                    neg_m = float(np.mean(neg_list))
                    # fold 내부 곡선 평균(길이 9)
                    pos_curve_m = np.mean(np.stack(pos_curve_list, axis=0), axis=0)
                    neg_curve_m = np.mean(np.stack(neg_curve_list, axis=0), axis=0)

                    all_buf[method_name]["pos_auc"].extend(pos_list)
                    all_buf[method_name]["neg_auc"].extend(neg_list)
                    fold_means[method_name]["pos_auc"].append(pos_m)
                    fold_means[method_name]["neg_auc"].append(neg_m)

                    writer.writerow([
                        fold,
                        method_name,
                        # 적분 x축 구간을 파일에 명시
                        "0.1-0.9",
                        pos_m,
                        neg_m,
                        pos_m * 100.0,
                        neg_m * 100.0,
                        ";".join(f"{v:.6f}" for v in pos_curve_m.tolist()),
                        ";".join(f"{v:.6f}" for v in neg_curve_m.tolist()),
                    ])

            if write_cv_mean:
                # 각 fold 평균값들의 평균
                for method_name in methods.keys():
                    if len(fold_means[method_name]["pos_auc"]) == 0:
                        continue
                    writer.writerow([
                        "CV_MEAN",
                        method_name,
                        "0.1-0.9",
                        float(np.mean(fold_means[method_name]["pos_auc"])),
                        float(np.mean(fold_means[method_name]["neg_auc"])),
                        float(np.mean(fold_means[method_name]["pos_auc"])) * 100.0,
                        float(np.mean(fold_means[method_name]["neg_auc"])) * 100.0,
                        "",
                        "",
                    ])

            if write_all:
                # 폴드 구분 없이 샘플 전체 평균
                for method_name in methods.keys():
                    pos_list = all_buf[method_name]["pos_auc"]
                    neg_list = all_buf[method_name]["neg_auc"]
                    if len(pos_list) == 0:
                        continue
                    writer.writerow([
                        "ALL",
                        method_name,
                        "0.1-0.9",
                        float(np.mean(pos_list)),
                        float(np.mean(neg_list)),
                        float(np.mean(pos_list)) * 100.0,
                        float(np.mean(neg_list)) * 100.0,
                        "",
                        "",
                    ])

        print(f"  Saved -> {out_csv}\n")
