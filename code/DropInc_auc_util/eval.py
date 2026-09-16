import os
import csv
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold

from pytorch_grad_cam.metrics.cam_mult_image import DropInConfidence, IncreaseInConfidence
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget


def _to_numpy_2d(exp_map):
    if torch.is_tensor(exp_map):
        arr = exp_map.detach().cpu().numpy()
    else:
        arr = np.array(exp_map)
    if arr.ndim != 2:
        raise ValueError("explanation_map must be 2D (H,W)")
    return arr


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def run_xai_drop_inc_cv(
    *,
    cfg,
    load_data_fn,
    model_name: str,
    model_loader,
    methods,
    device,
    print_errors: bool = True,
    out_suffix: str = "xai_drop_inc.csv",
    write_cv_mean: bool = True,
    write_all: bool = True,
):
    os.makedirs(cfg.out_dir, exist_ok=True)

    drop_metric = DropInConfidence()
    inc_metric = IncreaseInConfidence()

    for class_name in cfg.class_names:
        print(f"=== Processing Class: {class_name} ===")

        X_tensor, y_tensor = load_data_fn(class_name, img_size=(cfg.img_size, cfg.img_size), device="cpu")
        if len(X_tensor) == 0:
            print("  No samples found.\n")
            continue

        kf = KFold(n_splits=3,shuffle=True,random_state=cfg.seed)

        fold_results = {
            fold: {
                m: {
                    "drop_percent": [],
                    "inc_flag": [],
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

            eval_local = np.arange(len(X_test))
            if len(eval_local) == 0:
                print(f"  Fold {i}: no test samples")
                continue

            model_path = os.path.join(cfg.base_model_path, class_name, f"fold{i}", f"{model_name}.pt")
            if not os.path.exists(model_path):
                print(f"  Fold {i}: model not found -> {model_path}")
                continue

            model = model_loader(model_name, model_path, cfg.img_size, device)

            for local_idx in tqdm(eval_local, desc=f"{class_name} fold{i}", leave=False):
                x = X_test[local_idx].unsqueeze(0).to(device)
                y_true = int(y_test[local_idx].item())
                targets = [ClassifierOutputSoftmaxTarget(y_true)]

                with torch.no_grad():
                    base_prob = float(torch.softmax(model(x), dim=1)[0, y_true].item())

                for method_name, map_fn in methods.items():
                    try:
                        cam = _normalize_01(_to_numpy_2d(map_fn(model, x)))
                        cams = np.expand_dims(cam, axis=0)

                        # raw_drop = max(0, Y - O)
                        raw_drop = float(drop_metric(x, cams, targets, model)[0])
                        # inc_flag = 1 if O > Y else 0
                        inc_flag = float(inc_metric(x, cams, targets, model)[0])

                        # Average Drop% 수식: max(0, Y-O) / Y * 100
                        drop_percent = 0.0
                        if base_prob > 1e-12:
                            drop_percent = (raw_drop / base_prob) * 100.0

                        fold_results[i][method_name]["drop_percent"].append(drop_percent)
                        fold_results[i][method_name]["inc_flag"].append(inc_flag)
                    except Exception as e:
                        if print_errors:
                            print(f"[{method_name} ERROR] {class_name} fold{i} {type(e).__name__} {e}")

            counts = "  ".join(f"{m}={len(fold_results[i][m]['drop_percent'])}" for m in methods.keys())
            print(f"[COUNT] {class_name} fold{i}  {counts}")

        out_csv = os.path.join(cfg.out_dir, f"{class_name}_{out_suffix}")
        all_buf = {m: {"drop_percent": [], "inc_flag": []} for m in methods.keys()}
        fold_means = {m: {"drop_percent": [], "inc_flag": []} for m in methods.keys()}

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Fold", "Method", "AverageDropPercent", "IncreaseInConfidencePercent"])

            for fold in cfg.folds:
                for method_name in methods.keys():
                    drop_list = fold_results[fold][method_name]["drop_percent"]
                    inc_list = fold_results[fold][method_name]["inc_flag"]
                    if len(drop_list) == 0:
                        continue

                    drop_m = float(np.mean(drop_list))
                    inc_percent = float(np.mean(inc_list) * 100.0)

                    all_buf[method_name]["drop_percent"].extend(drop_list)
                    all_buf[method_name]["inc_flag"].extend(inc_list)
                    fold_means[method_name]["drop_percent"].append(drop_m)
                    fold_means[method_name]["inc_flag"].append(inc_percent)

                    writer.writerow([fold, method_name, drop_m, inc_percent])

            if write_cv_mean:
                for method_name in methods.keys():
                    if len(fold_means[method_name]["drop_percent"]) == 0:
                        continue
                    writer.writerow([
                        "CV_MEAN",
                        method_name,
                        float(np.mean(fold_means[method_name]["drop_percent"])),
                        float(np.mean(fold_means[method_name]["inc_flag"])),
                    ])

            if write_all:
                for method_name in methods.keys():
                    drop_list = all_buf[method_name]["drop_percent"]
                    inc_list = all_buf[method_name]["inc_flag"]
                    if len(drop_list) == 0:
                        continue
                    writer.writerow([
                        "ALL",
                        method_name,
                        float(np.mean(drop_list)),
                        float(np.mean(inc_list) * 100.0),
                    ])

        print(f"  Saved -> {out_csv}\n")
