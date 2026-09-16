import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils.utils import get_args, load_data
from models.efficientnetv2 import EfficientNetV2ForImageClassification_v2
from models.mambavision import MambaVisionForImageClassification_v2
from utils.train_loop import evaluate_model
import torch.nn as nn
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import KFold
import random
import datetime
from sklearn.model_selection import train_test_split
from zoneinfo import ZoneInfo
from sklearn.model_selection import StratifiedKFold
SEED = 42


def setSeed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
setSeed()


# -----------------------------
# 0) 기본 세팅
# -----------------------------
os.makedirs("./model_save", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
print("=" * 30)
print(now.strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 30)

# -----------------------------
# 1) 입력 인자 로드 및 데이터 로드
# -----------------------------
args = get_args()
class_name = args.class_name
print(f"now training for class: {class_name}")

X, y = load_data(class_name)

# -----------------------------
# 2) 3-Fold로 데이터 분할 (test fold만 뽑아둠)
# -----------------------------


kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
test_sets = []

for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_test, y_test = X[test_index], y[test_index]
    test_sets.append((X_test, y_test))

X_test1, y_test1 = test_sets[0]
X_test2, y_test2 = test_sets[1]
X_test3, y_test3 = test_sets[2]

# -----------------------------
# 3) 사용할 모델 종류 선택
# -----------------------------
model_name = "efficientnetv2_proposal"
#model_name = "mambavision_proposal"

# -----------------------------
# 4) 테스트 데이터에 가우시안 노이즈 주입한 DataLoader 생성
#    ✅ noise level: 0.1 ~ 0.5 (0.1 간격)
# -----------------------------
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

def make_noise_loaders(X_test, y_test, noise_levels=NOISE_LEVELS):
    """
    입력:
      - X_test: (N,C,H,W) 텐서 (0~1 범위 가정)
      - y_test: (N,) 라벨
    출력:
      - noise_levels에 해당하는 Gaussian noise를 더한 DataLoader 리스트
    """

    dev = X_test.device
    dtype = X_test.dtype

    loaders = []
    for nl in noise_levels:
        noise = torch.randn_like(X_test, dtype=dtype, device=dev) * nl
        X_noisy = torch.clamp(X_test + noise, 0, 1)

        ds = TensorDataset(X_noisy, y_test)
        loaders.append(DataLoader(ds, batch_size=32, shuffle=False))

    return loaders

# -----------------------------
# 5) 노이즈별 성능 평가 함수
# -----------------------------
def sensitivity_test(loaders, model, noise_levels=NOISE_LEVELS, device="cuda"):
    """
    입력:
      - loaders: noise level별 DataLoader 리스트
      - model: 이미 학습된 분류 모델
    출력:
      - 각 노이즈 수준에서 metric을 계산한 DataFrame
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    results = []

    for nl, test_loader in zip(noise_levels, loaders):
        test_loss, test_acc, y_true, y_pred, y_prob, _ = evaluate_model(
            model, test_loader, criterion, device
        )

        recall = recall_score(y_true, y_pred, average="binary")
        precision = precision_score(y_true, y_pred, average="binary")
        f1 = f1_score(y_true, y_pred, average="binary")
        ba = balanced_accuracy_score(y_true, y_pred)

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        if y_prob.ndim == 1 or y_prob.shape[1] == 1:
            y_score = y_prob.flatten()
        else:
            y_score = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob.flatten()

        try:
            auroc = roc_auc_score(y_true, y_score)
        except Exception as e:
            print(f"[noise={nl}] AUROC 계산 오류: {e}")
            auroc = np.nan

        try:
            aupr = average_precision_score(y_true, y_score)
        except Exception as e:
            print(f"[noise={nl}] AUPR 계산 오류: {e}")
            aupr = np.nan

        results.append({
            "NoiseLevel": float(nl),
            "BA": float(ba),
            "Loss": float(test_loss),
            "Recall": float(recall),
            "Precision": float(precision),
            "F1 Score": float(f1),
            "AUROC": float(auroc) if not np.isnan(auroc) else np.nan,
            "AUPR": float(aupr) if not np.isnan(aupr) else np.nan,
        })

    return pd.DataFrame(results)

# -----------------------------
# 6) 모델 생성 (fold별 3개)
# -----------------------------
if model_name == "mambavision_proposal":
    model1 = MambaVisionForImageClassification_v2(num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant="tiny")
    model2 = MambaVisionForImageClassification_v2(num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant="tiny")
    model3 = MambaVisionForImageClassification_v2(num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant="tiny")
else:
    model1 = EfficientNetV2ForImageClassification_v2(num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant="s")
    model2 = EfficientNetV2ForImageClassification_v2(num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant="s")
    model3 = EfficientNetV2ForImageClassification_v2(num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant="s")

model1.to(device)
model2.to(device)
model3.to(device)

# -----------------------------
# 7) fold별 가중치 로드
# -----------------------------
model1.load_state_dict(torch.load(f"./model_save/{class_name}/fold0/{model_name}.pt", map_location=device, weights_only=False))
model2.load_state_dict(torch.load(f"./model_save/{class_name}/fold1/{model_name}.pt", map_location=device, weights_only=False))
model3.load_state_dict(torch.load(f"./model_save/{class_name}/fold2/{model_name}.pt", map_location=device, weights_only=False))

# -----------------------------
# 8) fold별 sensitivity 테스트 수행
# -----------------------------
loaders1 = make_noise_loaders(X_test1, y_test1)
loaders2 = make_noise_loaders(X_test2, y_test2)
loaders3 = make_noise_loaders(X_test3, y_test3)

df1 = sensitivity_test(loaders1, model1, device=device)
df2 = sensitivity_test(loaders2, model2, device=device)
df3 = sensitivity_test(loaders3, model3, device=device)

print("Fold 0 Sensitivity Test Results:")
print(df1)
print("\nFold 1 Sensitivity Test Results:")
print(df2)
print("\nFold 2 Sensitivity Test Results:")
print(df3)

# -----------------------------
# 9) CSV 저장 (한 파일에 3폴드 합치고 Fold 컬럼 추가)
# -----------------------------
save_dir = f"./results_sensitivity"
os.makedirs(save_dir, exist_ok=True)

all_df = pd.concat(
    [
        df1.assign(Fold=0),
        df2.assign(Fold=1),
        df3.assign(Fold=2),
    ],
    ignore_index=True
)

save_path = os.path.join(save_dir, f"{class_name}.csv")
all_df.to_csv(save_path, index=False)

print(f"\n✅ Saved CSV: {save_path}")
