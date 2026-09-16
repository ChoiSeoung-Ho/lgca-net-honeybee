import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from utils.utils import get_args, load_data 
from sklearn.model_selection import train_test_split, KFold  
from utils.train_loop import evaluate_model
import torch.nn as nn
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, balanced_accuracy_score
import datetime
from zoneinfo import ZoneInfo
import random, os 

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

os.makedirs('./model_save', exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
print('='*30)
print(now.strftime('%Y-%m-%d %H:%M:%S'))
print('='*30)
 
args = get_args()
class_name = args.class_name 
print(f'now training for class: {class_name}')

##################################
# 모델 택 1
##################################
model_name = 'efficientnetv2_proposal'
#model_name = 'mambavision_proposal'




X, y = load_data(class_name)

kf = KFold(n_splits=3, shuffle=True, random_state=SEED)

test_sets = []
for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_test, y_test = X[test_index], y[test_index]
    test_sets.append((X_test, y_test))

X_test1, y_test1 = test_sets[0]
X_test2, y_test2 = test_sets[1]
X_test3, y_test3 = test_sets[2]

# =====================================================================
# 1. 비율에 맞춰 데이터를 추출하는 함수 (PyTorch 호환, 에러 방지 적용)
# =====================================================================
def extract_data(X_test, y_test, ratio_class0, ratio_class1):  
    if not torch.is_tensor(X_test):
        X_test = torch.tensor(X_test)
    if not torch.is_tensor(y_test):
        y_test = torch.tensor(y_test)

    idx_class0 = torch.where(y_test == 0)[0]
    idx_class1 = torch.where(y_test == 1)[0]
    
    count_0 = len(idx_class0)
    count_1 = len(idx_class1)

    p0 = ratio_class0 / 100.0
    p1 = ratio_class1 / 100.0

    max_total_by_0 = count_0 / p0 if p0 > 0 else float('inf')
    max_total_by_1 = count_1 / p1 if p1 > 0 else float('inf')
    
    max_total = int(min(max_total_by_0, max_total_by_1))

    target_count0 = int(max_total * p0)
    target_count1 = int(max_total * p1)

    print(f"목표비율[{ratio_class0}:{ratio_class1}] -> 실제샘플수: class0={target_count0}개, class1={target_count1}개 (총 {target_count0+target_count1}개)")
    print('='*35)

    perm_0 = torch.randperm(count_0)[:target_count0]
    chosen_class0 = idx_class0[perm_0]

    perm_1 = torch.randperm(count_1)[:target_count1]
    chosen_class1 = idx_class1[perm_1]

    chosen_idx = torch.cat([chosen_class0, chosen_class1])
    shuffle_perm = torch.randperm(len(chosen_idx))
    chosen_idx = chosen_idx[shuffle_perm]

    return X_test[chosen_idx], y_test[chosen_idx]

# =====================================================================
# (비율별 데이터로더 생성기)
# =====================================================================
def make_ratio_data(X_test, y_test): 
    data_10_90_X, data_10_90_y = extract_data(X_test, y_test, 10, 90)
    data_30_70_X, data_30_70_y = extract_data(X_test, y_test, 30, 70)
    data_50_50_X, data_50_50_y = extract_data(X_test, y_test, 50, 50)
    data_70_30_X, data_70_30_y = extract_data(X_test, y_test, 70, 30)
    data_90_10_X, data_90_10_y = extract_data(X_test, y_test, 90, 10)

    test_dataset10_90 = TensorDataset(data_10_90_X.clone().detach(), data_10_90_y.clone().detach())
    test_dataset30_70 = TensorDataset(data_30_70_X.clone().detach(), data_30_70_y.clone().detach())
    test_dataset50_50 = TensorDataset(data_50_50_X.clone().detach(), data_50_50_y.clone().detach())
    test_dataset70_30 = TensorDataset(data_70_30_X.clone().detach(), data_70_30_y.clone().detach())
    test_dataset90_10 = TensorDataset(data_90_10_X.clone().detach(), data_90_10_y.clone().detach())

    test_loader10_90 = DataLoader(test_dataset10_90, batch_size=32, shuffle=False)
    test_loader30_70 = DataLoader(test_dataset30_70, batch_size=32, shuffle=False)
    test_loader50_50 = DataLoader(test_dataset50_50, batch_size=32, shuffle=False)
    test_loader70_30 = DataLoader(test_dataset70_30, batch_size=32, shuffle=False)
    test_loader90_10 = DataLoader(test_dataset90_10, batch_size=32, shuffle=False)
    
    return test_loader10_90, test_loader30_70, test_loader50_50, test_loader70_30, test_loader90_10  

# =====================================================================
# 3. 안정성 평가 함수 
# =====================================================================
def stability_test(test_loader10_90, test_loader30_70, test_loader50_50, test_loader70_30, test_loader90_10, model, device='cuda'):
    ratios = ["10:90", "30:70", "50:50", "70:30", "90:10"]
    datasets = [test_loader10_90, test_loader30_70, test_loader50_50, test_loader70_30, test_loader90_10]
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    criterion = nn.CrossEntropyLoss()
    results = []

    for ratio, test_loader in zip(ratios, datasets): 
        test_loss, test_acc, y_true, y_pred, y_prob, all_losses = evaluate_model(model, test_loader, criterion, device)
        
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
            print(f"[{ratio}] AUROC 계산 오류: {e}")
            auroc = np.nan

        try:
            aupr = average_precision_score(y_true, y_score)
        except Exception as e:
            print(f"[{ratio}] AUPR 계산 오류: {e}")
            aupr = np.nan 

        results.append({ 
            "Ratio": ratio,
            "Acc": test_acc,
            "BA": ba,  
            "AUROC": auroc,
            "AUPR": aupr,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall
        })

    return pd.DataFrame(results)

# =====================================================================
# 4. 모델 로드 및 실행부
# =====================================================================
if model_name == 'mambavision_proposal':
    from models.mambavision import MambaVisionForImageClassification_v2
    model1 = MambaVisionForImageClassification_v2(num_labels=2,img_size=224,patch_size=16,hidden_dim=512,model_variant='tiny')
    model2 = MambaVisionForImageClassification_v2(num_labels=2,img_size=224,patch_size=16,hidden_dim=512,model_variant='tiny')
    model3 = MambaVisionForImageClassification_v2(num_labels=2,img_size=224,patch_size=16,hidden_dim=512,model_variant='tiny')
elif model_name == 'efficientnetv2_proposal':
    from models.efficientnetv2 import EfficientNetV2ForImageClassification_v2
    model1 = EfficientNetV2ForImageClassification_v2(num_labels=2,img_size=224,patch_size=16,hidden_dim=512,model_variant='s')
    model2 = EfficientNetV2ForImageClassification_v2(num_labels=2,img_size=224,patch_size=16,hidden_dim=512,model_variant='s')
    model3 = EfficientNetV2ForImageClassification_v2(num_labels=2,img_size=224,patch_size=16,hidden_dim=512,model_variant='s')
    
model1.to(device)
model2.to(device)
model3.to(device)

model1.load_state_dict(torch.load(f'./model_save/{class_name}/fold0/{model_name}.pt', map_location=device, weights_only=False)) 
model2.load_state_dict(torch.load(f'./model_save/{class_name}/fold1/{model_name}.pt', map_location=device, weights_only=False))
model3.load_state_dict(torch.load(f'./model_save/{class_name}/fold2/{model_name}.pt', map_location=device, weights_only=False))

df1 = stability_test(*make_ratio_data(X_test1, y_test1), model1, device=device)
df2 = stability_test(*make_ratio_data(X_test2, y_test2), model2, device=device)
df3 = stability_test(*make_ratio_data(X_test3, y_test3), model3, device=device)

save_root = "./results_stability"
os.makedirs(save_root, exist_ok=True)

df_all = pd.concat(
    [df1.assign(Fold=0), df2.assign(Fold=1), df3.assign(Fold=2)],
    ignore_index=True
)

df_all.insert(0, "Class", class_name)
df_all.insert(1, "Model", model_name)

out_path = os.path.join(save_root, f"{class_name}.csv")
df_all.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"[Saved] {out_path}")

print('Fold 0 Stability Test Results:')
print(df1)
print('\nFold 1 Stability Test Results:')
print(df2)
print('\nFold 2 Stability Test Results:')
print(df3)
