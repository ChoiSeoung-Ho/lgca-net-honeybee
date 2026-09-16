import os
import numpy as np
import torch
import torch.nn as nn 
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

import datetime
from zoneinfo import ZoneInfo

from utils.utils import save_csv, plot_confusion_matrix, roc_plot

# =========================
# 1️⃣ 한 epoch 학습
# =========================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == y_batch).sum().item()
        total_samples += y_batch.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


# =========================
# 2️⃣ 평가 함수 (val/test 공용)
# =========================
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_labels, all_preds, all_probs, all_losses = [], [], [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)[:, 1] if outputs.size(1) > 1 else torch.sigmoid(outputs)

            total_correct += (preds == y_batch).sum().item()
            total_samples += y_batch.size(0)

            all_labels.extend(y_batch.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_losses.extend([loss.item()] * X_batch.size(0))  # batch loss 기록

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc, np.array(all_labels), np.array(all_preds), np.array(all_probs), np.array(all_losses)


# =========================
# 3️⃣ Bootstrap CI 계산
# =========================
def bootstrap_ci(metric_fn, y_true, y_pred, probs=None, n_bootstrap=1000, alpha=0.95):
    rng = np.random.default_rng()
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_true_sample = y_true[idx]
        y_pred_sample = y_pred[idx]
        if probs is not None:
            probs_sample = probs[idx]
            scores.append(metric_fn(y_true_sample, probs_sample))
        else:
            scores.append(metric_fn(y_true_sample, y_pred_sample))

    lower = np.percentile(scores, ((1 - alpha) / 2) * 100)
    upper = np.percentile(scores, (1 - (1 - alpha) / 2) * 100)
    return [np.mean(scores), lower, upper]


def bootstrap_ci_loss(values, n_bootstrap=1000, alpha=0.95):
    rng = np.random.default_rng()
    n = len(values)
    means = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        sample = values[idx]
        means.append(sample.mean())
    lower = np.percentile(means, ((1 - alpha) / 2) * 100)
    upper = np.percentile(means, (1 - (1 - alpha) / 2) * 100)
    return [np.mean(means), lower, upper]


# =========================
# 4️⃣ 전체 학습/평가 루프
# =========================
def train_full_loop(train_loader, val_loader, test_loader, model_dicts, class_name, fold_num,
                    lr=1e-3, epochs=50, patience=10, device='cuda'):

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    for model_name, model in model_dicts.items():
        print(f"\n=== Training {model_name} on Fold {fold_num} ===")

        save_path = f'./model_save/{class_name}/fold{fold_num}/{model_name}.pt'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_acc = 0
        patience_counter = 0
        
        try:
            model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
            print(f"Loaded existing model for {model_name}")
            training_needed = False
        except Exception as e:
            print(f"No saved model found, training from scratch: {e}")
            training_needed = True

        best_val_acc = 0
        patience_counter = 0

        # -----------------------------
        # 학습
        # -----------------------------
        if training_needed:
            for epoch in range(epochs):
                train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
                val_loss, val_acc, _, _, _, _ = evaluate_model(model, val_loader, criterion, device)

                print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    torch.save(model.state_dict(), save_path)
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        # -----------------------------
        # 테스트 평가
        # -----------------------------
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=False))
        test_loss, test_acc, y_true, y_pred, y_prob, all_losses = evaluate_model(model, test_loader, criterion, device)

        # Bootstrap metrics
        test_loss_ci = bootstrap_ci_loss(all_losses)
        test_acc_ci = bootstrap_ci(balanced_accuracy_score, y_true, y_pred)
        test_recall_ci = bootstrap_ci(lambda y, yhat: recall_score(y, yhat, average='macro'), y_true, y_pred)
        test_prec_ci = bootstrap_ci(lambda y, yhat: precision_score(y, yhat, average='macro'), y_true, y_pred)
        test_f1_ci = bootstrap_ci(lambda y, yhat: f1_score(y, yhat, average='macro'), y_true, y_pred)

        try:
            test_auc_ci = bootstrap_ci(lambda y, p: roc_auc_score(y, p, multi_class='ovr'), y_true, y_pred, probs=y_prob)
        except Exception as e:
            print(f"ROC AUC calculation failed: {e}")
            test_auc_ci = [np.nan, np.nan, np.nan]

        # -----------------------------
        # 결과 저장
        # -----------------------------
        now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
        save_csv(f'{model_name}_{fold_num}', test_acc_ci, test_loss_ci, test_recall_ci, 
                 test_prec_ci, test_f1_ci, test_auc_ci, class_name, now)
 
        plot_confusion_matrix(y_true, y_pred, model_name, class_name, fold_num)
        roc_plot(y_true, y_prob, model_name, class_name, fold_num)
