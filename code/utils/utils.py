import os
import cv2
import torch
import argparse
import numpy as np
from sklearn.metrics import roc_curve, auc 
import pandas as pd  
import matplotlib.pyplot as plt  
from tensorflow.keras.utils import to_categorical # type: ignore[import] 
import seaborn as sns
from sklearn.metrics import confusion_matrix

def get_args():
    parser = argparse.ArgumentParser(description='class 입력') 
    parser.add_argument('--class_name', type=str, default='abnormal', help='클래스 이름을 입력하세요')
    args = parser.parse_args()
    return args

def load_data(class_name, img_size=(256, 256), device='cpu'):
    X, y = [], []
    data_path = f'./data/{class_name}/'

    for cls_name in ['normal', 'abnormal']:
        cls_path = os.path.join(data_path, cls_name)
        label = 0 if cls_name == 'normal' else 1

        for file_name in os.listdir(cls_path):
            image_path = os.path.join(cls_path, file_name)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(label)
 
    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.int64)
 
    X = np.transpose(X, (0, 3, 1, 2))
 
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)

    return X_tensor, y_tensor

def save_csv(model_name, acc, loss, recall, prec, f1, auc, class_name, time) : 
    acc = f'{acc[0]:.3f}({acc[1]:.3f}-{acc[2]:.3f})'
    loss = f'{loss[0]:.3f}({loss[1]:.3f}-{loss[2]:.3f})'
    recall = f'{recall[0]:.3f}({recall[1]:.3f}-{recall[2]:.3f})'
    prec = f'{prec[0]:.3f}({prec[1]:.3f}-{prec[2]:.3f})'
    f1 = f'{f1[0]:.3f}({f1[1]:.3f}-{f1[2]:.3f})'
    if auc is not None: 
        auc = f'{auc[0]:.3f}({auc[1]:.3f}-{auc[2]:.3f})'
    else:
        auc = 'Nan(Nan-Nan)'
    time = time.strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.exists(f'./results/{class_name}/metrics.csv'):
        os.makedirs(f'./results/{class_name}', exist_ok=True)

    header = ['model name', 'Accuracy', 'Loss', 'Recall', 'Precision', 'F1 score', 'AUROC', 'timestamp']
    new_row = [model_name, acc, loss, recall, prec, f1, auc, time]

    csv_filename = f'./results/{class_name}/metrics.csv'

    if not os.path.exists(csv_filename):
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
        df = pd.DataFrame([new_row], columns=header)
        df.to_csv(csv_filename, index=False, header=True)
    else:
        df = pd.read_csv(csv_filename)
    if model_name in df['model name'].values:
        df.loc[df['model name'] == model_name, ['Accuracy', 'Loss', 'Recall', 'Precision', 'F1 score', 'AUROC', 'timestamp']] = new_row[1:]
    else:
        df = pd.concat([df, pd.DataFrame([new_row], columns=header)], ignore_index=True)
    df.to_csv(csv_filename, index=False, header=True) 

def plot_confusion_matrix(y_test, y_pred, model_name, class_name, fold_num) :
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred_labels = np.argmax(y_pred, axis=1)
    else:
        y_pred_labels = y_pred 
    class_names = ['Normal', class_name] 

    confusion = confusion_matrix(y_test, y_pred_labels)

    cm_df = pd.DataFrame(confusion, index=class_names, columns=class_names)

    plt.figure(figsize=(10,8))
    sns.heatmap(cm_df, annot=True, cmap='Purples', fmt='d', cbar=False, annot_kws={"size": 60})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    os.makedirs(f'./results/{class_name}/cm', exist_ok=True)
    plt.savefig(f"./results/{class_name}/cm/{model_name}_{fold_num}.png")

def roc_plot(y_test, y_prob, model_name, class_name, fold_num, n_classes=None):
    if y_prob.ndim == 1:
        y_prob = np.stack([1 - y_prob, y_prob], axis=1)  # shape: [N, 2]
 
    y_test_oh = to_categorical(y_test, num_classes=2)

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red']
    class_labels = ['Normal', class_name]
 
    for i in range(2):
        fpr, tpr, _ = roc_curve(y_test_oh[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{class_labels[i]} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')

    os.makedirs(f'./results/{class_name}/roc', exist_ok=True)
    plt.savefig(f"./results/{class_name}/roc/{model_name}_{fold_num}.png")
    plt.close()