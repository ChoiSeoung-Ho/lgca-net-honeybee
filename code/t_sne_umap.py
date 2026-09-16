import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler

# ==========================
# 설정 부분
# ==========================
class_name = "you_foulbrood"  # 분석할 클래스명
base_dir = f"./data/{class_name}"
save_dir = f"./results/{class_name}"
os.makedirs(save_dir, exist_ok=True)

# ==========================
# 이미지 로드 함수
# ==========================
def load_images_from_folder(folder, label, img_size=(64, 64)):
    data = []
    labels = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            img_flat = img.flatten()
            data.append(img_flat)
            labels.append(label)
    return np.array(data), np.array(labels)

# ==========================
# 데이터 로드
# ==========================
normal_dir = os.path.join(base_dir, "normal")
abnormal_dir = os.path.join(base_dir, "abnormal")

normal_data, normal_labels = load_images_from_folder(normal_dir, "Normal")
abnormal_data, abnormal_labels = load_images_from_folder(abnormal_dir, "Abnormal")

X = np.vstack((normal_data, abnormal_data))
y = np.concatenate((normal_labels, abnormal_labels))

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================
# t-SNE 시각화
# ==========================
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for label, color in zip(["Normal", "Abnormal"], ["blue", "red"]):
    plt.scatter(X_tsne[y == label, 0], X_tsne[y == label, 1], label=label, alpha=0.6, s=20, color=color)

plt.title(f"t-SNE Visualization ({class_name})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"{class_name}_tsne.png"), dpi=300)
plt.close()

# ==========================
# UMAP 시각화
# ==========================
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
X_umap = reducer.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for label, color in zip(["Normal", "Abnormal"], ["blue", "red"]):
    plt.scatter(X_umap[y == label, 0], X_umap[y == label, 1], label=label, alpha=0.6, s=20, color=color)

plt.title(f"UMAP Visualization ({class_name})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"{class_name}_umap.png"), dpi=300)
plt.close()

print(f"✅ t-SNE, UMAP 시각화 결과가 저장되었습니다: {save_dir}")

