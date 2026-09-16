import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

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
            mean_val = np.mean(img)  # 이미지 평균 밝기값
            data.append(mean_val)
            labels.append(label)
    return np.array(data), np.array(labels)

# ==========================
# 데이터 로드
# ==========================
normal_dir = os.path.join(base_dir, "normal")
abnormal_dir = os.path.join(base_dir, "abnormal")

normal_data, normal_labels = load_images_from_folder(normal_dir, "Normal")
abnormal_data, abnormal_labels = load_images_from_folder(abnormal_dir, "Abnormal")

# ==========================
# t-test 계산
# ==========================
t_stat, p_value = ttest_ind(normal_data, abnormal_data, equal_var=False)
print("t test and p value")
print(t_stat,p_value)
# ==========================
# Violin Plot 시각화
# ==========================
plt.figure(figsize=(12, 8))
sns.violinplot(data=[normal_data, abnormal_data], palette=["#4C72B0", "#DD8452"])
plt.xticks([0, 1], ["Normal", "Abnormal"], fontsize=14)
plt.ylabel("Mean Pixel Intensity", fontsize=14)
plt.title(f"Violin Plot of {class_name}", fontsize=16, pad=20)  # ✅ 제목과 그래프 간격 확보

# ==========================
# p-value 시각적 표시 (선 + 텍스트)
# ==========================
y_max = max(np.max(normal_data), np.max(abnormal_data))
y_min = min(np.min(normal_data), np.min(abnormal_data))
y_gap = (y_max - y_min)

# 선 위치 (그래프 위 여유 공간 확보)
bar_y = y_max + y_gap * 0.05
text_y = bar_y + y_gap * 0.05

# p-value 선
plt.plot([0, 0, 1, 1], [bar_y, bar_y + y_gap * 0.02, bar_y + y_gap * 0.02, bar_y],
         color="black", lw=1.5)

# p-value 텍스트
plt.text(0.5, text_y, f"p = {p_value:.3e}", ha="center", fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"{class_name}_violin_pvalue.png"), dpi=300)
plt.close()

print(f"✅ Violin plot 저장 완료: {save_dir}/{class_name}_violin_pvalue.png")

