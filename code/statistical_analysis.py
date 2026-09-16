import os
import argparse
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import pandas as pd
import seaborn as sns

def rgb_to_lab(image):
    """RGB 이미지를 LAB 색공간으로 변환"""
    rgb = np.array(image)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return lab

def extract_color_features(image):
    """이미지에서 L*, a*, b* 평균값 추출"""
    lab = rgb_to_lab(image)
    L_mean = np.mean(lab[:, :, 0])
    a_mean = np.mean(lab[:, :, 1])
    b_mean = np.mean(lab[:, :, 2])
    return L_mean, a_mean, b_mean

def extract_morphological_features(image):
    """
    이미지에서 형태학적 특징 추출
    (주의: 피부 데이터의 경우 단순 Thresholding으로 병변 검출이 어려울 수 있으나, 
     전체적인 통계적 경향성을 보기 위해 기존 로직 유지)
    """
    gray = np.array(image.convert('L'))
    # Otsu Thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return 0, 0, 0
    
    # 가장 큰 컨투어(영역) 기준
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)
    else:
        circularity = 0
    
    if len(largest_contour) >= 5:
        ellipse = cv2.fitEllipse(largest_contour)
        (_, (MA, ma), _) = ellipse
        if ma > 0:
            aspect_ratio = MA / ma
        else:
            aspect_ratio = 0
    else:
        aspect_ratio = 0
    
    return area, circularity, aspect_ratio

def load_and_extract_features(folder_path, label, max_samples=500):
    """폴더에서 이미지 로드 및 특징 추출"""
    features = []
    
    # 지원 확장자 및 파일 리스트 확인
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}")
        return []

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_extensions)]
    # 샘플링 
    files = files[:max_samples]
    
    for filename in tqdm(files, desc=f"Processing {label}"):
        img_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            L_mean, a_mean, b_mean = extract_color_features(img)
            area, circularity, aspect_ratio = extract_morphological_features(img)
            
            features.append({
                'L_mean': L_mean,
                'a_mean': a_mean,
                'b_mean': b_mean,
                'Area': area,
                'Circularity': circularity,
                'Aspect_ratio': aspect_ratio,
                'label': label
            })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    return features

def statistical_comparison(df, output_dir):
    """통계적 비교 수행 및 CSV 저장"""
    feature_cols = ['L_mean', 'a_mean', 'b_mean', 'Area', 'Circularity', 'Aspect_ratio']
    
    normal_data = df[df['label'] == 'normal']
    abnormal_data = df[df['label'] == 'abnormal']
    
    results = []
    for col in feature_cols:
        # 데이터가 충분한지 확인
        if len(normal_data) < 2 or len(abnormal_data) < 2:
            t_stat, p_value = np.nan, np.nan
        else:
            t_stat, p_value = stats.ttest_ind(normal_data[col], abnormal_data[col], equal_var=False)

        results.append({
            'Feature': col,
            'Normal_mean': normal_data[col].mean() if not normal_data.empty else 0,
            'Normal_std': normal_data[col].std() if not normal_data.empty else 0,
            'Abnormal_mean': abnormal_data[col].mean() if not abnormal_data.empty else 0,
            'Abnormal_std': abnormal_data[col].std() if not abnormal_data.empty else 0,
            't_statistic': t_stat,
            'p_value': p_value
        })
    
    results_df = pd.DataFrame(results)
    save_path = os.path.join(output_dir, 'statistical_comparison.csv')
    results_df.to_csv(save_path, index=False)
    print(f"Statistics saved to {save_path}")
    
    return results_df

def plot_violin(df, dataset_name, output_dir):
    """Violin plot 생성"""
    feature_cols = ['L_mean', 'a_mean', 'b_mean', 'Area', 'Circularity', 'Aspect_ratio']
    
    if df.empty:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(feature_cols):
        sns.violinplot(data=df, x='label', y=col, ax=axes[idx], palette=['#0055A4', '#D84315'])
        axes[idx].set_xlabel('', fontsize=28)
        axes[idx].set_ylabel(col, fontsize=28)
        axes[idx].tick_params(labelsize=24)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_violin.png'), dpi=300)
    plt.close()

def analyze_features(data_root, dataset_name, output_dir, max_samples=500):
    """메인 분석 함수"""
    os.makedirs(output_dir, exist_ok=True)
    
    normal_path = os.path.join(data_root, 'normal')
    abnormal_path = os.path.join(data_root, 'abnormal')
    
    print(f"\n{'='*50}")
    print(f"Processing {dataset_name}")
    print(f"Path: {data_root}")
    print(f"{'='*50}")
    
    normal_features = load_and_extract_features(normal_path, 'normal', max_samples)
    abnormal_features = load_and_extract_features(abnormal_path, 'abnormal', max_samples)
    
    if not normal_features and not abnormal_features:
        print("No images found in both directories.")
        return

    all_features = normal_features + abnormal_features
    df = pd.DataFrame(all_features)
    
    print(f"Total samples: {len(df)} (normal: {len(normal_features)}, abnormal: {len(abnormal_features)})")
    
    stats_results = statistical_comparison(df, output_dir)
    print("\nStatistical comparison:")
    print(stats_results[['Feature', 'p_value']]) # 간단히 p-value만 출력 확인
    
    plot_violin(df, dataset_name, output_dir)
    
    print(f"Results saved to {output_dir}")

def get_args():
    parser = argparse.ArgumentParser(description='Statistical Analysis for Image Dataset')
    parser.add_argument('--class_name', type=str, default='abnormal', help='Target Class Name (folder name in ./data)')
    parser.add_argument('--max_samples', type=int, default=500, help='Max samples per class')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    base_path = './data' 
    output_base = './results/analysis'
    
    class_name = args.class_name
    
    data_root = os.path.join(base_path, class_name)
    output_dir = os.path.join(output_base, class_name)
    
    # 분석 실행
    if os.path.exists(data_root):
        analyze_features(data_root, class_name, output_dir, max_samples=args.max_samples)
        print("\nAnalysis completed!")
    else:
        print(f"Error: Data directory not found at {data_root}")
