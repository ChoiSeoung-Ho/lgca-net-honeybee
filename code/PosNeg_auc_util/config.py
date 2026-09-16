from dataclasses import dataclass

@dataclass(frozen=True)
class AUCConfig:
    # 똑같은 실험 결과를 다시 얻기 위한 시드
    seed: int = 42
    class_names: tuple = ("you_chalk_brood","you_foulbrood")
    # 학습된 모델(.pt)들이 저장된 기본 폴더
    base_model_path: str = "./model_save"
    # 결과 CSV 저장 폴더
    out_dir: str = "./results_POS_NEG"

    # 입력 이미지를 모델에 넣기 전 맞출 크기
    img_size: int = 256
    # 몇 개의 fold를 평가할지
    folds: tuple = (0, 1, 2)
    n_splits: int = 3
