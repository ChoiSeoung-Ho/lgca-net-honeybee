# PosNeg AUC Utility

이 폴더는 논문식 Positive/Negative Perturbation AUC 평가 코드를 담고 있습니다.

## What It Does

- 설명맵(relevance map)을 픽셀 중요도 순으로 정렬
- POS: 중요한 픽셀부터 제거
- NEG: 덜 중요한 픽셀부터 제거
- 제거율 10%~90% 구간에서 Top-1 hit 곡선 계산
- `np.trapz`로 PosAUC / NegAUC 계산

## Citation (BibTeX)

본 폴더의 평가 프로토콜은 아래 논문/공식 구현을 참고했습니다.

```bibtex
@InProceedings{Chefer_2021_CVPR,
  author    = {Chefer, Hila and Gur, Shir and Wolf, Lior},
  title     = {Transformer Interpretability Beyond Attention Visualization},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2021},
  pages     = {782--791}
}
```

- Official code repository: https://github.com/hila-chefer/Transformer-Explainability

## License Note

- Third-party license details are in:
  - `PosNeg_auc_util/THIRD_PARTY_LICENSES.md`