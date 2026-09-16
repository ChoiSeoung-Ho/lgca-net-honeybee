# Third-Party Licenses

이 폴더(`PosNeg_auc_util`)의 Pos/Neg perturbation 평가 코드는 아래 공개 자료를 참고해 구성되었습니다.

## Transformer Interpretability Beyond Attention Visualization (CVPR 2021)

- Repository: https://github.com/hila-chefer/Transformer-Explainability
- License: MIT License
- Usage:
  - Positive/Negative perturbation 프로토콜(10%~90% 픽셀 제거, Top-1 기반 곡선, AUC 계산) 참고
  - 현재 프로젝트 데이터/모델 구조(CV fold)에 맞게 코드 구조를 재작성

### MIT License (upstream)

Copyright (c) 2021 Hila Chefer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
