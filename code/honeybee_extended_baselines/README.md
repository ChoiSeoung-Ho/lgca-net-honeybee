# 확장 baseline 실행 패키지 (Conformer / Mobile-Former / LSNet-T / MambaVision)

원고에서 빠져 있는 네 개 baseline을 **감사 프로토콜 그대로** 학습·평가하고, 그 결과를
모든 표·그림에 자동 반영하기 위한 코드입니다. GPU가 있는 서버에서 실행하십시오.

## 왜 이 패키지가 필요한가

3차 코드 패키지의 `common/models.py`에는 이 네 모델이 `_external_factory(...)`
자리표시자로만 등록돼 있어 호출하면 `ImportError`가 납니다. 여기서는

* 네 아키텍처의 **실제 어댑터**(다중 소스 시도, 헤드 수술, 사전학습 가중치 검증),
* 학습 없이 먼저 확인하는 **사전 점검 도구**,
* 모델 목록을 **하드코딩하지 않는** 분석·표·그림 생성기

를 제공합니다. 마지막 항목이 핵심입니다 — 기존 분석 스크립트는 9개 모델을 코드에
박아두고 있어서, baseline을 추가하면 표와 그림을 손으로 고쳐야 했습니다. 여기 있는
스크립트는 `results/oof_<task>_<model>_<tag>.csv`를 훑어 모델을 발견하므로,
13개 모델로 다시 돌리면 모든 산출물이 저절로 13행이 됩니다.

## 파일

```
models/external/adapters.py       네 아키텍처 어댑터 (핵심)
models/external/README.md         설치 방법과 주의사항 (영문)
patch_models_registry.py          common/models.py에 어댑터 연결 (idempotent, --revert 지원)
verify_external_models.py         학습 없이 빌드·파라미터·forward 점검 → model_provenance.json
run_extended_baselines.sh         1) 점검 2) 학습 3) 평가 4) 표·그림 재생성
analysis/model_registry.py        모델 발견·표시명·순서·파라미터 수
analysis/common_io.py             경로·CLI·SMD·Holm 공용
analysis/matched_subsets.py       매칭 서브셋 + 부트스트랩 CI + DeLong/McNemar
analysis/confound_reliance.py     confound-reliance 분석 (원고 5.5절)
analysis/make_manuscript_tables.py  LaTeX 표 조각 9종
analysis/make_manuscript_figures.py 벡터 그림 4종
```

## 실행

```bash
# 0) 준비 — models/external/README.md 의 설치 절차를 먼저 수행
cd /path/to/code
python /path/to/이패키지/patch_models_registry.py --code-root .

# 1) 학습 전에 반드시 확인 (수 초)
python /path/to/이패키지/verify_external_models.py --code-root . \
       --models conformer mobileformer lsnet_t mambavision

# 2) 전체 실행
CODE_ROOT=/path/to/code DATA_ROOT=/path/to/data GPU=0 \
  bash /path/to/이패키지/run_extended_baselines.sh
```

일부만 설치됐다면 `NEW_MODELS="conformer lsnet_t"` 처럼 지정해 그것만 돌릴 수 있습니다.
단계만 다시 돌리려면 `STAGES="4"`.

## 설계 원칙 — 조용한 대체 금지

* 설치되지 않은 백본은 **명확한 설치 명령과 함께 예외**를 던집니다. 비슷한 다른
  아키텍처로 대체하지 않습니다.
* `pretrained=True`를 지킬 수 없으면 예외입니다. `--allow-random-init`으로 강제할 수는
  있지만, 그 경우 provenance에 `pretrained_loaded: false`가 기록되고 표의 사전학습
  칸에 `none (random)`이 찍힙니다.
* **Mobile-Former는 공식 코드도 ImageNet 가중치도 공개된 적이 없습니다.** 재구현
  구현체에 체크포인트가 없으면 무작위 초기화 상태이며, 그것은 사전학습 baseline이
  아닙니다. 원고에 그렇게 쓰면 안 됩니다.
* 파라미터 수는 **측정값**만 씁니다. 점검을 돌리지 않은 모델은 표에 `--`로 나옵니다.
* Conformer는 `[conv_logits, trans_logits]` 두 헤드를 반환합니다. 공식 평가대로 합산
  합니다(`MultiHeadSum`). 한쪽만 쓰면 발표된 모델과 다른 모델이 됩니다.
* 정규화: 이 파이프라인은 모든 모델에 ImageNet mean/std 없이 `[0,1]`을 넣습니다. 새
  백본만 정규화하면 기존 행과 비교 불가가 되므로 동일하게 처리합니다.

## 실행 후 손으로 고쳐야 하는 것

표·그림은 자동 갱신되지만, 본문에 **숫자로 적힌 모델 개수**는 자동이 아닙니다.
다음을 grep 해서 갱신하십시오.

```
"six ImageNet-1K"   "six pretrained baselines"   "nine networks"   "nine models"
"the three plain pretrained CNNs"                "24 comparisons"  "of the 24"
```

Holm 보정은 표 안의 비교 개수에 걸리므로, 모델이 늘면 **유의 개수가 달라집니다**
(비교 수가 24 → 36으로 늘면 보정이 강해집니다). 5.3절·5.5절·6장의 개수 서술과
Table 13(pairwise summary)을 다시 확인해야 합니다.

## 검증 상태

이 패키지의 분석 스크립트는 현재 9개 모델 결과에 대해 **원고 수치를 정확히 재현**하는
것을 확인했습니다(매칭 쌍 109/174, L\* SMD −2.484→−0.418 및 −1.347→−0.045,
Holm 유의 2/24 및 8/24, confound 기준선 0.945/0.845/0.853/0.583). 어댑터 자체는
torch·GPU가 없는 환경에서 작성돼 **문법 검사만** 거쳤습니다 — 서버에서 먼저
`verify_external_models.py`로 확인하십시오.

## 부수적으로 발견한 결함 (원고에 반영됨)

발표된 model-free 기준선은 **비그룹화 stratified fold**(테스트 334/333/333)에서,
딥러닝 모델은 **hive-grouped fold**(429/342/229)에서 나왔습니다. 원고는 "딥러닝
모델과 정확히 같은 fold"라고 서술하고 있었으므로 사실이 아니었습니다.
`matched_subsets.py`는 두 경우 모두 감사 fold를 사용해 다시 계산합니다.
결론은 바뀌지 않지만 수치가 조금 달라집니다(Chalkbrood 0.918/0.953/0.956 →
0.909/0.945/0.947, Foulbrood 0.839/0.833/0.940 → 0.841/0.845/0.900).
