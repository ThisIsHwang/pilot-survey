# 5일 H100 Qwen3.5 비추론 모드 행동–문서 크레딧 실험 가이드

## 이번 버전에서 바뀐 것

모델을 `Qwen/Qwen2.5-7B-Instruct`에서 **`Qwen/Qwen3.5-9B`**로 바꿨습니다. Qwen3.5의 **thinking(추론 모드)**은 답을 내기 전에 별도 추론 내용을 생성하는 모드입니다. 이 실험에서는 추론 모드를 완전히 끕니다.

Qwen3.5는 `/nothink` 문자열로 모드를 바꾸는 방식을 지원하지 않으므로, 다음 네 곳에서 모두 `enable_thinking=false`를 강제합니다.

1. vLLM 서버의 기본 채팅 템플릿
2. 모든 OpenAI 호환 생성 요청
3. 그래디언트·LoRA 학습용 로컬 토크나이저
4. 시작 전 동시 요청 재현성 검사

응답에 `<think>` 태그나 별도 reasoning 필드가 한 번이라도 나타나면 **추론 모드 누출(thinking leakage)**로 간주합니다. 누출은 실행별 원장에 즉시 추가되며, 한 건이라도 있으면 분석을 중단하여 추론 모드와 비추론 모드 결과가 섞이지 않게 합니다.

## 이 실험이 묻는 질문

검색 에이전트는 다음 순서로 작동합니다.

```text
검색어를 만듦 → 검색기가 문서를 돌려줌 → 문서를 읽고 답함
```

여기서 **크레딧(credit)**은 최종 결과의 공을 중간 단계에 나눠 주는 점수입니다. 이 실험은 다음 두 점수가 같은 검색어를 좋다고 판단하는지 검사합니다.

1. **행동 크레딧**: 같은 상태에서 다른 검색어를 실행했을 때 최종 결과가 얼마나 달라지는가?
2. **문서 크레딧**: 검색어는 그대로 두고 문서 하나만 다른 문서로 교체했을 때 최종 결과가 얼마나 달라지는가?

문서 점수가 행동 점수를 대신할 수 없다면, 문서 점수를 검색어 토큰의 보상으로 전달하는 학습법은 별도 검증이 필요합니다.

## 문서 개입

주 실험은 **고정 크기 문서 교체(fixed-cardinality swap)**입니다. 문서 수와 위치를 유지하면서 내용만 바꿉니다.

```text
원래 관찰: 문서 A, B, C
교체 관찰: 문서 X, B, C
```

- 문서 수는 항상 3개입니다.
- 문서 위치를 유지합니다.
- X는 검색 순위 4~10위 중 원래 문서와 길이가 가장 비슷한 문서입니다.
- 정답, 정답 근거 제목, 이후 보상은 X 선택에 사용하지 않습니다.
- 기존 문서 제거는 작은 민감도 분석에서만 실행합니다.

## Qwen3.5 서빙 방식

Qwen3.5의 **GDN(Gated DeltaNet: 일부 토큰 정보를 순환 상태로 처리하는 계층)**은 현재 사용하는 vLLM 설정에서 batch-invariant 모드를 사용하지 않습니다.

대신 각 데이터 병렬 모델이 한 번에 요청 하나만 처리하게 `--max-num-seqs 1`을 사용합니다.

```text
GPU 0~6: Qwen3.5-9B 사본 7개, 각 사본은 동시에 요청 1개 처리
GPU 7: E5 검색기
```

서버가 뜨면 같은 시드의 동일 요청을 여러 GPU에 동시에 보냅니다. 모든 응답이 글자 단위로 같고 추론 모드가 없어야 본실험이 시작됩니다.

## 환경 준비

검색기와 데이터 처리용 기존 환경을 준비합니다.

```bash
bash scripts/bootstrap.sh
bash scripts/bootstrap_vllm.sh
```

Qwen3.5 전용 로컬 학습 환경을 별도로 준비합니다.

```bash
bash scripts/bootstrap_qwen35.sh
```

별도 환경을 쓰는 이유는 Qwen3.5가 새 Transformers 계열을 요구하지만, 기존 검색기 환경의 일부 패키지는 이전 Transformers 계열에 고정되어 있기 때문입니다.

## Qwen2.5 결과 재사용 규칙

**캐시(cache)**는 이미 계산한 결과를 저장한 파일입니다. Qwen3.5 결과는 다음 전용 경로에 저장합니다.

```text
work/query_credit_weekend_qwen35
```

이전 Qwen2.5 결과 경로와 물리적으로 분리되므로 라벨, 정보이득, 그래디언트 또는 학습 결과가 섞이지 않습니다.

Qwen3.5에서 반드시 다시 계산하는 항목은 다음입니다.

- Qwen3.5가 직접 생성한 검색어 후보
- 원본·문서 교체·문서 제거 후속 실행
- 행동–문서 크레딧 감사
- 정보이득 점수
- LoRA 그래디언트
- 12개 시드 미세학습과 최종 보고서

Wiki-18 말뭉치와 BM25/E5 인덱스처럼 모델 출력과 무관한 공통 자산만 무결성 검사를 통과한 뒤 재사용합니다. 자세한 범위는 `query_credit/QWEN35_RERUN_MATRIX_KO.md`에 정리되어 있습니다.

## 자동 하드웨어 모드

### H100 8장 노드

- 프로필: `node8`
- 모델: Qwen3.5-9B, 추론 모드 끔
- GPU 0~6: 텍스트 전용 vLLM 데이터 병렬 사본 7개
- GPU 7: E5 검색기
- 검색기: BM25와 E5
- 데이터셋: 2WikiMultiHopQA, HotpotQA, MuSiQue
- 목표 상태: 180개, 데이터셋×검색기 조합당 30개
- 조합당 최소 24개가 없으면 본실험을 시작하지 않음

### GPU 1장 환경

- 프로필: `single`
- Qwen3.5-9B 한 사본과 BM25 사용
- 총 상태: 45개
- 코드와 효과 방향 점검용이며 최종 논문 표본으로는 부족합니다.

## 8장 노드의 주 수집량

| 항목 | 설정 |
|---|---:|
| 상태 | 180 |
| 상태당 검색어 | 최대 6 |
| 연속 실행 시드 | 6 |
| 주 개입 | 원본 1회 + 문서 교체 3회 |
| 주 재실행 궤적 | 약 25,920 |
| 문서 제거 민감도 | 약 3,240 |
| 전체 예상 궤적 | 약 29,160 |

**연속 실행 시드(continuation seed)**는 검색어 이후 모델 생성에 들어가는 무작위 번호입니다. 원본과 교체 조건에 같은 번호를 사용합니다.

## 120시간 배분

| 단계 | 최대 시간 | 목적 |
|---|---:|---|
| 반사실적 수집 | 80시간 | 행동 결과와 문서 교체 결과 저장 |
| 정보이득 기준선 | 8시간 | 실제 문서와 길이 맞춤 무작위 문서에서 정답 확률 비교 |
| 그래디언트 감사 | 10시간 | 학습 방향이 얼마나 달라지는지 측정 |
| 일치 LoRA 학습 | 18시간 | 같은 학습량에서 보조점수 효과 비교 |
| 예비 시간 | 4시간 | 보고서 생성과 재시작 여유 |
| **합계** | **120시간** | **5일** |

**그래디언트(gradient)**는 모델을 어느 방향으로 학습시킬지 나타내는 화살표입니다. **LoRA**는 전체 모델 대신 작은 추가 파라미터만 학습해 비용을 줄이는 방법입니다.

## 실행

이전 causal-query 실험의 상태 파일 위치를 지정합니다.

```bash
export QUERY_CREDIT_INPUTS='/absolute/path/to/causal_query_audit/results/full/states/*/*.json'
```

먼저 아주 작은 전체 경로 점검을 합니다. 작은 표본은 과학적 게이트를 통과하기 어려우므로 디버깅에서만 `FORCE_CONTINUE=1`을 사용합니다.

```bash
FORCE_CONTINUE=1 PROFILE=smoke \
  bash query_credit/run_qwen35_five_day.sh
```

추론 모드가 꺼졌다는 계약 파일을 확인합니다.

```bash
cat work/query_credit_weekend_qwen35/runtime/qwen35_runtime_contract.json
```

다음 항목이 있어야 합니다.

```text
enable_thinking: false
language_model_only: true
batch_invariant: false
max_num_seqs: 1
unique_probe_outputs: 1
```

본실험을 실행합니다.

```bash
mkdir -p logs
PROFILE=auto bash query_credit/run_qwen35_five_day.sh \
  > logs/query_credit_qwen35_driver.log 2>&1
```

노드 스케줄러의 벽시계 제한도 5일에 맞춰 설정해야 합니다. 스크립트 내부 단계 예산 합계는 120시간입니다.

## 재시작

같은 명령을 다시 실행하면 완료된 Qwen3.5 상태를 이어서 처리합니다.

```bash
PROFILE=auto bash query_credit/run_qwen35_five_day.sh
```

이미 Qwen3.5 수집을 끝냈고 뒤 단계만 다시 실행할 때는 다음을 사용합니다.

```bash
SKIP_COLLECTION=1 PROFILE=node8 \
  bash query_credit/run_qwen35_five_day.sh
```

수집 결과가 사전 기준을 통과하지 못하면 비싼 학습 단계는 자동으로 생략됩니다. `FORCE_CONTINUE=1` 결과는 논문 주장 근거로 사용하면 안 됩니다.

## 주요 출력

### 추론 모드와 런타임 계약

```text
work/query_credit_weekend_qwen35/runtime/qwen35_runtime_contract.json
work/query_credit_weekend_qwen35/runtime/run_manifest.txt
work/query_credit_weekend_qwen35/runtime/thinking_leaks.jsonl
```

### 평균 전 원자료

```text
work/query_credit_weekend_qwen35/<profile>/data/raw_replays.jsonl
work/query_credit_weekend_qwen35/<profile>/data/candidate_credits.jsonl
work/query_credit_weekend_qwen35/<profile>/data/collection_errors.jsonl
```

`thinking_leaks.jsonl`에 한 줄이라도 생기면 파이프라인이 중단됩니다. 같은 설정과 코드로 재시작해도 이 원장은 유지됩니다.

### 최종 보고서

```text
work/query_credit_weekend_qwen35/<profile>/reports/audit/AUDIT_REPORT_KO.md
work/query_credit_weekend_qwen35/<profile>/reports/ig/IG_REPORT_KO.md
work/query_credit_weekend_qwen35/<profile>/reports/gradient/GRADIENT_REPORT_KO.md
work/query_credit_weekend_qwen35/<profile>/reports/micro/MICRO_REPORT_KO.md
work/query_credit_weekend_qwen35/<profile>/reports/WEEKEND_DECISION_KO.md
```

## 비교하는 학습 조건

1. `outcome-only`: 최종 결과 점수만 사용
2. `outcome-plus-swap`: 최종 결과 + 부호를 유지한 문서 교체 점수
3. `outcome-plus-positive`: 최종 결과 + 양수 문서 점수만 합친 값
4. `outcome-plus-ig`: 실제 문서가 무작위 문서보다 정답 확률을 얼마나 높였는지 사용
5. `outcome-plus-shuffled`: 점수 분포는 유지하되 검색어와의 연결만 섞은 대조군

모든 조건은 같은 초기화, 예시, 미니배치 순서와 업데이트 횟수를 사용합니다. Qwen3.5의 전체 어텐션 계층뿐 아니라 GDN 계층의 선형 투영에도 LoRA를 적용합니다.

## 이 실험이 하지 않는 것

이번 학습은 이미 수집한 검색어 후보를 대상으로 한 **오프라인 미세 학습**입니다. 업데이트된 모델이 새 검색어를 만들고 다시 검색하는 완전한 폐루프 강화학습은 포함하지 않습니다.

기존 상태 파일은 고정 평가 문맥으로 사용할 수 있지만, 주 분석의 검색 행동은 모두 Qwen3.5가 새로 생성합니다. 따라서 이번 실험은 고정 상태에서의 Qwen3.5 행동 비교이며, 질문 처음부터 끝까지 Qwen3.5가 직접 방문한 상태만 사용하는 완전한 온폴리시 실험은 아닙니다.

따라서 다음은 주장하지 않습니다.

- 문서 점수는 항상 쓸모없다.
- 모든 온라인 검색 에이전트에서 문서 보상이 성능을 떨어뜨린다.
- Qwen3.5 결과가 모든 모델과 도구 환경에 그대로 적용된다.
