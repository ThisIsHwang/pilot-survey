# 3일 H100 행동–문서 크레딧 실험 가이드

## 이 실험이 묻는 질문

검색 에이전트는 다음 순서로 작동합니다.

```text
검색어를 만듦 → 검색기가 문서를 돌려줌 → 문서를 읽고 답함
```

여기서 **크레딧(credit)**은 최종 결과의 공을 중간 단계에 나눠 주는 점수입니다.
이 실험은 다음 두 점수가 같은 검색어를 좋다고 판단하는지 검사합니다.

1. **행동 크레딧**: 같은 상태에서 다른 검색어를 실행했을 때 최종 결과가 얼마나 달라지는가?
2. **문서 크레딧**: 검색어는 그대로 두고 문서 하나만 다른 문서로 교체했을 때 최종 결과가 얼마나 달라지는가?

문서 점수가 행동 점수를 대신할 수 없다면, 문서 점수를 검색어 토큰의 보상으로 전달하는 학습법은 별도 검증이 필요합니다.

## 가장 중요한 설계 변경

기존의 문서 제거 실험은 문서 수가 3개에서 2개로 줄어드는 문제가 있었습니다. 이번 구현은 **고정 크기 문서 교체(fixed-cardinality swap)**를 주 실험으로 씁니다.

```text
원래 관찰: 문서 A, B, C
교체 관찰: 문서 X, B, C
```

- 문서 수는 항상 3개입니다.
- 문서 위치도 유지합니다.
- X는 검색 순위 4~10위 중 길이가 가장 비슷한 문서로, 결과를 보기 전에 정합니다.
- 정답, 정답 근거 제목, 이후 보상은 X 선택에 사용하지 않습니다.
- 기존의 문서 제거는 작은 민감도 분석에서만 같이 실행합니다.

## 자동 하드웨어 모드

`query_credit/run_weekend_h100.sh`는 보이는 GPU 수를 자동으로 확인합니다.

### H100 8장 노드

- 프로필: `node8`
- GPU 0~6: Qwen2.5-7B vLLM 서버
- GPU 7: E5 검색기
- 검색기: BM25와 E5
- 데이터셋: 2WikiMultiHopQA, HotpotQA, MuSiQue
- 목표 상태: 180개, 데이터셋×검색기 조합당 30개
- BM25와 E5에 공통으로 존재하는 같은 질문을 짝지어 선택
- 조합당 최소 24개가 없으면 본실험을 시작하지 않음

### GPU 1장 환경

- 프로필: `single`
- 검색기: BM25만 사용
- 총 상태: 45개, 데이터셋당 15개
- 이 모드는 코드·효과 방향 점검용이며 최종 논문 표본으로는 부족합니다.

## 8장 노드의 주 수집량

| 항목 | 설정 |
|---|---:|
| 상태 | 180 |
| 상태당 검색어 | 6 |
| 연속 실행 시드 | 4 |
| 주 개입 | 원본 1회 + 문서 교체 3회 |
| 주 재실행 궤적 | 17,280 |
| 문서 제거 민감도 | 2,160 |
| 전체 예상 궤적 | 약 19,440 |

**연속 실행 시드(continuation seed)**는 검색어 이후의 모델 생성에 들어가는 무작위 번호입니다. 원본과 교체 조건에 같은 번호를 써서 생성 운의 차이를 줄입니다.

## 72시간 배분

| 단계 | 최대 시간 | 목적 |
|---|---:|---|
| 반사실적 수집 | 44시간 | 행동 결과와 문서 교체 결과 저장 |
| 그래디언트 감사 | 8시간 | 학습 방향이 얼마나 달라지는지 측정 |
| 일치 LoRA 학습 | 16시간 | 같은 학습량에서 보조점수 효과 비교 |
| 예비 시간 | 4시간 | 보고서 생성·재시작 여유 |

**그래디언트(gradient)**는 모델 파라미터를 어느 방향으로 바꿀지 정하는 학습 화살표입니다.
**LoRA**는 전체 모델 대신 작은 추가 파라미터만 학습해 비용을 줄이는 방법입니다.

## 실행

먼저 이전 causal-query 실험의 상태 파일 위치를 지정합니다. 기본 경로에 파일이 있으면 생략할 수 있습니다.

```bash
export QUERY_CREDIT_INPUTS='/absolute/path/to/causal_query_audit/results/full/states/*/*.json'
```

그다음 짧은 코드·서비스 점검을 합니다.

```bash
PROFILE=smoke bash query_credit/run_weekend_h100.sh
```

점검이 통과하면 주말 실험을 실행합니다.

```bash
PROFILE=auto bash query_credit/run_weekend_h100.sh
```

터미널 연결과 무관하게 노드 작업 관리자가 프로세스를 유지하는 환경이라면 다음처럼 로그를 남길 수 있습니다.

```bash
PROFILE=auto bash query_credit/run_weekend_h100.sh \
  > logs/query_credit_weekend_driver.log 2>&1
```

이미 끝난 수집을 다시 하지 않고 뒤 단계만 재개할 수 있습니다.

```bash
SKIP_COLLECTION=1 PROFILE=node8 bash query_credit/run_weekend_h100.sh
```

특정 단계를 생략할 수도 있습니다.

```bash
SKIP_GRADIENT=1 PROFILE=node8 bash query_credit/run_weekend_h100.sh
SKIP_MICRO=1 PROFILE=node8 bash query_credit/run_weekend_h100.sh
```

수집 결과가 사전 기준을 통과하지 못하면 비싼 학습 단계는 자동으로 중단됩니다. 디버깅 목적으로만 강제 진행하려면 다음을 사용합니다.

```bash
FORCE_CONTINUE=1 PROFILE=node8 bash query_credit/run_weekend_h100.sh
```

강제 진행 결과는 사전등록형 주장 근거로 쓰면 안 됩니다.

## 재시작과 캐시

각 상태는 완성되는 즉시 독립 JSON 파일로 저장됩니다.

```text
work/query_credit_weekend/<profile>/cache/<backend>/<state_id>.json
```

중간에 작업이 종료되어도 같은 명령을 다시 실행하면 완성된 상태는 건너뜁니다. 캐시 서명에는 상태 내용, 모델 경로, 수집 설정이 들어가므로 설정이 달라진 결과를 조용히 재사용하지 않습니다.

## 주요 출력

### 수집 원자료

```text
work/query_credit_weekend/<profile>/data/raw_replays.jsonl
work/query_credit_weekend/<profile>/data/candidate_credits.jsonl
work/query_credit_weekend/<profile>/data/state_prefixes.jsonl
work/query_credit_weekend/<profile>/data/state_manifest.jsonl
```

`raw_replays.jsonl`은 평균 전의 시드별 원자료입니다. 논문 검증과 오류 추적에서 가장 중요한 파일입니다.

### 행동–문서 감사

```text
work/query_credit_weekend/<profile>/reports/audit/AUDIT_REPORT_KO.md
work/query_credit_weekend/<profile>/reports/audit/audit_state_metrics.csv
work/query_credit_weekend/<profile>/reports/audit/decision.json
```

주 지표는 다음과 같습니다.

- `action_self_pairwise`: 행동 점수를 시드 절반씩 계산했을 때 검색어 우열 판단이 스스로 일치하는 비율
- `document_action_pairwise`: 서로 다른 시드 절반에서 계산한 문서 점수와 행동 점수가 같은 검색어를 선호하는 비율
- `reliability_gap`: 위 두 값의 차이. 문서와 행동이 같은 생성 잡음을 공유해 생기는 가짜 일치를 피하도록 교차평가합니다.
- `normalized_regret`: 문서 점수로 고른 검색어가 실제 최선 검색어보다 잃는 보상 비율

### 그래디언트 감사

```text
work/query_credit_weekend/<profile>/reports/gradient/GRADIENT_REPORT_KO.md
work/query_credit_weekend/<profile>/reports/gradient/gradient_state_results.csv
```

### 일치 학습

```text
work/query_credit_weekend/<profile>/reports/micro/MICRO_REPORT_KO.md
work/query_credit_weekend/<profile>/reports/micro/micro_effects.csv
```

비교 조건은 다음과 같습니다.

1. `outcome-only`: 최종 결과 점수만 사용
2. `outcome-plus-swap`: 최종 결과 + 부호를 유지한 문서 교체 점수
3. `outcome-plus-positive`: 최종 결과 + 양수 문서 점수만 합친 값
4. `outcome-plus-shuffled`: 문서 점수의 분포는 유지하되 검색어와의 연결만 섞은 대조군

모든 조건은 같은 초기화, 예시, 미니배치 순서, 업데이트 횟수를 사용합니다. 보조점수의 크기도 결과 점수와 같은 RMS로 맞춥니다. **RMS**는 점수의 전형적인 크기를 나타내는 값입니다.

### 최종 주장 판정

```text
work/query_credit_weekend/<profile>/reports/WEEKEND_DECISION_KO.md
```

결과에 따라 다음 주장을 별도로 허용하거나 차단합니다.

- 문서 점수가 행동 점수의 신뢰할 만한 대체값이 아니다.
- 문서 점수를 행동에 전달하면 학습 방향이 달라진다.
- 문서 보조점수가 일치 학습 성능을 악화한다.
- 인과적으로 부정확한 문서 점수도 학습 힌트로는 도움이 된다.

## 이 3일 실험이 하지 않는 것

이 파이프라인의 학습 실험은 이미 실행된 검색어 후보를 대상으로 한 **오프라인 미세 학습**입니다. 업데이트된 모델이 새 검색어를 만들고 다시 검색하는 완전한 폐루프 강화학습은 포함하지 않습니다.

따라서 주말 결과만으로 다음을 주장하면 안 됩니다.

- 모든 온라인 검색 에이전트 학습에서 문서 보상이 성능을 떨어뜨린다.
- 문서 크레딧은 쓸모없다.
- 모든 모델과 모든 도구 환경에 같은 결론이 적용된다.

주말 실험의 목적은 논문의 핵심 현상과 학습 메커니즘이 충분히 강한지 판정하는 것입니다. 완전한 폐루프 결과는 그 뒤의 최종 논문 보강 실험입니다.
