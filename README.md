# llm-eval-gate

![CI](https://github.com/zerda217/llm-eval-gate/actions/workflows/ci.yml/badge.svg)

**LLM 기능을 위한 평가 게이트.** 평가 지표와 기준을 YAML로 선언하면, 프롬프트나 모델을 바꿨을 때 품질이 기준보다 낮아지거나 이전보다 나빠진 경우 0이 아닌 종료 코드로 배포를 막습니다.

데이터 파이프라인 품질 게이트인 [pipeguard](https://github.com/zerda217/pipeguard)의 구조(플러그인 레지스트리, severity, 종료 코드)를 LLM 출력 평가로 옮긴 프로젝트입니다.

## 시작한 이유

텔레그램으로 잡은 약속을 캘린더에 옮기는 일을 자동으로 하고 싶어서, 대화에서 약속을 찾아 구글 캘린더에 넣어주는 봇을 만들기로 했습니다. LLM에게 대화를 주고 일정을 뽑게 하면 금방 동작하는 것처럼 보이지만, 이런 실수가 생길 수 있습니다.

- "다음 주 수요일"을 이번 주 수요일로 계산한다
- 자정 넘어 온 답장을 기준으로 "모레"를 하루 늦게 잡는다
- 취소된 약속, 답장이 없던 제안까지 일정으로 넣는다
- "저녁 7시"를 07:00으로 적는다

캘린더에 틀린 일정이 들어가는 것은 일정을 하나 놓치는 것보다 해롭습니다. 그래서 봇보다 먼저, **프롬프트를 바꿀 때마다 이런 실수가 늘지 않았는지 자동으로 검사하는 게이트**를 만들었습니다.

## 실행 예시

LLM이 흔히 저지르는 실수를 일부러 넣은 출력을 게이트에 통과시키면 이렇게 막힙니다.

```
llm-eval-gate · telegram-appointments · 40 cases

  FAIL  json_valid       0.975  (min 0.980, baseline -)  39/40 parsed  <- below min 0.980
  PASS  schema_valid     0.975  (min 0.950, baseline -)  39/40 valid
  FAIL  event_precision  0.594  (min 0.900, baseline -)  19 correct / 32 predicted  <- below min 0.900
  FAIL  no_false_plans   0.667  (min 0.900, baseline -)  8/12 correctly empty  <- below min 0.900
  WARN  event_recall     0.655  (min 0.800, baseline -)  19 found / 29 expected  <- below min 0.800

Cases with problems (15):
  - s002 [absolute_date]: unexpected (2026-11-19, 01:00); missed (2026-11-19, 13:00)
  - s012 [relative_day, cross_midnight]: unexpected (2026-11-09, 20:00); missed (2026-11-08, 20:00)
  - s026 [cancelled]: unexpected (2026-11-01, 10:30); expected no items, got 1
  ...

By tag (cases with problems / total):
  changed          4/5
  cross_midnight   1/1
  tentative        3/6
  ...

Result: FAILED (3 error(s), 1 warning(s)) -> exit 1
```

어떤 지표가 왜 떨어졌는지, 어떤 케이스에서 무엇을 틀렸는지, 어떤 유형(태그)에 실패가 몰려 있는지를 한 번에 보여줍니다.

## 빠른 시작

```bash
pip install git+https://github.com/zerda217/llm-eval-gate

# 모델 호출 없이: 미리 만든 출력으로 게이트 동작 확인
llm-eval-gate run examples/schedule/eval.yaml --outputs examples/schedule/fixtures/broken_outputs.jsonl --no-baseline

# 실제 Gemini로 평가 (무료 키: aistudio.google.com)
export GEMINI_API_KEY=...
llm-eval-gate run examples/schedule/eval.yaml
```

### 휴대폰에서 실제 모델 평가 돌리기

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zerda217/llm-eval-gate/blob/main/notebooks/run_eval.ipynb)

API 키를 저장소에 두지 않는 방식입니다.

1. 위 배지로 노트북을 열고, Colab 보안 비밀(🔑)에 `GEMINI_API_KEY`를 등록합니다. 키는 본인 Google 계정에만 저장됩니다
2. 셀을 순서대로 실행하면 실제 Gemini로 평가하고, 응답 캐시를 `eval_results.zip`으로 내려받습니다. 내려받기 전에 키가 들어 있지 않은지 확인합니다
3. zip 안의 `examples/schedule/cache/`를 저장소에 커밋하면, 이후 CI는 키 없이 이 응답을 재생해 게이트를 돌립니다

평가는 키가 있는 곳에서 한 번만 하고, 검증은 키 없이 몇 번이든 재현합니다. 저장소에 GitHub Secret으로 키를 등록해도 된다면 Actions의 **Record eval** 워크플로로 같은 일을 GitHub 안에서 할 수도 있습니다.

## 설정

```yaml
task:
  dataset: cases.jsonl         # {"id", "input", "expected", "tags"} 한 줄에 한 케이스
  system: system.txt
  prompt: prompt.txt           # {{conversation}} 처럼 input 필드를 넣는 템플릿
  output_schema: schema.json   # JSON Schema

provider:
  type: gemini                 # gemini | file (미리 만든 출력 채점)
  model: gemini-3.5-flash-lite
  temperature: 0
  rpm: 10                      # 무료 티어 분당 한도에 맞춘 호출 간격
  cache_dir: cache

metrics:
  - name: event_precision
    type: set_precision
    path: events
    keys: [date, start_time]   # 이 필드가 같으면 같은 일정으로 본다
    min: 0.90
    severity: error            # error(기본) | warn

gate:
  baseline: baseline.json
  max_regression: 0.03         # 기준선보다 3%p 넘게 떨어지면 차단
```

### 지표

| type | 측정 내용 |
|---|---|
| `json_valid_rate` | JSON으로 파싱되는 출력 비율 |
| `schema_valid_rate` | 출력 스키마를 만족하는 비율 |
| `set_precision` | 출력한 항목 중 정답인 비율 (`keys`로 항목을 식별, 전체 케이스 합산) |
| `set_recall` | 정답 항목 중 출력에 포함된 비율 |
| `empty_when_expected_rate` | 정답이 "없음"인 케이스에서 출력도 비어 있는 비율 |
| `exact_match_rate` | 특정 필드가 정답과 일치하는 비율 (분류 태스크용) |

모든 지표는 0~1 사이이고 높을수록 좋습니다. 그래서 게이트 규칙이 "기준 이상인가, 기준선에서 얼마나 떨어졌나" 두 가지로 통일됩니다.

### CLI와 종료 코드

```
llm-eval-gate run CONFIG [--mode cached|replay|live] [--outputs FILE] [--json-out FILE] [--md-out FILE]
                         [--fail-on-warn] [--update-baseline] [--no-baseline]
llm-eval-gate validate CONFIG
llm-eval-gate list-metrics
```

| 코드 | 의미 |
|---|---|
| 0 | 통과 |
| 1 | error 지표가 기준 미달이거나 기준선 대비 하락 |
| 2 | 설정 오류 (모델 호출 전에 걸러짐) |
| 3 | 데이터셋·모델 호출 실패, replay 모드에서 캐시 없음 |

종료 코드 체계는 pipeguard와 같습니다. 같은 파이프라인 안에서 데이터 게이트와 LLM 게이트를 같은 방식으로 다룰 수 있습니다.

## 설계 결정

**기준을 비대칭으로 잡았다.** 틀린 일정이 들어가는 것(precision)과 약속이 없는데 일정을 만드는 것(no_false_plans)은 배포를 막는 error로, 일정을 놓치는 것(recall)은 warn으로 두었습니다. 놓친 일정은 사용자가 직접 넣으면 되지만, 잘못 들어간 일정은 캘린더에 대한 신뢰를 잃게 만들기 때문입니다. 지표의 기준은 기술이 아니라 사용자가 겪는 비용으로 정해야 한다고 생각했습니다.

**정답은 손으로 쓰지 않고 계산한다.** `examples/schedule/generate.py`가 날짜, 시간, 동의 여부, 취소 여부를 먼저 정하고, 그 값으로 대화문과 정답을 함께 만듭니다. 대화문과 정답이 어긋날 수 없고, CI가 생성기를 다시 돌려 커밋된 데이터셋과 같은지 확인합니다. 실제 대화는 개인정보라 저장소에 둘 수 없다는 제약도 이렇게 해결했습니다.

**게이트 자체를 테스트한다.** 정답을 그대로 넣은 출력은 반드시 통과하고, 날짜 계산 오류·취소 무시·오전/오후 혼동 같은 실수를 넣은 출력은 반드시 막혀야 합니다. 이 두 가지를 CI가 매번 확인합니다. 평가 도구가 틀리면 그 위의 모든 판단이 틀리기 때문입니다.

**응답을 캐시하고 재생한다.** (모델, 설정, 시스템 프롬프트, 프롬프트)의 해시를 키로 응답을 저장합니다. 바뀐 케이스만 API를 호출하므로 무료 한도를 아끼고, 저장된 응답을 커밋해 두면 CI는 키 없이 같은 결과를 재현합니다. 프롬프트를 바꾸고 다시 기록하지 않으면 replay 모드에서 캐시가 없어 실패하는데, 이는 의도한 동작입니다. 평가하지 않은 프롬프트는 배포되지 않습니다.

**모델 호출 전에 실패한다.** 모르는 지표, 빠진 파라미터, 프롬프트에 필요한 필드가 없는 케이스는 API를 한 번도 부르기 전에 종료 코드 2·3으로 걸러집니다. 지표 계산 중 예외가 나면 점수는 0으로 처리됩니다(fail closed).

**태그별로 나눠 본다.** 전체 점수만 보면 "어디가 약한지"를 알 수 없습니다. 케이스마다 유형 태그(상대 날짜, 변경, 취소, 미확정 등)를 달아, 프롬프트를 고칠 방향이 결과에서 바로 보이게 했습니다.

## 데모 데이터셋

합성 텔레그램 대화 40건, 정답 일정 29개입니다.

| 태그 | 건수 | 확인하는 것 |
|---|---|---|
| absolute_date | 6 | "11월 3일" 같은 명시적 날짜 |
| relative_day | 6 | 내일, 모레 (자정 넘긴 답장 포함) |
| weekday | 8 | 이번 주 / 다음 주 X요일 |
| changed | 5 | 시간·날짜·장소 변경, 거절된 변경 제안 |
| cancelled | 4 | 확정 후 취소 |
| tentative | 6 | 답 없는 제안, 보류, 거절, 막연한 약속 |
| no_plan | 3 | 지난 일 이야기 등 약속이 없는 대화 |
| multiple | 3 | 한 대화에 약속 여러 개, 단체 대화 |

## 구조

```
src/llm_eval_gate/
  registry.py   지표 플러그인 레지스트리
  metrics.py    내장 지표 6종
  config.py     YAML 로딩과 사전 검증
  dataset.py    JSONL 데이터셋, 프롬프트 렌더링
  providers.py  Gemini(REST, 재시도·속도 제한), 파일, 응답 캐시
  runner.py     평가 실행, 채점, 기준선 저장
  report.py     콘솔·Markdown·JSON 리포트
  cli.py        run / validate / list-metrics
examples/schedule/
  generate.py   합성 대화와 정답 생성기
  system.txt, prompt.txt, schema.json, eval.yaml
  fixtures/     게이트 자체 테스트용 출력 (정답 / 실수 주입)
```

## 다음 단계

- 이 게이트를 통과한 프롬프트로 텔레그램 봇 → 구글 캘린더 등록 구현
- LLM에 보내기 전 대화 속 개인정보 마스킹 (pii-guard)
- 호출별 토큰·비용·지연 집계 (token-meter)

## License

MIT
