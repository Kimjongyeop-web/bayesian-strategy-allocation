# Bayesian Dynamic Strategy Allocation

KOSPI200과 S&P 500에서 2년 고가 모멘텀(M), 60일 저변동성(L), 현금을 베이지안 사후예측과 비용을 고려해 배분합니다. 두 시장은 별도 계좌와 달력으로 실행합니다.

## 보고서 작성 안내

1. [최종 연구 보고서](reports/final/README.md): 두 시장의 성과표·통계적 비교·결론과 한계.
2. [수리 모형](docs/mathematics.md): Part 1·2와 B0·B1·B2·B2.1·B2.2의 신호·사전·사후추정·예측·배분·평가식.
3. [연구 방법](docs/methodology.md) · [데이터 규격](docs/data.md): 거래 시점·장부·비용·원본 정의.
4. [수치 검증](docs/numerical_validation.md): 표본·시드·ESS에 따른 해석 범위.

필요한 설명·성과·선정 근거·압축 재현 입력은 이 저장소에 포함돼 있다. 다른 브랜치나 Notion을 읽지 않아도 보고서를 작성하고 저장된 결과를 재계산할 수 있다.

## 실행

Python 3.11, 저장소 최상위 PowerShell 기준입니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m sprint2 run --market all --run-id full
.\.venv\Scripts\python -m sprint2 report reports/final
.\.venv\Scripts\python tools/verify_final.py
```

원본 두 파일을 `data/raw/`에 준비합니다. 파일명·해시는 [데이터 규격](docs/data.md)에 있습니다. 원본 없이도 포함된 최종 자료로 report 명령을 실행할 수 있습니다. 단독 실행은 `--market kospi200` 또는 `--market sp500`, 데이터 검사는 `check-data --market all`입니다. 외부 경로는 단독 실행의 `--data` 또는 시장별 설정으로 지정합니다.

## 구성

| 경로 | 역할 |
|---|---|
| sprint2/ | 신호, 베이지안 모형, 공통 장부, 평가와 CLI |
| config/common.yaml | 공통 전략·모형·평가 설정 |
| config/kospi200.yaml, config/sp500.yaml | 시장별 데이터·비용·통화 |
| reports/final/ | 비교표·그림·시장별 결과와 압축 재계산 자료 |
| tests/ | 체결·누출·캐시·시장·발행 자료 검증 |
| docs/ | 방법론·데이터·핵심 선정 근거 |

## 결과

[두 시장 비교표](reports/final/comparison.csv) · [비교 그림](reports/final/comparison.png) · [한국](reports/final/kospi200/summary_by_period.csv) · [미국](reports/final/sp500/summary_by_period.csv)

공통 평가기간 2012~2025년, 개발 2012~2019년, 테스트 2020~2025년입니다. 정책은 M_only, L_only, EW_monthly, Cash_timing, B0, B1, B2, B2_1, B2_2입니다. 코스피는 모멘텀 단독 성과가 강하고, 미국 B2는 테스트 낙폭을 완화했으나 B1·B2의 50:50 대비 통계적 우위는 입증되지 않았습니다.

B2.1은 국면 간 평균 공유, B2.2는 M·L 공동 국면·공분산 추정입니다. 확장 모형 중 한국은 B2.1, 미국 위험조정 지표는 B2.2가 유망했으나 확장 모형의 우위를 확정할 근거는 부족합니다. B2.2의 낮은 유효표본수와 표본 수에 따른 민감도는 [방법론](docs/methodology.md)에 명시했습니다.

성과는 배당 제외 가격수익률·비용 차감 기준입니다. 미국 양방향 6.5bp·추가 비용 0은 가정값이며 한국 매도세를 적용하지 않습니다. 현금 이자 0, 가격 상실 보유의 마지막 평가액 유지, 과거 자료를 이용한 선택의 한계가 있습니다. 새 미관측 표본 검증이 아닙니다.

## 재현

전체 실행은 `results/comparison/<run_id>/`에 상세 주문까지 저장합니다. 기존 결과는 보존하고 중복 이름에는 번호를 붙입니다. 최종 묶음은 daily/decisions/allocation/forecasts 압축 CSV, schema, config, manifest로 표·그림·진단을 재생성합니다. 원본과 캐시는 Git에 포함하지 않습니다.

[방법론](docs/methodology.md) · [선정 근거](docs/evidence/README.md)
