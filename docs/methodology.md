# 연구 방법

## 연구 대상

KOSPI200과 S&P 500에서 M(HIGH2Y_BASE), L(LOWVOL60), 현금을 독립 시장 계좌로 배분한다. 공통 평가기간은 2012~2025, 개발은 2012~2019, 테스트는 2020~2025다. 신호 이력은 2008년부터, 준비 구간은 2009~2011년이다.

[수리 모형](mathematics.md)은 Part 1·2의 신호·선정·비중, B0·B1·B2·B2.1·B2.2의 관측식·사전·조건부 사후분포·예측과 공통 배분·평가식을 정의한다. 모든 수식은 실행 코드와 동일한 모수화를 사용한다. 특히 B1의 변동성 혁신 정밀도 사전은 shape–rate Gamma(1/2,0.005)다.

## 계좌와 실행

- 두 가상 전략을 배분 계좌와 독립적으로 운용한다. 모형 관측은 가상 계좌의 당일 비용만 복원한 gross 로그수익률이며 비활성 보유일은 결측이다.
- 당일 편입과 유효 신호를 사용해 상위 20% 신규 선정·상위 40% 기존 보유 유지. 월말 재선정, 기존 비중 유지, 유효 신호 50개 미만이면 보류한다.
- t 종가 관측·결정 후 다음 거래일 종가 체결. 보유기간 예측은 t+2~t+H+1이다. 매도 후 매수, 현금 제약·NAV 0.25% 최소주문·거래가능·순주문 상계를 적용한다.
- 가격 없는 보유는 마지막 유효 가격으로 평가하되 현금으로 회수하지 않는다. locked 비중은 신규 배분 가용 자본에서 제외한다. 평가는 체결 가능한 가격과 구분한다.
- QP 후보의 실제 실행계획을 재평가하여 HOLD보다 점수가 1e-10 넘게 높고 시장 주문이 있을 때 TRANSFER한다. gamma=5, 비중은 음수가 아니며 가용 자본을 넘지 않는다.

## 시장·수익률·비용

원본 정의와 해시는 [데이터 규격](data.md), 초기자본과 비용은 config/kospi200.yaml·sp500.yaml에 있다. 한국은 10,000,000 KRW, 미국은 100,000 USD다. 미국에는 한국 매도세를 적용하지 않는다. 미국 양방향 6.5bp·기타 비용 0은 가정이다.

배당 제외 수정주가 기반·비용 후 성과다. 인접 유효 가격만으로 수익률을 계산하며 휴장일·결측을 채우지 않는다. S&P 증권 구간과 주식 클래스를 합치지 않는다. 현금 이자 0·분수 수량·가격 상실 평가 유지 가정이며 배당·상폐 결제 장부는 별도로 구현하지 않았다.

## 결과와 해석

[최종 보고서](../reports/final/README.md)에 두 시장의 전체·테스트 성과와 주요 비교를 제공한다. 개발기간·비용·회전율·HOLD/TRANSFER·유의성 원수치는 같은 폴더의 CSV에 있다. 신호 선정 근거는 [선정 기준](evidence/README.md), 모형의 표본·시드·수치 한계는 [수치 검증](numerical_validation.md)에 있다. 참조 계산과 최종 계좌를 구분하며, 단일 전략 대비 보편적인 우위나 새로운 표본외 검증을 주장하지 않는다.

## 참고문헌

West, Harrison & Migon (1985), Dynamic Generalized Linear Models and Bayesian Forecasting. Kim, Shephard & Chib (1998), Stochastic Volatility: Likelihood Inference and Comparison with ARCH Models. Kim & Nelson (1999), State-Space Models with Regime Switching. Pettenuzzo & Ravazzolo (2016), Optimal Portfolio Choice under Decision-Based Model Combinations. Boyd et al. (2017), Multi-Period Trading via Convex Optimization. 코드·설정·원본 해시·실행 환경은 최종 묶음의 config·manifest에 기록한다.
