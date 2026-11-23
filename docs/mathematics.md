# 수리 모형과 계산식

가격·신호·선정·장부·베이지안 추정·예측·배분·평가를 실제 실행 코드 기준으로 정의한다. $t$는 시장별 거래일, $w$는 주간 관측, $m$은 월간 관측이다.

## 1. Part 1·2 — 두 전략과 장부

### 1.1 전략 정의 (모든 실험에서 고정)

| 항목 | 내용 |
| --- | --- |
| M | HIGH2Y\_BASE: 현재가 ÷ 504거래일 최고가(유효가격 403개 이상) |
| L | LOWVOL60: 최근 60거래일의 인접 유효 가격 단순수익률이 모두 있을 때 표본표준편차가 낮은 순 |
| 현금 | 수익률 0 (원장 설정) |
| 선정 | 각 시장의 당시 지수 편입 증권의 신호 순위. 상위 20% 신규 선정 + 기존 보유 중 상위 40% 유지. 유효 신호 50개 미만이면 보류 |
| 비중 | 기존 비중 유지, 신규 자금 균등 배분 |
| 재선정 | 월말 결정, 다음 거래일 종가 체결 |
| 비용 | 한국: 매수 6.5bp, 매도 6.5bp + 체결일 세율. 미국: 양방향 6.5bp·추가 세금/기타 비용 0의 가상 실험 가정. 순주문 상계 후 한 번만 차감. 최소주문 NAV 0.25% |

### 1.1.1 Part 1 — 2년 고가 모멘텀

P는 수정주가, t는 각 시장의 거래일, i는 원본 증권코드다. 유효 가격은 유한한 양수다.

$$
\mathcal W_{t,i}=\{u:\max(0,t-503)\le u\le t,\ P_{u,i}>0\text{이고 유한}\},\qquad
M_{t,i}=\frac{P_{t,i}}{\max_{u\in\mathcal W_{t,i}}P_{u,i}}.
$$

현재 가격이 유효하고 $|\mathcal W_{t,i}|\ge403$일 때만 신호를 낸다. M이 클수록 504거래일 고가에 가깝다. 준비 전 다른 모멘텀으로 대체하지 않는다. 편입 여부는 당일 투자대상 선정에서 적용하며, 가격 이력 자체를 현재 구성종목으로 제한하지 않는다.

### 1.1.2 Part 2 — 60일 저변동성

$$
r_{t,i}=\frac{P_{t,i}}{P_{t-1,i}}-1,\qquad
\bar r_{t,i}=\frac1{60}\sum_{j=0}^{59}r_{t-j,i},\qquad
L_{t,i}=-\sqrt{\frac1{59}\sum_{j=0}^{59}(r_{t-j,i}-\bar r_{t,i})^2}.
$$

인접 두 거래일 가격이 모두 유효할 때만 r을 계산한다. 최근 60개 수익률이 모두 있어야 L이 유효하다. 음의 표준편차이므로 높은 순위가 저변동성이다. 결측을 0이나 전일 가격으로 채우지 않는다.

### 1.1.3 두 전략 공통 — 20/40 선정과 기존 비중 유지

$\mathcal U_t$는 당일 편입 증권, $N_t=|\mathcal U_t|$다. 각 전략의 유효 신호만 내림차순 정렬하되 선정 수는 전체 편입 수 N으로 계산한다. 동점은 원본 증권코드의 고정 순서를 따른다.

$$
n_{\rm buy}=\max(1,\operatorname{round}(0.2N_t)),\qquad
n_{\rm keep}=\operatorname{round}(0.4N_t),\qquad
\mathcal A_t^s=\operatorname{Top}_{n_{\rm buy}}(s_t)\cup
\left(\mathcal H_t^s\cap\operatorname{Top}_{n_{\rm keep}}(s_t)\right).
$$

$\mathcal H_t^s$는 해당 계좌의 기존 보유 증권이다. 유효 신호가 50개 미만이면 월말 재선정을 보류한다. round는 코드의 Python 반올림이다. 편출은 관측 후 다음 거래일 매도를 계획하고, locked는 주문하지 않고 평가액을 유지한다.

월말 가용 예산을 b=1−locked 비중, 거래 가능한 선정 집합의 크기를 m, 기존 유지 종목을 K, 신규 종목을 J라고 하자. u는 현재 슬리브 내 비중, n=\|J\|, A=b−Σ\_K u\_i, q=b/m이다.

$$
n>0,\ A\ge nq:\quad W_i=u_i\ (i\in K),\quad W_j=A/n\ (j\in J).
$$

$$
n>0,\ A<nq:\quad W_i=u_i\frac{b-nq}{\sum_{\ell\in K}u_\ell}\ (i\in K),\quad W_j=q\ (j\in J).
$$

신규 종목이 없으면 기존 선정 종목 비중을 유지하고 자동 균등 재조정하지 않는다. locked 비중은 별도로 유지한다. 비월말에는 기존 비중을 유지하되 편출 주문을 계획한다. 이 W는 결정일 목표이며 실제 주문은 최소주문·현금·거래가능·비용·다음날 체결가격을 반영한다.

### 1.2 가상 계좌와 실제 계좌

두 전략은 실제 배분과 무관하게 독립 가상 계좌로 계속 운용된다(학습 자료가 배분 결과에 오염되지 않음). 관측용 수익은 당일 비용만 복원한 gross 수익이다.

$$
g_d^s=\frac{V_d^s+F_d^s}{V_{d-1}^s}-1,\qquad y_d^s=\log(1+g_d^s),\qquad s\in\{M,L\}
$$

실제 계좌는 M 슬리브, L 슬리브, 현금으로 구성된 하나의 수량 장부다. 매일 $V^{post}=B+\sum_i n_iP_i$ 항등식을 확인하고, 매도 후 매수·현금에 맞춘 매수 축소·미체결·가격 없는 보유(locked)를 그대로 기록한다.

## 2. Part 3 공통 — 위험 대용치와 배분 규칙

### 2.1 위험 대용치 V

$$
V_{t+1}=\beta V_t+(1-\beta)\,y_ty_t^\top,\qquad \beta=0.97,\qquad V_{\rm init}=V_{\rm ref}
$$

$V_{\rm ref}$는 준비 구간(2009–2011) 공통 유효 로그수익 공분산 + $10^{-8}I$. 0평균 2차 모멘트 근사다. B0·B1·B2·B2.1은 이 V에 조건부이며 V의 불확실성은 적분하지 않는다. B2.2는 공동 국면 공분산을 추정한다.

### 2.2 배분 결정 (모든 모형 공통)

각 모형은 결정일 t에 다음 보유기간 단순수익의 예측 평균 $\widehat m$과 공분산 $\widehat S$를 낸다. 로그수익 예측 $G\sim N(a,B)$에서:

$$
\widehat m_s=\exp\!\left(a_s+\tfrac{B_{ss}}2\right)-1,\qquad
\widehat S_{s\ell}=\exp\!\left(a_s+a_\ell+\tfrac{B_{ss}+B_{\ell\ell}}2\right)\left[e^{B_{s\ell}}-1\right]
$$

$$
w_t^*=\arg\max_{w}\;w^\top\widehat m-\frac{\gamma}{2}w^\top\widehat S\,w-TC_t(w),\qquad \gamma=5
$$

$$
w\ge0,\qquad \mathbf1^\top w\le 1-\text{locked},\qquad
TC_t(w)=\sum_i c_i^b\,z_i^++c_i^s\,z_i^-,\quad z=Q_tw-x_{\rm old}
$$

- 남는 비중 $1-\mathbf1^\top w$는 현금이다(현금 허용).
- QP 후보로 실제 실행계획(최소주문, 현금, 거래가능, 상계)을 만든 뒤 같은 점수로 다시 평가한다. HOLD보다 점수가 $10^{-10}$ 넘게 높고 시장 주문이 실제로 있을 때만 TRANSFER.
- 추가 λ, 배분 밴드, 확률 문턱은 쓰지 않는다.
- 결정일 t, 체결일 t+1 종가. 예측 구간은 t+2부터 t+H+1까지.

## 3. 베이지안 모형

### B0 — 기본 베이지안 (기준선)

전략별 기대 로그수익 $\mu$가 **고정**되어 있다고 보고, 들어오는 자료로 사후분포를 갱신만 한다(파일 `part3_b0_basic.py`).

$$
y_t=\mu+\varepsilon_t,\quad \varepsilon_t\sim N(0,V_t),\qquad \mu\sim N(m_0,P_0),\ m_0=0,\ P_0=V_{\rm ref}/60
$$

$$
K_t=P_{t-1}(P_{t-1}+V_t)^{-1},\quad m_t=m_{t-1}+K_t(y_t-m_{t-1}),\quad P_t=(I-K_t)P_{t-1}(I-K_t)^\top+K_tV_tK_t^\top
$$

H = 5일 예측: $a=Hm_t$, $B=H^2P_t+HV_{t+1}$. 결측 성분은 관측된 성분만으로 갱신. 매일 검토.

### B1 — 수익 차이 학습 + 공통 수준 (현금 판단)

두 부분으로 나눠 예측한 뒤 합친다(파일 `part3_b1_spread.py`, `part3_b1_level.py`). 공통 수준 $c=(y_M+y_L)/2$, 차이 $d=y_M-y_L$.

**① 차이 δ: "두 전략은 같다"로 축소하는 AR(1)**

$$
d_t=\delta_t+e_t,\ e_t\sim N(0,u^\top V_tu),\ u=(1,-1);\qquad
\delta_t=\phi\,\delta_{t-1}+\eta_t,\ \eta_t\sim N\!\big(0,(1-\phi^2)\sigma^2\big)
$$

$\delta_0\sim N(0,\sigma^2)$, $\sigma=5\%/252$(연 5% 차이 정도의 사전 폭), $\phi=0.5^{1/252}$(반감기 252일: 증거가 없으면 차이가 0으로 돌아감). 칼만 필터로 갱신하고 H = 21일 차이의 평균·분산을 예측한다.

**② 공통 수준 c: 과거평균 + 확률적 변동성 (PM-SV)**

월간 $c_m$에 대해(Pettenuzzo & Ravazzolo 2016의 사전분포):

$$
c_{m+1}=\mu+e_{m+1},\ e\sim N(0,e^{h_{m+1}});\qquad h_{m+1}=\lambda_0+\lambda_1h_m+\xi,\ \xi\sim N(0,\sigma_\xi^2)
$$

$\mu\sim N(\bar c,\psi^2s^2/n)$, $\psi=1$; $h_1\sim N(\ln s^2,0.01)$; $\lambda_1\sim N(0.95,10^{-6})$; $\sigma_\xi^{-2}\sim G(1/2,0.005)$. Gibbs(Kim–Shephard–Chib 7개 혼합 근사), 매 월말 확장 창으로 재추정, burn-in 1,000 + 보존 2,000.

**③ 결합:** $E[c],\mathrm{Var}(c)$는 수준 모형, $E[d],\mathrm{Var}(d)$는 차이 모형, $\mathrm{corr}(c,d)$는 V에서 가져와 $(y_M,y_L)=(c+d/2,\ c-d/2)$로 되돌린다. 매일 검토.

**B1 차이 필터와 수준 모형의 사후 갱신**

차이 필터는 상태 예측 후 당일 관측으로 갱신한다. V\_t는 당일 관측 전 위험 대용치다.

$$
a_t^\delta=\phi m_{t-1}^\delta,\quad R_t^\delta=\phi^2P_{t-1}^\delta+W,\quad
v_t=u^\top V_tu,\quad K_t^\delta=\frac{R_t^\delta}{R_t^\delta+v_t},\quad
m_t^\delta=a_t^\delta+K_t^\delta(d_t-a_t^\delta),\quad
P_t^\delta=(1-K_t^\delta)^2R_t^\delta+(K_t^\delta)^2v_t.
$$

한쪽 전략 관측이 결측이면 관측 갱신을 생략하고 상태 예측만 전진한다. V 역시 두 전략 관측이 모두 있을 때 갱신한다.

PM-SV의 n개 월간 수준 관측에 대해 $m_0=\bar c$, $v_0=\psi^2s_c^2/n$다. 잠재 변동성 h에 조건부인 평균 사후분포는 다음과 같다.

$$
v_\mu=\left(v_0^{-1}+\sum_{m=1}^{n}e^{-h_m}\right)^{-1},\qquad
m_\mu=v_\mu\left(v_0^{-1}m_0+\sum_{m=1}^{n}c_me^{-h_m}\right),\qquad
\mu|\cdot\sim N(m_\mu,v_\mu).
$$

$Z_m=(1,h_{m-1})$, $\lambda=(\lambda_0,\lambda_1)^\top$, 사전 평균 $m_\lambda=(0,0.95)^\top$, 사전 공분산 $D_\lambda=\operatorname{diag}(10,10^{-6})$라 하면:

$$
C_\lambda=\left(D_\lambda^{-1}+Z^\top Z/\sigma_\xi^2\right)^{-1},\qquad
b_\lambda=C_\lambda\left(D_\lambda^{-1}m_\lambda+Z^\top h_{2:n}/\sigma_\xi^2\right),\qquad
\lambda|\cdot\sim N_2(b_\lambda,C_\lambda)\ \text{단 }|\lambda_1|<1.
$$

$$
\sigma_\xi^{-2}|\cdot\sim G\left(\frac12+\frac{n-1}{2},\
0.005+\frac12\sum_{m=2}^{n}(h_m-\lambda_0-\lambda_1h_{m-1})^2\right).
$$

G는 shape–rate 정의다. 잠재 h의 갱신에는 log-χ² 오차의 7성분 정규혼합 근사를 사용한다. 예측은 각 보존 표본에서 $h_{n+1}=\lambda_0+\lambda_1h_n+\xi_{n+1}$, $c_{n+1}=\mu+\exp(h_{n+1}/2)\epsilon_{n+1}$를 모의한다.

**B1 예측 결합의 행렬식**

차이 필터의 사후분포가 $\delta_t|\mathcal F_t\sim N(m_t^\delta,P_t^\delta)$일 때, H=21일·다음날 체결을 반영한 누적 차이 예측은 다음과 같다.

$$
C_H=\sum_{h=2}^{H+1}\phi^h,\qquad
D_H=\sum_{j=1}^{H+1}\left(\sum_{\substack{h=2\\h\ge j}}^{H+1}\phi^{h-j}\right)^2,\qquad
a_d=C_Hm_t^\delta,\quad v_d=C_H^2P_t^\delta+D_HW+H\,u^\top V_{t+1}u.
$$

$W=(1-\phi^2)\sigma^2$다. 월간 수준 모형의 사후예측 모의표본에서 $a_c,v_c$를 계산한다. 수준과 차이의 상관은 $u_c=(1/2,1/2)$, $u_d=(1,-1)$에 대해 V에서 얻는다.

$$
\rho_{cd}=\frac{u_c^\top Vu_d}{\sqrt{(u_c^\top Vu_c)(u_d^\top Vu_d)}},\quad
S_{cd}=\begin{bmatrix}v_c&\rho_{cd}\sqrt{v_cv_d}\\\rho_{cd}\sqrt{v_cv_d}&v_d\end{bmatrix},\quad
A=\begin{bmatrix}1&1/2\\1&-1/2\end{bmatrix},\quad
a=A\begin{bmatrix}a_c\\a_d\end{bmatrix},\quad B=AS_{cd}A^\top.
$$

PM-SV의 나머지 사전은 $\lambda_0\sim N(0,10)$, $\lambda_1\in(-1,1)$, $\sigma_\xi^{-2}\sim G(1/2,0.005)$이며 G는 shape–rate 정의다. 코드의 kξ=0.01은 정밀도 사전 평균 100을 뜻하며, Gamma의 shape가 100이라는 뜻은 아니다. 잠재 h는 Kim–Shephard–Chib 7성분 혼합을 이용해 Gibbs로 표집한다. 월간 c는 두 전략이 모두 관측된 날짜의 $(y_M+y_L)/2$를 해당 월에 합산한 값이다. 최신 완료 월말의 수준 예측을 일별 차이 예측과 결합한다.

### B2 — 전략별 2국면 모형

M과 L 각각에 "평온 / 불안" 두 국면을 둔다(파일 `part3_b2_regime.py`, 주간 로그수익).

$$
y_t\mid S_t=k\sim N(\mu_k,\sigma_k^2),\quad k\in\{0\ \text{평온},1\ \text{불안}\},\qquad
P=\begin{bmatrix}p_{00}&1-p_{00}\\1-p_{11}&p_{11}\end{bmatrix}
$$

| 사전분포 | 값 | 의미 |
| --- | --- | --- |
| $\mu_k$ | $N(\bar y,(0.20/52)^2)$ | 두 국면 평균 모두 과거평균 중심 |
| $\sigma_k^2$ | $IG(3,\,2s^2)$ | 표본분산 중심 |
| $p_{kk}$ | $\mathrm{Beta}(19.2,0.8)$ | 주간 지속 0.96, 평균 약 25주(6개월) |

- 식별: 평온 = 분산이 작은 국면.
- 추정: FFBS(전진 필터·후진 표집) + 켤레 Gibbs. 매 월말 확장 창, 이전 체인으로 시작(burn 1,000 → 이후 300, 보존 1,000).
- 예측: 원점의 국면 필터확률에서 4주 경로를 모의(국면·모수 불확실성 포함) → 21거래일로 환산. 두 전략 상관은 V에서.
- 검토: 첫날과 월말에만.

**B2 공통 추정식**

각 재추정 원점까지의 사용 가능한 주간 표본으로 $\bar y,s^2$를 계산한다. $\tau^2=(0.20/52)^2$이며 $IG(a,b)$는 밀도가 $x^{-a-1}\exp(-b/x)$에 비례하는 정의다. $n_k$는 국면 k의 관측 수, $n_{k\ell}$은 k→ℓ 전이 수다.

$$
v_k=\left(\tau^{-2}+n_k/\sigma_k^2\right)^{-1},\quad
m_k=v_k\left(\bar y/\tau^2+\sum_{w:S_w=k}y_w/\sigma_k^2\right),\quad
\mu_k|\cdot\sim N(m_k,v_k).
$$

$$
\sigma_k^2|\cdot\sim IG\left(3+\frac{n_k}{2},\,2s^2+\frac12\sum_{w:S_w=k}(y_w-\mu_k)^2\right),\quad
p_{kk}|\cdot\sim \operatorname{Beta}(19.2+n_{kk},\,0.8+n_{k,1-k}).
$$

국면 경로는 FFBS로 표집한다. 국면 필터확률 $\alpha_w(k)$의 전진 갱신은 다음과 같다. $\ell_k$는 해당 모형의 관측 밀도다.

$$
\alpha_w(k)=\frac{\ell_k(y_w)\sum_j P_{jk}\alpha_{w-1}(j)}
{\sum_\ell\ell_\ell(y_w)\sum_jP_{j\ell}\alpha_{w-1}(j)}.
$$

초기 확률은 전이행렬의 정상분포다. 뒤로는 $p(S_w=j|S_{w+1}=k,y_{1:T},\theta)\propto\alpha_w(j)P_{jk}$로 표집한다. 매 원점 T까지의 자료만 사용하며, 과거 경로를 다시 표집해도 이미 집행한 결정은 바꾸지 않는다.

### B2.1 — 국면 간 평균 공유

전략 s마다 평균은 공유하고 위험 국면은 구분한다. M과 L의 평균을 서로 같게 강제하지 않는다.

$$
y_{s,w}|S_{s,w}=k\sim N(\mu_s,\sigma_{s,k}^2),\qquad
\mu_s\sim N(\bar y_s,\tau^2),\quad
\sigma_{s,k}^2\sim IG(3,2s_s^2),\quad p_{s,kk}\sim\operatorname{Beta}(19.2,0.8).
$$

두 국면의 자료가 하나의 공통 평균 사후분포에 함께 들어간다.

$$
v_s=\left(\tau^{-2}+\sum_w\sigma_{s,S_{s,w}}^{-2}\right)^{-1},\qquad
m_s=v_s\left(\frac{\bar y_s}{\tau^2}+\sum_w\frac{y_{s,w}}{\sigma_{s,S_{s,w}}^2}\right),\qquad
\mu_s|\cdot\sim N(m_s,v_s).
$$

분산의 조건부 분포는 위 B2 식에서 $\mu_k$를 $\mu_s$로 바꾼다. FFBS → 공통 평균 → 국면별 분산 → 전이확률을 반복한다. 하나의 평균 표본을 두 국면에 동일하게 사용하여 평균 불확실성을 예측에 반영한다. 국면별 평균을 추정한 뒤 사후 평균내는 방식이 아니다.

### B2.2 — M·L 공동 국면·공분산

두 전략의 주간 로그수익을 같은 주차로 정렬해 공동 관측 벡터를 만든다. 공통 국면 하나가 M·L 평균과 공분산을 동시에 결정한다.

$$
\mathbf y_w=\begin{bmatrix}y_{M,w}\\y_{L,w}\end{bmatrix},\qquad
\mathbf y_w|S_w=k\sim N_2(\boldsymbol\mu_k,\Sigma_k),\qquad
\boldsymbol\mu_k=\begin{bmatrix}\mu_{M,k}\\\mu_{L,k}\end{bmatrix},\qquad
\Sigma_k=\begin{bmatrix}\sigma_{M,k}^2&\rho_k\sigma_{M,k}\sigma_{L,k}\\
\rho_k\sigma_{M,k}\sigma_{L,k}&\sigma_{L,k}^2\end{bmatrix}.
$$

평균과 공분산 사전은 서로 독립이다. 표본 평균·분산은 해당 원점까지의 공동 주간 표본에서 계산한다.

$$
\boldsymbol\mu_k\sim N_2(\bar{\mathbf y},\tau^2I_2),\qquad
\Sigma_k\sim IW(\nu_0,\Psi_0),\qquad
\nu_0=7,\quad \Psi_0=4\,\operatorname{diag}(s_M^2,s_L^2).
$$

이 정의에서 $E[\Sigma_k]=\Psi_0/(\nu_0-2-1)=\operatorname{diag}(s_M^2,s_L^2)$이고 각 주변 분산 사전은 기존 B2의 $IG(3,2s_s^2)$와 같다. 사전 상관은 0 쪽으로 축소되지만 사후 상관 $\rho_k$는 국면별 자료로 추정한다.

$$
C_k=\left(\tau^{-2}I_2+n_k\Sigma_k^{-1}\right)^{-1},\qquad
b_k=C_k\left(\tau^{-2}\bar{\mathbf y}+\Sigma_k^{-1}\sum_{w:S_w=k}\mathbf y_w\right),\qquad
\boldsymbol\mu_k|\cdot\sim N_2(b_k,C_k).
$$

$$
\Sigma_k|\cdot\sim IW\left(7+n_k,\,
\Psi_0+\sum_{w:S_w=k}(\mathbf y_w-\boldsymbol\mu_k)(\mathbf y_w-\boldsymbol\mu_k)^\top\right).
$$

국면은 이변량 정규 밀도로 FFBS 표집하고 전이는 B2와 같은 Beta 조건부 분포를 사용한다. $\operatorname{tr}(\Sigma_0)\le\operatorname{tr}(\Sigma_1)$이 되도록 국면 라벨을 정리한다. 같은 주에 두 전략 관측이 모두 있는 표본만 사용하므로 L의 초기 74주가 제외된다. B2·B2.1은 전략별 주간 합, B2.2는 공동 주간 합을 쓰며 월말의 현재 부분 주는 그날까지 관측된 일간 값만 합산한다. 결측을 0으로 대체하지 않는다.

### B2 계열 — 사후예측에서 배분 입력까지

매 월말 확장 창에서 초기 burn 1,000(이후 warm burn 300), 보존 1,000, 표본당 경로 2개를 사용한다. 각 보존 표본에서 원점의 필터확률로 국면을 시작하고 전이행렬에 따라 미래 4주를 모의한다. B2·B2.1은 전략별 경로, B2.2는 공통 국면 아래 상관된 이변량 경로를 모의한다.

$$
\widetilde{\mathbf G}^{(b)}=\sum_{h=1}^{4}\mathbf y_{T+h}^{(b)},\qquad
c=\frac{21}{5\cdot4}=\frac{21}{20},\qquad
a=c\,\overline{\widetilde{\mathbf G}},\qquad
B=c\,\widehat{\operatorname{Cov}}(\widetilde{\mathbf G}).
$$

B2·B2.1의 비대각 원소는 모의한 주변 분산과 EWMA 상관으로 구성한다. B2.2는 공동 경로의 공분산을 직접 사용한다. 이 a,B를 공통 로그정규 변환식에 넣어 단순수익 평균·공분산을 얻고 비용 반영 QP와 HOLD를 비교한다. 실제 국면 혼합분포가 단일 정규분포인 것은 아니며, 로그정규 모멘트 변환은 현재 엔진의 근사다. 4주를 21거래일로 환산하는 것도 시간 척도 근사다.

두 확장은 이미 관찰한 자료에서 선택한 탐색 모형이다. B2.2는 낮은 ESS와 보존 표본 수에 따른 민감도가 남는다. [사전·수치 진단](numerical_validation.md)

### 비교 기준

EW\_monthly는 첫날과 월말 목표 M·L 비중을 각각 1/2로 두고 locked 가용 한도를 적용한다. M\_only·L\_only는 각 가상 전략의 독립 계좌다. Cash\_timing은 B1 결합식에서 $a_d=0$, $v_d=H\,u_d^\top Vu_d$로 두어 차이에 대한 견해만 제거하고 공통 수준의 현금 판단을 유지한다.

| 이름 | 내용 |
| --- | --- |
| EW\_monthly | 월말 50:50 (주 비교 기준) |
| M\_only, L\_only | 단일 전략 |
| Cash\_timing | B1에서 차이 예측을 0으로 둔 것: M↔L 판단 없이 현금 비중만 조절. B1·B2의 개선이 현금 타이밍 때문인지 가르는 기준 |

## 4. 평가 기준

- 준비: 2009\~2011년. 평가: 2012\~2025년. 개발: 2012\~2019년. 테스트: 2020\~2025년.
- 비용 후 일별 수익률로 CAGR, Sharpe(연율화 252), CER(γ=5), MDD, 회전율과 평균 현금을 계산한다. 월간 power CER는 A=5다.
- 평균 현금은 저장된 노출의 보완값 1−exposure 기준이다.
- 같은 시장의 정책과 기준정책을 동일 날짜로 비교한다. 순환 블록 부트스트랩은 블록 21일·2,000회·seed 42다.
- p\_boot는 차이가 0 이하인 재표집 비율인 단측 지표다. 95% 구간과 구분하며 다중비교 보정은 하지 않는다.
- B1의 M−L 차이 예측은 0 예측 대비 R²와 Diebold–Mariano 통계로 진단한다.

**성과 지표의 수식**

실제 계좌의 비용 후 일별 수익률을 R\_d, 관측 수를 n, 표본표준편차를 s\_R(ddof=1)라 하자. 현금 이자와 Sharpe의 무위험수익률은 0이다.

$$
W_0=1,\quad W_j=\prod_{d=1}^j(1+R_d),\quad
\operatorname{CAGR}=W_n^{252/n}-1,\quad
\operatorname{Vol}=\sqrt{252}\,s_R,\quad
\operatorname{Sharpe}=\sqrt{252}\frac{\bar R}{s_R}.
$$

$$
\operatorname{CER}=252\bar R-\frac{\gamma}{2}\,252s_R^2,\quad\gamma=5,\qquad
\operatorname{MDD}=\min_{0\le j\le n}\left(\frac{W_j}{\max_{0\le u\le j}W_u}-1\right).
$$

이는 표의 일별 평균–분산 CER다. 별도 월간 power CER는 월말 NAV 비율 $Z_m$에 대해 $12\{[\overline{Z_m^{1-A}}]^{1/(1-A)}-1\}$, A=5로 계산한다. 최초 월말 이전의 부분 월 수익은 이 별도 지표에 포함하지 않는다.
