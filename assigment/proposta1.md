# Simulació Assigment 1: Proposta 1

L'entregable per a la primera part del curs de "Simulació" ha de consistir en un treball bastat en un dels següents temes desenvolupats a classe:

```
Option 1, 2: Resampling techniques(Permutational test or Bootstrap Confidence Intervals). Case of studies. 
Option 3: Simulation studies in Statistics. Case of Study
```

La meva primera proposta es basa en *Option 3*, ja que considero que és la que obre la porta a un treball més versàtil i no tant restringit com la resta. A falta de detallls en què ha de contenir aquesta opció de treball, la meva idea en terms generals es descriu a continuació.

El *Case of Study* és la introducció dels *e-values* com a eina per al contrast d'hipòtesi i la seva comparació ambs els *p-values*. Els *e-values* són una línia d'investigació molt activa actualment ja que corregeixen moltes de les limitacions dels p-valors que s'han anat identificat durant els anys en totes les branques de la literatura acadèmica que els apliquen, tot i ser "statistical cousins" d'aquests mateixos. En particular, destacar les següents propietats (descrites a [3]):

1.  Els *p-values* i els *e-values* com a objectes existeixen exactament sota les mateixes condicions, sigui quina sigui l'estructura del contrast d'hipòtesi. A més a més, un pot transformar d'un a l'altre en un procés que es coneix com a *calibració*.

2. Els *e-values* són flexibles en el disseny i continuació seqüencial d'un experiment: així com la validesa dels *p-valors* només està garantida sota un disseny rigorós **a priori** de l'experiment, els e-valors obren la porta a la flexibilitat en la experimentació, permetent aturar o continuar l'exeperiment en qualsevol instant de temps, tot guarantint el control de l'error de Tipus I; o el que és encara més destacat, permeten modificar el nivell de significació $\alpha$ en base a les dades prèviament observades. 
Observem que tot es duu a terme en una "sequential fashion". 

El treball mantindrà l'estructura *ADMEP* descrita a [2] i trindrà un component teòric que no entrarà en molt detall en els fonaments matemàtics. La idea serioa enumerar una serie de propietats que volem verificar mitjaçant simulació de Monte Carlo.

Proposo que, un cop estiguem alineats amb el projecte, elaborem una proposta formal del projecte i l'enviem al professor de l'assignatura per assegurar-nos que seria una treball vàlid.

------------
# ADMEP

## A. Aims

The primary aim of this simulation is to validate the statistical properties of e-values and e-processes, specifically their robustness against optional stopping (or "peeking"), in contrast to the known failure of traditional $p$-values in this setting.

The objectives are structured around verifying core properties of e-processes in the context of **Sequential Anytime-Valid Inference (SAVI)**:

1.  **Validity (Type I Error Control):** Verify that the e-process maintains the Type I error rate below the predefined significance level $\alpha$ (e.g., 0.05), regardless of the data-dependent stopping time ($\tau$) chosen by the user.
2.  **Comparison and Demonstration:** Empirically demonstrate how the traditional $p$-value procedure fails in the presence of optional stopping, leading to the inflation of the Type I error rate potentially up to 100% (see [1]).
3.  **Efficiency:** Assess the statistical efficiency (power and speed of rejection) of the e-process when the data is generated under the alternative hypothesis $H_1$.

## D. Data-Generating Mechanisms (DGM)

The DGM defines how pseudo-random sampling is used to create data. This simulation study will focus on sequentially generated data under a Gaussian model with unknown variance.

### 1. Model Specification

*   **Data Structure:** $\textit{i.i.d}$. observations $X_1, X_2, \ldots, X_N$ are collected sequentially up to a maximum sample size $N$.
*   **Distribution:** $X_i \sim N(\mu, \sigma^2)$, with $\sigma^2$ unknown.
*   **Test Setup (Sequential $t$-test):** Null Hypothesis $H_0: \mu = 0$ (or $P \in \mathcal{P}=\{N(0, \sigma^2): \sigma>0\}$) against Alternative $H_1: \mu \neq 0$ (or $Q \in \mathcal{Q} = \{N(\mu, \sigma²): \mu \neq 0, \sigma > 0\}$).

### 2. Scenarios

Two main scenarios to assess validity and power:

*   **DGM 1 ($H_0$ is true):** Set $\mu = 0$. This scenario is used to estimate the Type I Error Rate.
*   **DGM 2 ($H_0$ is not true):** Set $\mu = \mu_1 \neq 0$. This scenario is used to estimate the Power and speed of rejection.

### 3. Optional Stopping Rule $(\tau)$

 For a fixed significance level $\alpha$ (e.g., $\alpha=0.05$):
    $$\tau_E = \inf \left\{ n \in [2, N]: E_n \geq 1/\alpha \right\}$$
    $$\tau_P = \inf \left\{ n \in [2, N]: P_n \leq \alpha \right\}$$
    where $E_n$ and $P_n$ are the $E$-value and $p$-value at time $n$, respectively.
    (If the condition is never met, set $\tau = N$.)

### 4. Number of Repetitions ($n_{sim}$)

The number of repetitions must be chosen to achieve an acceptable Monte Carlo Standard Error (MCSE) for the Type I error rate (Coverage). For instance, if we set $\alpha=0.05$ and desire an MCSE of $0.005$ then $n_{sim} \approx 1,900$ repetitions, calculated based on the binomial distribution for coverage/rejection rate [2].

## M. Methods



### Method 1: Anytime-Valid E-Process

The test procedure is defined by the e-process $M = (M_n)_{n \in N}$ constructed for the $t$-test (see Example 3.24 in [3]):

$$
E_n = \sqrt{\frac{c²}{n+c²}}\left(\frac{(n+c²)V_n}{(n+c²)V_n -S_n²}\right)^{n/2} \quad \text{is an }E\text{-variable for any }c>0.
$$

*   **E-Process:** Calculate $M_n$ based on $X_1, \ldots, X_n$ for $n=2, 3, \ldots, N$.
*   **Decision Rule:** Reject $H_0$ if the value of the e-process at the stopping time $\tau$ satisfies $M_\tau \geq 1/\alpha$.

The validity of this test ($P(M_\tau \geq 1/\alpha) \leq \alpha$) is guaranteed by Ville's inequality regardless of how the stopping time $\tau$ is defined.

### Method 2: Naive Sequential P-value

*   **Test Statistic:** Calculate the $t$-test $p$-value $P_n$ based on $X_1, \ldots, X_n$ for $n=2, 3, \ldots, N$.
*   **Decision Rule:** Reject $H_0$ if the $p$-value at the stopping time $\tau$ satisfies $P_\tau \leq \alpha$.

This method is expected to inflate Type I error due to the non-validity of $p$-values under optional stopping.

## E. Estimands and Other Targets

The target of this simulation study is fundamentally **Testing a Null Hypothesis**.

| Statistical Task | Target | Description |
| :--- | :--- | :--- |
| Hypothesis Testing | **Null Hypothesis $H_0$** | To evaluate Type I error rate and power. |
| Efficiency | **Expected Stopping Time ($\mathbb{E}[\tau]$)** | To assess efficiency under the alternative $H_1$. |

## P. Performance Measures

| Measure Category | Performance Measure | Computation / Relevance | MCSE Formula|
| :--- | :--- | :--- | :--- |
| **Validity (DGM 1)** | **Type I Error Rate** | Proportion of runs under $H_0$ where the test rejects at time $\tau$. This is expected to be $\leq \alpha$ for the e-process and $> \alpha$ for the $p$-value. | $\sqrt{\frac{\text{Rate} \times (1 - \text{Rate})}{n_{sim}}}$ |
| **Efficiency (DGM 2)** | **Power** | Proportion of runs under $H_1$ where the test rejects at time $\tau$. | $\sqrt{\frac{\text{Power} \times (1 - \text{Power})}{n_{sim}}}$ |
| **Efficiency (DGM 2)** | **Expected Stopping Time ($\mathbb{E}[\tau]$)** | The average sample size used in runs that resulted in a rejection under $H_1$.| MCSE for the mean (TODO) |



## Referències

[1] Johari, R., Koomen, P., Pekelis, L., & Walsh, D. (2017). Peeking at A/B Tests: Why it matters, and what to do about it. Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 1517-1525. https://doi.org/10.1145/3097983.3097992

[2] Morris, T. P., White, I. R., & Crowther, M. J. (2019). Using simulation studies to evaluate statistical methods. Statistics in Medicine, 38(11), 2074-2102. https://doi.org/10.1002/sim.8086

[3] Ramdas, A., & Wang, R. (2025). Hypothesis testing with e-values. Foundations and Trends® in Statistics, 1(1-2), 1-390. https://doi.org/10.1561/3600000002






