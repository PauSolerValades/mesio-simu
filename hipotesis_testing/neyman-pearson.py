import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Exemple sobre Neyman-Pearson

    La millor manera d'entendre quelcom és programar-la. Per a fer-ho considerem l'exemple més senzill possible d'on aplicar un test d'hipòtesi paramètric per a trobar si la hipòtesi és certa o no, que és un *Z-test*.

    ## Context

    Volem plantejar un experiment on volem veure que la mitjana d'una població segueix una normal $N(\mu, \sigma^2)$ on la variancia és coneguda. Encara no hem recollit les dades, i per tant, el primer que volem és trobar quantes dades s'han de recollir.

    ## 1. Effect size
    Quan tingui el context plantejat potser penso en fer això.

    ## 2. Selecció d'hipòtesis

    En aquest cas, volem veure que la mitjana $\mu$ és un valor $\mu_0$ o un altre, per tant, escollim un test de doble cua amb les següents hipòtesi

    $$
    H_0: \mu = \mu_0 \\
    H_1: \mu = \mu_1
    $$

    ## 3. Error de tipus 1

    Definim l'error de tipus 1 $\alpha$ com l'error d'equivocar-nos (és a dir, rebutjar $H_0$) si $H_0$ era certa.

    $$
    \alpha = \mathbb{P}(\textbf{x} \in R_\alpha | H_0 \text{true}) = sup_{\theta_0 \in \Theta} \mathbb{P}_{\theta_0}(\mathbb{X} \in R_\alpha)
    $$

    $\alhpa$ és el llindar de l'error que estem disposats a cometre rebutjant la hipòtesi nul·la, o el que és el mateix, quants falsos positius ens podem permetre.

    La fixem en el seu estàndard habitual $\alhpa = 0.05$
    """
    )
    return


@app.cell
def _():
    alpha = 0.05
    return (alpha,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4. Error de tipus 2

    Definim l'error de tipus 2 $1- \beta$ com l'error d'equivocar-nos no rebutjant $H_0$ quan resulta que $H_0$ era falsa.

    $$
    1-\beta = \mathbb{P}(\textbf{x} \notin R_\alpha | H_0 \text{false}) = sup_{\theta_1 \in \Theta_1} \mathbb{P}_{\theta_1}(\mathbb{X} \notin R_\alpha)
    $$

    $1 - \beta$ és quants falsos negatius ens volem permetre.

    La fixem en el seu estàndard habitual $1-\beta = 0.2 \Leftarrow \beta = 0.8$

    Amb l'elecció d'aquests dos valors, la regió crítica $R_\alpha$ ha quedat explícitament determinada, ho veurem al següent apartat.
    """
    )
    return


@app.cell
def _():
    beta = 0.8
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5. Càlcul del tamany mostral $n$

    Primer, repàs de l'estadístic $Z$. Tenint una srs $X_1 \ldots X_n \sim N(\mu, \sigma^2)$ sabem que sota $H_0$ podem estandaritzar la normal

    $$
    Z = \frac{\sqrt{n}(\bar{X}_n - \mu_0)}{\sigma} \sim N(0,1)
    $$

    i sota $H_1$

    $$
    Z = \frac{\sqrt{n}(\bar{X}_n - \mu_1)}{\sigma} \sim N(0,1)
    $$

    Per a calcular el tamany mostral hem d'utilitzar la definició de regió crítica i l'estdístic del test $Z$, que és la següent:

    $$ R_\alpha = \{\textbf{X} \in \mathbb{R} | Z \le C\} $$

    on $C$ és el límit de la regió crítica. En el cas d'un test amb doble cua, això vol dir que la regió de rebuig serà tan a l'esquerra com a la dreta de la distribució, tal com es mostra la següent imatge (els valors donats han estat escollits a dit) 
    """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    C_ip, mu_0p, C_sp = -3, 0, 3 

    lb, ub = -8, 8
    sigmap = 2
    x = np.linspace(lb, ub, 1000)
    plt.plot(x, norm.pdf(x, loc=0, scale=sigmap), 'b-', label="$H_0$")
    plt.axis()

    plt.axvline(C_ip, color='red')
    plt.axvline(mu_0p, color='black')
    plt.axvline(C_sp, color='red')

    x_left  = np.linspace(lb, C_ip, 200)
    plt.fill_between(x_left, norm.pdf(x_left, scale=sigmap), color='orange', alpha=0.3)
    x_right = np.linspace(C_sp, ub, 200)
    plt.fill_between(x_right, norm.pdf(x_right, scale=sigmap), color='orange', alpha=0.3)

    plt.text(C_ip, 0.02, r'$C_i(\alpha)$',   ha='left', va='bottom', fontsize=12, color='red')
    plt.text(mu_0p, 0.02, r'$\mu_0$', ha='left', va='top', fontsize=12, color='black')
    plt.text(C_sp, 0.02, r'$C_s(\alpha)$',   ha='right', va='bottom', fontsize=12, color='red')

    plt.xlabel('X')
    plt.ylabel('Densitat de Probabilitat')
    plt.legend()
    plt.grid(True)
    plt.show()
    return lb, mu_0p, norm, np, plt, sigmap, ub, x


@app.cell
def _(mo):
    mo.md(
        r"""
    Per tant, volem trobar els valor de $C_i$ i $C_s$ que determinen la regió crítica. Aquests estan explícitament determinats per $\alpha$ i per la $\sigma^2$ de la mostra. Per tant, volem trobar la relació entre la normal sobta $H_0$ i la normal estàndar, ja que allà sabem que l'àrea sota la corba ve donada per $F_N^{-1}(\alpha/2)$ i $F_N^{-1}(1 - \alpha/2)$.

    $$
    z_{\alpha/2} = F_N^{-1}(\alpha/2)\\
    -z_{\alpha/2} = F_N^{-1}(1 - \alpha/2)
    $$

    La següent imatge ens ensenya amb els mateix eix que el dibuix anterior on són els valors de la normal escalada centrada a zero vs la normal estandaritzada.
    """
    )
    return


@app.cell
def _(alpha, norm, np):
    z_a2 = norm.ppf(alpha/2)
    min_z_a2 = norm.ppf(1 - alpha/2)

    print(z_a2, min_z_a2)

    assert np.isclose(z_a2, -min_z_a2, atol=1e-12), \
           f"Values differ: {z_a2} vs {-min_z_a2}"
    return (z_a2,)


@app.cell
def _(lb, mu_0p, norm, np, plt, ub, x, z_a2):
    plt.plot(x, norm.pdf(x, loc=0), 'b-', label="$H_0$")
    plt.axis()

    plt.axvline(z_a2, color='red')
    plt.axvline(mu_0p, color='black')
    plt.axvline((-1)*z_a2, color='red')

    x_left_e  = np.linspace(lb, z_a2, 200)
    plt.fill_between(x_left_e, norm.pdf(x_left_e), color='orange', alpha=0.3)
    x_right_e = np.linspace((-1)*z_a2, ub, 200)
    plt.fill_between(x_right_e, norm.pdf(x_right_e), color='orange', alpha=0.3)

    plt.text(z_a2, 0.02, r'$z_{\alpha/2}$',   ha='left', va='bottom', fontsize=12, color='red')
    plt.text((-1)*z_a2, 0.02, r'$z_{\alpha/2}$',   ha='right', va='bottom', fontsize=12, color='red')

    plt.xlabel('X')
    plt.ylabel('Densitat de Probabilitat')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Per tant, recapitulant, volem trobar $C_s$ i $C_i$ en funció d'$z_{\alpha/2}$.

    $$
    R_\alpha = \{\textbf{X} \in \mathbb{R}^n | Z \ge C_s \land Z \le C_i \} 
    $$

    Estandaritzem la mostra per trobar-ne la relació segons els talls a l'àrea crítica. 

    $$
    Z = \frac{\sqrt{n}(\bar{X}_n - \mu_0)}{\sigma} \le z_{\alpha/2} \Rightarrow \bar{X}_n \le \mu_0 + z_{\alpha/2}\frac{\sigma}{\sqrt{n}} = C_s
    $$

    $$
    Z = \frac{\sqrt{n}(\bar{X}_n - \mu_0)}{\sigma} \le -z_{\alpha/2} \Rightarrow \bar{X}_n \ge \mu_0 - z_{\alpha/2}\frac{\sigma}{\sqrt{n}} = C_i
    $$

    L'interval ($C_i$, $C_s$) de trobar l'interval que ens defineix el contrari de la regió crítica! Aquesta és la decisió directa sobre l'elecció d'$\alpha$.
 
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Relació amb $\beta$

    L'error tipus 1 ens determina la regió crítica sota $H_0$. L'error tipus II en canvi assumeix que $H_1$ és certa, i per tant es posa en el cas $\mu \ne \mu_0$. La primera assumpció que farem és només per comoditat, que és assumir que $\exists \mu_1 \ne \mu_0$ la qual determina la hipòtesi alternativa. Aquest paràmetre és absolutament fictici, és a dir, no sabem mai quin valor pren realment, però sabem segur que estarà definit i que serà un valor concret $\mu_1 \in \mathbb{R}$. Per això, prenc unilateralment la llibertat de denotar $\beta$ com una funció $\beta(\mu_1)$

    A més a més, i només per usos didàctics i no repetir els càlculs, que $\mu_1 > \mu_0$. La distribució és simètria, per tant això no comporta cap pèrdua de generalitat.

    L'error tipus II és

    $$ \beta = \mathbb{P}(\mathbf{X} \notin \mathbb{R}^n | \mu = \mu_1) = \mathbb{P}(\bar{X}_n \le C_s)$$

    L'error tipus II ens està dient que, si saps que $H_1$ és certa, la probabilitat d'equivocarte és que la mitjana mostral caigui fora de la regió crítica superior (recordem que hem dit que $\mu_1 > \mu_0$). Per exemple, a la imatge següent seria el cas en que s'accepta la hipòtesi nul·la!

    """
    )
    return


@app.cell
def _(norm, np, plt, sigmap):
    C_ip_h0, mu_0p_h0, C_sp_h0 = -3, 0, 3 
    mu_1p_h1 = 2 

    sigmap_h0 = 2
    sigmap_h1 = 3
    lbm, ubm = -8, 16
    xb = np.linspace(-8, 16, 2000)
    plt.plot(xb, norm.pdf(xb, loc=0, scale=sigmap_h0), 'b-', label="$H_0$")
    plt.axis()

    plt.axvline(C_ip_h0, color='red')
    plt.axvline(mu_0p_h0, color='black')
    plt.axvline(C_sp_h0, color='red')

    x_left_h0  = np.linspace(lbm, C_ip_h0, 200)
    plt.fill_between(x_left_h0, norm.pdf(x_left_h0, scale=sigmap), color='orange', alpha=0.3)
    x_right_h0 = np.linspace(C_sp_h0, ubm, 200)
    plt.fill_between(x_right_h0, norm.pdf(x_right_h0, scale=sigmap), color='orange', alpha=0.3)

    plt.text(C_ip_h0, 0.02, r'$C_i(\alpha)$',   ha='left', va='bottom', fontsize=12, color='red')
    plt.text(mu_0p_h0, 0.02, r'$\mu_0$', ha='left', va='top', fontsize=12, color='black')
    plt.text(C_sp_h0, 0.02, r'$C_s(\alpha)$',   ha='right', va='bottom', fontsize=12, color='red')

    # H_1
    plt.plot(xb, norm.pdf(xb, loc=mu_1p_h1, scale=sigmap_h1), 'b-', label="$H_1$")
    plt.axvline(mu_1p_h1, color='black')
    plt.text(mu_1p_h1, 0.02, r'$\mu_1$', ha='left', va='top', fontsize=12, color='black')

    plt.xlabel('X')
    plt.ylabel('Densitat de Probabilitat')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Per a trobar-ho, operem la desigualtat de dins de la probabilitat fins que estigui normalitzada, tal que així

    $$
    \mathbb{P}\lparen\ \frac{\sqrt{n}(\bar{X}_n - \mu_1)}{\sigma} \le  \frac{\sqrt{n}(C_s - \mu_1)}{\sigma} \rparen = \beta(\mu_1)
    $$

    I això és la definició de $F_N(x) = P(N \le x)$ amb $N$ sent una normal estàndard, i per tant

    $$
    z_\beta = \frac{\sqrt{n}(C_s - \mu_1)}{\sigma} 
    $$

    hem trobat la relació entre $z_\beta$ i el límit $C_s$.

    Ara, igualant la $C_s$ amb l'expressió de sota $H_0$ i $H_1$
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
