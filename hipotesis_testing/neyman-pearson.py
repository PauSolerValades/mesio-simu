import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


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
    El tamany de l'effecte és la distància més petita que estem segurs de voler detectar. Al no tenir una $\mu_1$ fixa, sinó ser un valor que encompassa tots els valors excepte $\mu_0$, hem d'acotar com de prim volem mirar al detectar una differència.

    El tamany de l'effecte va molt lligat amb l'estadístic escollit, en el sentit que l'estadístic que escullim determinarà _com_ expressem el tamany de l'efecte, però nosaltres n'hem de determinar la magnitud segons el context. Imaginem-nos que un effect size suficient seria detectar una diferència de 1 entre el valor d'$H_0$ i el el valor d'$H_1$.

    Denominem el tamany d'effecte com a $d$, i en el cas d'un Z-test serà $d = \mu_1 - \mu_0$

    ## 2. Selecció d'hipòtesis

    En aquest cas, volem veure que la mitjana $\mu$ és un valor $\mu_0$ o un altre, per tant, escollim un test de doble cua amb les següents hipòtesi

    $$
    H_0: \mu = \mu_0 \\
    H_1: \mu \ne \mu_0
    $$

    ## 3. Error de tipus 1

    Definim l'error de tipus 1 $\alpha$ com l'error d'equivocar-nos (és a dir, rebutjar $H_0$) si $H_0$ era certa.

    $$
    \alpha = \mathbb{P}(\textbf{X} \in R_\alpha | H_0 \text{true}) = sup_{\theta_0 \in \Theta} \mathbb{P}_{\theta_0}(\mathbb{X} \in R_\alpha)
    $$

    $\alpha$ és el llindar de l'error que estem disposats a cometre rebutjant la hipòtesi nul·la, o el que és el mateix, quants falsos positius ens podem permetre.

    La fixem en el seu estàndard habitual $\alpha = 0.05$
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

    Definim l'error de tipus 2 $\beta$ com l'error d'equivocar-nos no rebutjant $H_0$ quan resulta que $H_0$ era falsa.

    $$
    \beta = \mathbb{P}(\textbf{X} \notin R_\alpha | H_0 \text{false}) = sup_{\theta_1 \in \Theta_1} \mathbb{P}_{\theta_1}(\mathbb{X} \notin R_\alpha)
    $$

    $\beta$ és quants falsos negatius ens volem permetre.

    La fixem en el seu estàndard habitual $1-\beta = 0.2 \Rightarrow \beta = 0.8$

    Amb l'elecció d'aquests dos valors, la regió crítica (o regió de rebuig) $R_\alpha$ ha quedat explícitament determinada, ho veurem al següent apartat.
    """
    )
    return


@app.cell
def _():
    beta = 0.8
    return (beta,)


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

    Per a calcular el tamany mostral hem d'utilitzar la definició de regió de rebuig:

    $$ R_\alpha = \{\textbf{X} \in \mathbb{R}^n  : \space |\bar{X}_n| \ge C\} $$

    on $C$ és el límit de la regió crítica. En el cas d'un test amb doble cua, això vol dir que la regió de rebuig serà tan a l'esquerra com a la dreta de la distribució, tal com es mostra la següent imatge (els valors donats han estat escollits a dit)
    """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    import numpy as np

    def plot_normal_pdf(ax, mu, sigma, label, color='blue', linestyle='-', x_range=None):
        if x_range is None:
            x_range = [mu - 4*sigma, mu + 4*sigma]
        x = np.linspace(x_range[0], x_range[1], 1000)
        y = norm.pdf(x, loc=mu, scale=sigma)
        ax.plot(x, y, color=color, label=label, linestyle=linestyle)
        ax.axvline(mu, color=color, linestyle='--', linewidth=0.8)
        # Col·loca el text de la mitjana a sobre del pic
        ax.text(mu, np.max(y) * 1.05, f'$\\mu={mu}$', ha='center', color=color)

    def shade_rejection_region(ax, mu, sigma, C_i, C_s, x_range, label=r'Rebuig ($\alpha$) sota $H_0$'):
        # Cua esquerra
        x_left = np.linspace(x_range[0], C_i, 200)
        y_left = norm.pdf(x_left, loc=mu, scale=sigma)
        ax.fill_between(x_left, y_left, color='orange', alpha=0.4, label=label)
    
        # Cua dreta
        x_right = np.linspace(C_s, x_range[1], 200)
        y_right = norm.pdf(x_right, loc=mu, scale=sigma)
        ax.fill_between(x_right, y_right, color='orange', alpha=0.4) # Sense label per no duplicar

        # Línies de valors crítics
        ax.axvline(C_i, color='red', linestyle='-')
        ax.axvline(C_s, color='red', linestyle='-')
        max_y = norm.pdf(mu, loc=mu, scale=sigma) # Per col·locar text
        ax.text(C_i, max_y * 0.1, r'$C_i$', ha='right', color='red')
        ax.text(C_s, max_y * 0.1, r'$C_s$', ha='left', color='red')

    def shade_acceptance_region(ax, mu, sigma, C_i, C_s, label, color='green', alpha=0.4):
        x_mid = np.linspace(C_i, C_s, 400)
        y_mid = norm.pdf(x_mid, loc=mu, scale=sigma)
        ax.fill_between(x_mid, y_mid, color=color, alpha=alpha, label=label)

    def setup_ax(ax, title=""):
        ax.set_xlabel('Valor de $\\bar{X}_n$')
        ax.set_ylabel('Densitat de Probabilitat')
        ax.set_title(title)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()

    mu_0_plot, sigma_plot = 0, 2
    C_i_plot, C_s_plot = -3, 3 # Valors crítics d'exemple
    x_range_plot = [-8, 8]

    fig, ax = plt.subplots()
    plot_normal_pdf(
        ax, mu_0_plot, sigma_plot, 
        label=r'$H_0$', color='blue', x_range=x_range_plot
    )
    shade_rejection_region(
        ax, mu_0_plot, sigma_plot, 
        C_i_plot, C_s_plot, x_range_plot
    )
    shade_acceptance_region(
        ax, mu_0_plot,sigma_plot,
        C_i_plot, C_s_plot,
        label = "Regió d'Acceptació"
    )
    setup_ax(ax, title="Distribució sota H_0 amb Regió de Rebuig (\alpha)")

    plt.show()
    return (
        C_s_plot,
        mu_0_plot,
        norm,
        np,
        plot_normal_pdf,
        plt,
        setup_ax,
        shade_acceptance_region,
        shade_rejection_region,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    Per tant, volem trobar els valor de $C_i$ i $C_s$ que determinen la regió crítica. Aquests estan explícitament determinats per $\alpha$ i per la $\sigma^2$ de la mostra. Per tant, volem trobar la relació entre la normal sobta $H_0$ i la normal estàndar, ja que allà sabem que l'àrea sota la corba ve donada per $F_Z^{-1}(\alpha/2)$ i $F_Z^{-1}(1 - \alpha/2)$.

    $$
    z_{\alpha/2} = F_Z^{-1}(\alpha/2)\\
    -z_{\alpha/2} = F_Z^{-1}(1 - \alpha/2)
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
def _(norm, plot_normal_pdf, plt, setup_ax, shade_rejection_region, z_a2):
    mu_z, sigma_z = 0, 1
    C_i_z, C_s_z = z_a2, -z_a2 
    x_range_z = [-4, 4]

    fig_z, ax_z = plt.subplots()
    plot_normal_pdf(
        ax_z, mu_z, sigma_z, 
        label=r'$N(0, 1)$', color='purple', x_range=x_range_z
    )
    shade_rejection_region(
        ax_z, mu_z, sigma_z, 
        C_i_z, C_s_z, x_range_z, label=r'Rebuig ($\alpha$)'
    )

    # Actualitzem etiquetes per Z
    max_y = norm.pdf(0)
    ax_z.text(C_i_z, max_y * 0.1, r'$z_{\alpha/2}$', ha='right', color='red')
    ax_z.text(C_s_z, max_y * 0.1, r'$-z_{\alpha/2}$', ha='left', color='red')

    setup_ax(ax_z, title="Normal Estàndard amb Regió de Rebuig")

    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Per tant, recapitulant, volem trobar $C_s$ i $C_i$ en funció d'$z_{\alpha/2}$.

    $$
    R_\alpha = \{\textbf{X} \in \mathbb{R}^n | \bar{X}_n \ge C_s \land \bar{X}_n \le C_i \} 
    $$

    Ara, manipulem la desigualtat de dins la probabilitat per a obtenir-hi $Z$, que és una normal estandaritzada. 

    $$
    \mathbb{P}(\frac{\sqrt{n}(\bar{X}_n - \mu_0)}{\sigma} \ge \frac{\sqrt{n}(C_s - \mu_0)}{\sigma} ) =
    \mathbb{P}( Z \ge z_{\alpha/2}) = 1 - \mathbb{P}( Z \le z_{\alpha/2}) = \alpha/2
    $$

    Ara usem la definició de la funció de probabilitat.

    $$
    1 - F_Z(z_{\alpha/2}) = \alpha/2 \Leftrightarrow F_Z(z_{\alpha/2}) = 1 -\alpha/2 
    $$

    I aplicant la funció inversa a les dues bandes obtenim

    $$
    z_{\alpha/2} = F^{-1}_Z (1 - \alpha/2)
    $$

    $$
    F^{-1}_Z (a) \ge \alpha/2 \space \land F^{-1}_Z (a) \le \alpha/2
    $$
    """
    )
    return


@app.cell
def _(norm, np, plt, z_a2):
    def plot_normal_cdf_with_shade(ax, mu, sigma, x_value, x_range, color):
        """
        Plots the normal CDF and shades the area from -inf to x_value on a given axis.
        Returns the cumulative probability (p-value).
        """
        # 1. Define the x range for the main CDF line
        x = np.linspace(x_range[0], x_range[1], 400)
        # 2. Calculate the CDF values
        y_cdf = norm.cdf(x, mu, sigma)
        # 3. Plot the CDF line
        ax.plot(x, y_cdf, 'blue', label='Standard Normal CDF')
    
        # 4. Define the x range for the shaded area
        x_shade = np.linspace(x_range[0], x_value, 100)
        # 5. Calculate the CDF values for the shaded area
        y_shade = norm.cdf(x_shade, mu, sigma)
    
        # Get the probability at x_value
        p_value = norm.cdf(x_value, mu, sigma)
    
        # 6. Fill the area
        ax.fill_between(x_shade, 0, y_shade, color=color, alpha=0.7, label=f'Area = {p_value:.3f}')
    
        # 7. Add a vertical line at the x_value
        ax.vlines(x_value, 0, p_value, color='red', linestyle='--', label=f'z = {x_value:.3f}')
    
        # 8. Set limits (other formatting will be done outside)
        ax.set_xlim(x_range)
        ax.set_ylim(0, 1.05)
    
        return p_value

    def format_cdf_plot(ax, z_value, p_value, z_label, p_label, title):
        """
        Aplica la personalització d'etiquetes i estil a un eix (axis) donat.
        """
        ax.set_xlabel('OCse') # Manté l'etiqueta de l'usuari
        ax.set_ylabel('Funció de Probabilitat')
        # Corregit: 'Densitat' (PDF) -> 'Distribució' (CDF)
        ax.set_title(title) 
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
        # --- Personalització de les etiquetes dels eixos ---
        # Afegim els nostres valors als eixos
        ax.set_xticks(list(ax.get_xticks()) + [z_value])
        ax.set_yticks(list(ax.get_yticks()) + [p_value])

        # Obtenim els ticks actuals
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()

        # Creem noves etiquetes
        new_xticklabels = [
            # Utilitza l'etiqueta z_label personalitzada
            f'{z_label}' if np.isclose(tick, z_value)
            else f'{tick:.1f}' 
            for tick in xticks
        ]
        new_yticklabels = [
            # Utilitza l'etiqueta p_label personalitzada
            f'{p_label}' if np.isclose(tick, p_value)
            else f'{tick:.1f}' 
            for tick in yticks
        ]

        ax.set_xticklabels(new_xticklabels, rotation=30) # Afegit rotació per evitar solapament
        ax.set_yticklabels(new_yticklabels)

        # Tornem a cridar legend() per incloure totes les etiquetes
        ax.legend(loc='upper left')

    # --- SCRIPT PRINCIPAL ---

    # 1. Definir paràmetres
    x_range_cdf = [-4, 4]
    mu_cdf, sigma_cdf = 0, 1 # Per la Normal Estàndard

    # 2. Calcular els valors z per als percentils
    p_lower = 0.025
    p_upper = 0.975
    z_1_minus_a2 = norm.ppf(p_upper, mu_cdf, sigma_cdf) # El nou valor z

    # 3. Crear la figura amb 2 subplots (1 fila, 2 columnes)
    # 'figsize' s'ha ajustat per donar espai als dos gràfics
    fig_cdf, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # --- Gràfic 1: Percentil 0.025 (El teu original) ---
    p_value_lower = plot_normal_cdf_with_shade(
        ax1, 
        mu_cdf, 
        sigma_cdf, 
        x_value=z_a2, 
        x_range=x_range_cdf,
        color='purple'
    )
    # Aplica el format personalitzat
    format_cdf_plot(
        ax1, 
        z_a2, 
        p_value_lower, 
        z_label=f'$z_{{\\alpha/2}} \\approx {z_a2:.2f}$', 
        p_label=f'$\\alpha/2 = {p_value_lower:.3f}$',
        title="Funció de Distribució (CDF) per a $\\alpha/2$"
    )

    # --- Gràfic 2: Percentil 0.975 (El nou) ---
    p_value_upper = plot_normal_cdf_with_shade(
        ax2, 
        mu_cdf, 
        sigma_cdf, 
        x_value=z_1_minus_a2, 
        x_range=x_range_cdf,
        color='orange' # Un color diferent
    )
    # Aplica el format personalitzat
    format_cdf_plot(
        ax2, 
        z_1_minus_a2, 
        p_value_upper, 
        z_label=f'$z_{{1-\\alpha/2}} \\approx {z_1_minus_a2:.2f}$', 
        p_label=f'$1-\\alpha/2 = {p_value_upper:.3f}$',
        title="Funció de Distribució (CDF) per a $1-\\alpha/2$"
    )

    # --- Finalitzar i mostrar ---
    plt.tight_layout() # Ajusta l'espaiat
    plt.savefig('cdf_plots_side_by_side.png')
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Ara que hem vist que el valor que ens queda dins la probabilitat és el valor que estem cercant $z_{\alpha/2}$ i representa el que volem, podem trobar el valor de les cotes superiors i inferirors $C_s$ i $C_i$ de la següent manera:

    $$
    Z = \frac{\sqrt{n}(\bar{X}_n - \mu_0)}{\sigma} \le z_{\alpha/2} \Rightarrow \bar{X}_n \le \mu_0 + z_{\alpha/2}\frac{\sigma}{\sqrt{n}} = C_s
    $$

    $$
    Z = \frac{\sqrt{n}(\bar{X}_n - \mu_0)}{\sigma} \le -z_{\alpha/2} \Rightarrow \bar{X}_n \ge \mu_0 - z_{\alpha/2}\frac{\sigma}{\sqrt{n}} = C_i
    $$

    L'interval ($C_i$, $C_s$) de trobar l'interval que ens defineix el contrari de la regió crítica, la regió d'acceptació! Aquesta és la decisió directa sobre l'elecció d'$\alpha$.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    L'error tipus 1 ens determina la regió crítica sota $H_0$. L'error tipus II en canvi assumeix que $H_1$ és certa, i per tant es posa en el cas $\mu \ne \mu_0$. La primera assumpció que farem és només per comoditat, que és assumir que $\exists \mu_1 \ne \mu_0$ la qual determina la hipòtesi alternativa. Aquest paràmetre és absolutament fictici, és a dir, no sabem mai quin valor pren realment, però sabem segur que estarà definit i que serà un valor concret $\mu_1 \in \mathbb{R}$. Per això, prenc unilateralment la llibertat de denotar $\beta$ com una funció $\beta(\mu_1)$

    A més a més, i només per usos didàctics i no repetir els càlculs, que $\mu_1 > \mu_0$. La distribució és simètria, per tant això no comporta cap pèrdua de generalitat.

    L'error tipus II és

    $$ \beta = \mathbb{P}(\mathbf{X} \notin \mathbb{R}^n | \mu = \mu_1) = \mathbb{P}(\bar{X}_n \le C_s)$$

    L'error tipus II ens està dient que, si saps que $H_1$ és certa, la probabilitat d'equivocarte és que la mitjana mostral caigui fora de la regió crítica superior (recordem que hem dit que $\mu_1 > \mu_0$). Per exemple, a la imatge següent seria el cas en que s'accepta la hipòtesi nul·la!
    """
    )
    return


@app.cell
def _(
    C_s_plot,
    mu_0_plot,
    norm,
    plot_normal_pdf,
    plt,
    setup_ax,
    shade_acceptance_region,
):
    # mu_0_plot, sigma_0_plot = 0, 2 # ja estan definits a dalt
    sigma_0_plot = 2
    mu_1_plot, sigma_1_plot = 2, 2 # Fem sigmes iguals i H1 més lluny per claredat
    # C_i_plot, C_s_plot = 3, 3 # ja estan definints a dalt!
    # A la pràctica C_i seria ~ -3. Ajustem per a la imatge
    C_i_plot_real = -3 
    x_range_plot1 = [-8, 14]

    fig_h1, ax_h1 = plt.subplots()

    # 1. Dibuixar H0 (amb línia discontínua)
    plot_normal_pdf(
        ax_h1, mu_0_plot, sigma_0_plot, 
        label=r'$H_0$', color='blue', linestyle='--', x_range=x_range_plot1
    )

    # 2. Dibuixar H1
    plot_normal_pdf(
        ax_h1, mu_1_plot, sigma_1_plot, 
        label=r'$H_1$', color='green', x_range=x_range_plot1
    )

    # 3. Omplir l'error tipus II (àrea sota H1 a la regió d'acceptació de H0)
    shade_acceptance_region(
        ax_h1, 
        mu_1_plot, 
        sigma_1_plot, 
        C_i_plot_real, 
        C_s_plot, 
        label=r'Error Tipus II ($\beta$)',
        color='red',
        alpha=0.4
    )

    # 4. Dibuixar les línies crítiques
    ax_h1.axvline(C_i_plot_real, color='red', linestyle='-')
    ax_h1.axvline(C_s_plot, color='red', linestyle='-')
    max_y1 = norm.pdf(mu_0_plot, loc=mu_0_plot, scale=sigma_0_plot)
    ax_h1.text(C_i_plot_real, max_y1 * 0.1, r'$C_i$', ha='right', color='red')
    ax_h1.text(C_s_plot, max_y1 * 0.1, r'$C_s$', ha='left', color='red')

    setup_ax(ax_h1, title="Distribució sota $H_0$ i $H_1$ amb Error Tipus II")

    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Per a trobar-ho, operem la desigualtat de dins de la probabilitat fins que estigui normalitzada, tal que així

    $$
    \mathbb{P}\lparen\ \frac{\sqrt{n}(\bar{X}_n - \mu_1)}{\sigma} \le  \frac{\sqrt{n}(C_s - \mu_1)}{\sigma} \rparen =
    \mathbb{P}(Z \le z_\beta) = \beta(\mu_1)
    $$

    I això és la definició de funció de probabilitat $F_Z(x) = P(Z \le x)$

    $$
    F_Z(z_\beta) =\beta(\mu_1) \Leftrightarrow z_\beta = F^{-1}_Z(\beta) = \frac{\sqrt{n}(C_s - \mu_1)}{\sigma} 
    $$

    I per tant hem trobat la relació entre $z_\beta$ i el límit $C_s$.

    És a dir, sota la hipòtesi $H_1$ certa no rebutjarem $H_0$ (definició d'error tipus II), veiem aïllant la part de sobre 

    $$
    C_s = \mu_1 + z_\beta \frac{\sigma}{\sqrt{n}} \le \bar{X}_n 
    $$
    """
    )
    return


@app.cell
def _(beta, norm):
    z_beta = norm.ppf(beta)
    z_beta
    return (z_beta,)


@app.cell
def _(mo):
    mo.md(
        r"""
    Ara, recapitulem. Hem aconseguit expressar $C_s$ de dues maneres diferents.
    + Sota $H_0$: $\bar{X}_n \le \mu_0 + z_{\alpha/2}\frac{\sigma}{\sqrt{n}} = C_s$
    + Sota $H_1$: $C_s = \mu_1 + z_\beta \frac{\sigma}{\sqrt{n}} \ge \bar{X}_n$

    I per tant, podem igualar respecte la cota $C_s$ i aïllem $n$

    $$
    n = \frac{\sigma^2 (z_{\alpha/2} - z_{1-\beta})^2}{(\mu_1 - \mu_0)^2} = \frac{\sigma^2 (z_{\alpha/2} + z_{1-\beta})^2}{d^2} 
    $$

    On recordem que $d$ era el tamany de l'effecte. Aquesta fòrmula és fantàstica, perque relaciona totes les quantitats que s'han hagut d'escollit a priori pel test. Anem-la a calcular pel nostre cas.

    Com que $d = \mu_0 - \mu_1 > 2$ i $\mu_0 = 0$, això vol dir que el mínim valor de la hipòtesi alternativa a considerar serà $\mu_1$, ja que és el mínim del que volem detectar. Si detectem quantitats més grans, doncs tampoc passa res
    """
    )
    return


@app.cell
def _(z_a2, z_beta):
    mu_0 = 0
    d = (mu_0-2)**2
    sigma = 5 
    n = (sigma*(z_a2 - z_beta)**2)/d
    print(n)
    return mu_0, n, sigma


@app.cell
def _(mo):
    mo.md(
        r"""
    Per tant, només amb només 10 mostres serem capaços de garantir resultats amb els paràmetres demanats.

    ## 7. Valors crítics

    Ara hem de trobar els valors crítics de $C_s$ i $C_i$. Ja n'hem trobat les expressions, així que ho fem directament. Cal notar que aquest valor el determina $\alpha$, i amb això vull dir que sota la hipòtesi alternativa no sabem el valor real que pot prendre $\mu_1$, per tant hem de trobar la cota que hem trobat sota $H_0$.
    """
    )
    return


@app.cell
def _(mu_0, n, sigma, z_a2):
    from math import sqrt, ceil

    C_s = mu_0 + z_a2*sigma/sqrt(ceil(n))
    C_s
    return


@app.cell
def _(mo):
    mo.md(r"""I amb això completem els passos a priori!!!""")
    return


if __name__ == "__main__":
    app.run()
