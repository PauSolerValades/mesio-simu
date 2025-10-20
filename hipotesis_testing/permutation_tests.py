import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import scipy.stats as stats
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, pl, plt, stats


@app.cell
def _(mo):
    mo.md(
        r"""
    # Cas d'estudi 1: Tast de vins

    Tenim tres grups de persones, A, B, i C. Cada grup ha fet un tast de vi amb el mateix vi, però amb circumstàncies diferents. El grup A ha estat un tast prèmium, el grup B ha estat un tast normal i el C un tast deixat.

    L'objectiu del test d'hipòtesi seria veure si l'entorn ha tingut algun tipus d'influència sobre la percepció del vi dels participants de l'estudi.
    """
    )
    return


@app.cell
def _(pl):
    df1 = pl.read_csv("casestudy1.csv")
    df1
    return (df1,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 1. Test

    Seleccionem un test ANOVA d'anàlisi de la variança entre els tres grups.

    ## 2. Hipòtesi Nula

    Escollim l'hipòtesi nula com la més conservadora que hem d'intentar rebutjar. Fem per tant

    $$
    H_0: \mu_A = \mu_B = \mu_C \\
    $$

    El test d'hipòtesi ens intentarà rebutjar o no rebutjar aquesta hipòtesi.

    NOTA: el test de permutacions NO admet una hipòtesi alternativa (és a dir, és de l'escola de Fisher) ja que és impossible permutar la hipòtesi alternativa per a obtenir el llindar. PREGUNTAR: fer servir $\alpha$ i $H_1$ aquí seria INCORRECTE

    ## 3. Càlcul del $p$-valor

    Per a fer-ho emprem un test de permutacions. Els passos dins del test de permutacions seran els següents:

    ### 3.1 Estadístic

    Escollim l'$F$ estadístic obtingut de l'anova.

    ### 3.2 Trobem l'estadístic mostral

    $F_M$ serà el valor de l'$F$ estadístic de la mostra actual, és a dir, realitzar un ANOVA sobre els tres grups A, B i C.
    """
    )
    return


@app.cell
def _(df1, stats):
    A = df1["A"].to_numpy()
    B = df1["B"].to_numpy()
    C = df1["C"].to_numpy()

    f_sample = stats.f_oneway(A, B, C).statistic
    print(f"L'estadístic F de la mostra és {f_sample}")
    return A, B, C, f_sample


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Nombre de permutacions
    Un cop sabem l'ANOVA de la mostra, hem de decidir quin mètode usar:
    - Exacte: calcularem l'ANOVA de totes les permutacions
    - MonteCarlo: calcularem l'ANOVA de les permutacions informatives mitjançant el mètode de MonteCarlo.

    Per a saber-ho, depen exclusivament del nombre de permutacions possibles que té aquest problema.

    Tenim tres grups de persones completament diferents, això ens dona un total de 24 observacions. Aquestes persones es poden partir en tres grups, A, B, C iguals, i és clar que la permutació que volem realitzar és moure una mostra del grup A a un altre grup i observar com canvia l'ANOVA. Això és clarament 

    $$R_{8,8,8}^{24} = \binom{24}{8} \binom{16}{8} \binom{8}{8}$$
    """
    )
    return


@app.cell
def _():
    from scipy.special import comb

    n_perm = comb(N=24, k=8, exact=True)*comb(N=16, k=8, exact=True)
    print(f"Hi ha {n_perm} ({n_perm:.2e}) permutacions possibles")
    return (n_perm,)


@app.cell
def _(mo):
    mo.md(
        r"""
    Degut a ser un nombre molt gran, el mètode de MonteCarlo és la única opció factible.

    ### 3.4 Trobar la distribució sota $H_0$

    Utilitzant un subconjunt de permutacions informatives, generem la distribució sota la hipòtesi nula.
    """
    )
    return


@app.cell
def _(A, B, C, np, stats):
    n_resamples = 9999 
    total_sample = np.concatenate((A, B, C))

    perms = [np.random.permutation(total_sample) for _ in range(0,n_resamples)]

    f_statistics = [
        stats.f_oneway(element[0:8], element[8:16], element[16:24]).statistic
        for element in perms
    ]
    return f_statistics, n_resamples


@app.cell
def _(f_sample, f_statistics, plt):
    plt.figure()
    plt.hist(f_statistics, bins=20)
    plt.axvline(x=f_sample, color='r') 
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 3.5 Càlcul del $p$-valor

    Aproximem el $p$-valor mitjançant MonteCarlo dividint el nombre d'elements per sota del valor de l'estadístic mostral entre el nombre de sampleig de hem volgut utilitzar, és a dir amb la fòrmula.

    $$
    p \approx \frac{\sum_{i=1}^n I(F_M \le F_i) + 1}{n + 1}
    $$

    on $I$ és la funció indicatriu. El $+1$ s'ha de col·locar com a garantia a que sempre hi haurà un element major que la distribució mostral, ja que en cas contrari, ens quedaria un $p$-valor de 0, valor estadísticament sense sen
    """
    )
    return


@app.cell
def _(f_sample, f_statistics, n_resamples):
    extreme_observations = (sum((f_sample <= f_statistic for f_statistic in f_statistics)) + 1)
    pvalue = extreme_observations / (n_resamples +1)
    print(f"El p-valor és de {pvalue:.3e}. Hi ha {extreme_observations} observacions sobre F mostral")
    return


@app.cell
def _(C):
    del C
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Com es pot veure, el $p$-valor (a menys que s'hagi tingut molta mala sort...) és bastant petit.

    ## 4. Significancia Estadística

    El $p$-valor és baix, és a dir, que la simulació sota $H_0$ són poc probables de succeir com a fenòmen aleatòri, ja que el valor de $p$ és significant.

    ## 5. Conclusió

    Per tant, degut a la significancia del $p$-valor, es pot dir que hi ha una forta evidència en que l'observació d'aquest esdeveniment degut a un efecte de l'aleatorietat és molt poc probable, i per tant les dades es poden prendre com una evidència clara i de gran magnitud contra la hipòtesi nul·la; per tant la presentació dels vins ha tingut effecte sobre la percepció dels individus encara que sigui el mateix vi.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Cas d'estudi 2: Tast de vins 2. 

    Assumim ara que en comptes de 24 mostres, les mateixes 8 persones fan el test A, B, i C. Un cop això és així, podem seguir demostrant la mateixa hipòtesi?

    ## 1. Test

    Seguirem utilitzant un ANOVA entre els tres grups, però la metodologia de l'experiment seguirà el mètode de [Randomized Block Design](https://en.wikipedia.org/wiki/Generalized_randomized_block_design). S'utilitzen per mirar la interacció entre [blocs](https://en.wikipedia.org/wiki/Blocking_(statistics)) (agrupaments de dades similars en una o diverses caractererístiques) contra tractaments. Cada tractament es replica com a mínim dues vegades per bloc, permetent l'estimació dels termes d'interacció d'un _model lineal_. La idea enginyosa és que els blocs s'agrupen amb la _variable de no interès_ (nuisance parameter), és a dir, la que creiem que no explicarà la varianca de la relació i és la qual haurem de permutar per comprovar si les altres dues tenen realment relació.

    Hem de definir per tant els blocs, els termes interactius i el model lineal:
    + Objectiu: quina relació volem modelitzar? la relació entre la variable `tast` i la variable `valoració`.
    + Blocs: La variable de no interès son els `subjecte`, perque cadascun ha participat en els tres grups, i la relació real està en els grups, no amb quin individu ha donat la valoració. Per tant, els blocks seran `(valoracio_A_i, valoracio_B_i, valoracio_B_i)` on `i=1..8` nombre de subjectes diferents.
    + Randomització: dins de cada bloc randomitzem les valoracions (això és el $3!$ que s'ha mencionat abans al text.) 

    ## 2. Hipòtesi Nul·la

    Utilitzem la mateixa hipòtesi nul·la

    $$H_0: \mu_A = \mu_B = \mu_C$$

    ## 3. Càlcul del $p$-valor

    Al utilitzar el mateix estadístic, ja tenim el còmput mostral. El que canvia ara són les permutacions.

    Al primer cas d'estudi, permutavem l'etiqueta A, B o C segons el grup. Ara però, no té sentit permutar etiquetes de persones diferents, així que per a cada individu només tenim $3!$ posicions. Sabent el nombre de mostres obtenim

    $$R_3^8 = (3!)^8$$
    """
    )
    return


@app.cell
def _():
    from math import factorial

    num_perm_total = factorial(3)**8
    print(f"Hi ha un total de {num_perm_total} ({num_perm_total:.2e}) combinacions")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    El nombre és adequat per un test de MonteCarlo, tot i que un exacte (si tens 15 minuts) també és possible d'utilitzar. Per a fer el codi lleugerament més llegible i eficient, primer he generat totes les possibles iteracions mitjançant un generador de python a `all_combinations_iterator`, i d'allà n'extrec una mostra per a fer montecarlo.

    Per a calcular, primer convertim el dataset de tres columnes A, B i C en un dataset per facilitat.
    """
    )
    return


@app.cell
def _(df1):
    df2 = (
        df1 
        .with_row_index(name="Subjecte", offset=1) # Adds index col 'Subjecte' starting at 1
        .unpivot(
            index="Subjecte", 
            on=["A", "B", "C"],
            variable_name="Tast", 
            value_name="Valoracio"
        )
        .sort("Subjecte")
    )
    df2 = df2.to_pandas()
    df2
    return (df2,)


@app.cell
def _(mo):
    mo.md(r"""Un cop amb les dades, creem un model lineal que relacioni Valoracio amb Tast i Grup amb les dades actuals i en calculem l'$F$ estadístic.""")
    return


@app.cell
def _(df2):
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from time import time

    def f_statistic_sample(df2):
        model_obs = ols('Valoracio ~ C(Subjecte) + C(Tast)', data=df2).fit()
        anova_table_obs = sm.stats.anova_lm(model_obs, type=2)

        return anova_table_obs

    t1 = time()
    anova_table_obs = f_statistic_sample(df2)
    t2 = time()

    print(anova_table_obs)
    f_obs = anova_table_obs.loc['C(Tast)', 'F']
    print(f"\nF-estatistic mostral (f.obs): {f_obs:.4f} in {t2-t1}")
    return f_obs, ols, sm, time


@app.cell
def _(df2, sm, time):
    from sklearn.preprocessing import OneHotEncoder

    def f_statistic_sample_manual(df): 
        y_obs = df['Valoracio'].to_numpy()
        subjects = df[['Subjecte']].to_numpy()  # Needs to be 2D for the encoder
        tast_and_subj = df[['Subjecte', 'Tast']].to_numpy()

        # Reduced Model (X_reduced): Valoracio ~ Subjecte
        encoder_subj = OneHotEncoder(drop='first', sparse_output=False)
        X_subj_dummies = encoder_subj.fit_transform(subjects)
        X_reduced = sm.add_constant(X_subj_dummies, prepend=True) # Adds the intercept

        # Full Model (X_full): Valoracio ~ Subjecte + Tast
        encoder_full = OneHotEncoder(drop='first', sparse_output=False)
        X_full_dummies = encoder_full.fit_transform(tast_and_subj)
        X_full = sm.add_constant(X_full_dummies, prepend=True) # Adds the intercept

        df_diff = X_full.shape[1] - X_reduced.shape[1]  # DFs for the 'Tast' factor (2)
        df_full = len(y_obs) - X_full.shape[1]         # Residual DFs for the full model (14)

        # Fit both models on the original, un-shuffled data
        model_reduced_obs = sm.OLS(y_obs, X_reduced).fit()
        model_full_obs = sm.OLS(y_obs, X_full).fit()

        # Get the Residual Sum of Squares (RSS) for each
        rss_r_obs = model_reduced_obs.ssr
        rss_f_obs = model_full_obs.ssr

        # Manually calculate the F-statistic
        f_obs = ((rss_r_obs - rss_f_obs) / df_diff) / (rss_f_obs / df_full)
        return f_obs

    t_s = time()
    f_obs_manual = f_statistic_sample_manual(df2)
    t_e = time()

    print(f"Observed F-statistic (f.obs): {f_obs_manual:.4f} in {t_e-t_s:.4f}")
    return (OneHotEncoder,)


@app.cell
def _(mo):
    mo.md(r"""Permutem per a fer montecarlo: ens guardem els $F$-estadístics d'un subconjunt de premutacions informatives per a fer la distribució de l'estadístic sota la hipòtesi nu""")
    return


@app.cell
def _(OneHotEncoder, df2, f_obs, n_perm, n_resamples, np, sm, time):
    f_permutations = [] # Pre-allocate a numpy array for speed
    subjects = df2[['Subjecte']].to_numpy()  # Needs to be 2D for the encoder
    tast_and_subj = df2[['Subjecte', 'Tast']].to_numpy()
    # Get subject groups for efficient shuffling
    subject_groups = df2['Subjecte'].to_numpy()
    original_indices = np.arange(len(df2))
    y_obs = df2['Valoracio'].to_numpy()
    encoder_subj = OneHotEncoder(drop='first', sparse_output=False)
    X_subj_dummies = encoder_subj.fit_transform(subjects)
    X_reduced = sm.add_constant(X_subj_dummies, prepend=True) # Adds the intercept_

    encoder_full = OneHotEncoder(drop='first', sparse_output=False)
    X_full_dummies = encoder_full.fit_transform(tast_and_subj)
    X_full = sm.add_constant(X_full_dummies, prepend=True) # Adds the intercept

    # --- Calculate degrees of freedom (these are constant) ---
    df_diff = X_full.shape[1] - X_reduced.shape[1]  # DFs for the 'Tast' factor (2)
    df_full = len(y_obs) - X_full.shape[1]         # Residual DFs for the full model (14)

    print(f"\nRunning {n_resamples} permutations...")
    start_time = time()

    for i in range(n_resamples):
        # Permute the 'Valoracio' values by shuffling indices within each subject group
        permuted_indices = np.concatenate(
            [np.random.permutation(original_indices[subject_groups == g]) 
             for g in np.unique(subject_groups)]
        )
        y_perm = y_obs[permuted_indices]
    
        # Fit the two models using the permuted y and pre-built matrices
        # This is now just fast numpy math, no slow formula parsing
        rss_r_perm = sm.OLS(y_perm, X_reduced).fit().ssr
        rss_f_perm = sm.OLS(y_perm, X_full).fit().ssr
    
        # Calculate F-stat for this permutation
        f_p = ((rss_r_perm - rss_f_perm) / df_diff) / (rss_f_perm / df_full)
        f_permutations.append(f_p)

    end_time = time()
    print(f"Done. Loop took {end_time - start_time:.2f} seconds.")


    # --- 4. Calculate Final p-value ---
    n_greater = np.sum(f_permutations >= f_obs)
    p_value = (n_greater + 1) / (n_perm + 1)

    print("\n--- Results ---")
    print(f"Permuted F-stats >= observed: {n_greater} out of {n_perm}")
    print(f"Monte Carlo p-value: {p_value:.4f}")
    return


@app.cell
def _(df2, n_resamples, np, ols, pl, sm):
    def permute_and_fit(df: pl.DataFrame):
        df_perm_pd = df.copy()

        df_perm_pd['Valoracio'] = df_perm_pd.groupby('Subjecte')['Valoracio'] \
                                          .transform(np.random.permutation)

        model_perm = ols('Valoracio ~ C(Subjecte) + C(Tast)', data=df_perm_pd).fit()
        anova_table_perm = sm.stats.anova_lm(model_perm, type=2)
        return anova_table_perm.loc['C(Tast)', 'F']

    f_perm_comp = [
        permute_and_fit(df2) for _ in range(n_resamples)    
    ]
    f_perm_comp
    return


@app.cell
def _():
    """f_permutations = np.array(f_permutations)
    n_greater = np.sum(f_permutations >= f_obs)

    p_value = (n_greater + 1) / (n_perms + 1)

    print(f"Observed F-statistic: {f_obs:.4f}")
    print(f"Permuted F-stats >= observed: {n_greater} out of {n_perms}")
    print(f"Monte Carlo p-value: {p_value:.4f}")"""
    return


if __name__ == "__main__":
    app.run()
