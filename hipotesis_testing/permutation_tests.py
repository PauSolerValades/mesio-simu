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

    Farem el mateix experiment dues vegades, seguint la filosofia de Fisher i la de Neyman-Pearson.
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

    NOTA: el test de permutacions NO admet una hipòtesi alternativa (és a dir, és de l'escola de Fisher) ja que és impossible permutar la hipòtesi alternativa per a obtenir el llindar. PREGUNTAR: fer servir $\alpha$ i $H_1$ aquí seria INCORRECTE.

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
    return


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

    Seguirem utilitzant un ANOVA entre els tres grups.

    ## 2. Hipòtesi Nul·la

    Utilitzem la mateixa hipòtesi nul·la

    $$H_0: \mu_A = \mu_B = \mu_C$$

    ## 3. Càlcul del $p$-valor

    Al utilitzar el mateix estadístic, ja tenim el còmput mostral. El que canvia ara són les permutacions.

    Al primer cas d'estudi, permutavem l'etiqueta A, B o C segons el grup. Ara però, no té sentit permutar etiquetes de persones diferents, així que per a cada individu només tenim $3!$ posicions. Sabent el nombre de mostres obtenim

    $$R_3_^8 = (3!)^8$$
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
    mo.md(r"""El nombre és adequat per un test de MonteCarlo, però és prou petit com per fer un test exacte si ho preferissim.""")
    return


@app.cell
def _(df1, np):
    from itertools import permutations, product

    combinations_per_individual = np.array([[*permutations(row)] for row in df1.iter_rows()])

    # funció que implementa el producte cartesià recursiu
    def get_all_combinations_recursive(list_of_lists):
        # Base Case: If there's only one list left, return its items.
        # Each item needs to be wrapped in a list so it can be combined later.
        if len(list_of_lists) == 1:
            return [[item] for item in list_of_lists[0]]

        # Recursive Step
        # 1. Get all combinations from the REST of the lists.
        combinations_from_rest = get_all_combinations_recursive(list_of_lists[1:])
    
        # 2. Get the choices from the CURRENT list.
        current_choices = list_of_lists[0]
    
        all_combinations = []
        for choice in current_choices:
            for combo in combinations_from_rest:
                all_combinations.append([choice] + combo)
            
        return all_combinations

    #all_combinations = get_all_combinations_recursive(combinations_per_individual)
    # fent servir l'unpakcing operator tot va finíssim
    all_combinations_iterator = np.array(list(product(*combinations_per_individual)))
    print(len(all_combinations_iterator))
    return (all_combinations_iterator,)


@app.cell
def _(all_combinations_iterator, np, stats):
    all_combinations_colums = np.array([array.reshape(array.shape[1], array.shape[0]) for array in all_combinations_iterator])

    f_statistics_exact = [
        stats.f_oneway(A, B, C).statistic
        for A, B, C in all_combinations_colums
    ]

    print(len(f_statistics_exact))
    return (all_combinations_colums,)


@app.cell
def _(mo):
    mo.md(r"""Per a fer-ho amb Monte-Carlo, al ja haver calculat totes les permutacions possibles, només hem d'agafar una mostra de `n_sample` aleatòriament.""")
    return


@app.cell
def _(all_combinations_colums, n_resamples, np, stats):
    sample_idx = np.random.choice([i for i in range(len(all_combinations_colums))], size=n_resamples)

    f_statistics_mc = [
        stats.f_oneway(A, B, C).statistic
        for (A, B, C) in all_combinations_colums[sample_idx] 
    ]

    print(len(f_statistics_mc))
    return (f_statistics_mc,)


@app.cell
def _(f_sample, f_statistics_mc, n_resamples):
    extreme_observations_2 = (sum((f_sample <= f_statistic for f_statistic in f_statistics_mc)) + 1)
    pvalue_2 = extreme_observations_2 / (n_resamples +1)
    print(f"El p-valor és de {pvalue_2:.3e}. Hi ha {extreme_observations_2} observacions sobre F mostral")
    return


@app.cell
def _(pl):
    df3= pl.read_csv("casestudy3.csv")

    df3
    return


if __name__ == "__main__":
    app.run()
