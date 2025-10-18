# Simulació Assigment 1: Proposta 1

L'entregable per a la primera part del curs de "Simulació" ha de consistir en un treball bastat en un dels següents temes desenvolupats a classe:

```
Option 1, 2: Resampling techniques(Permutational test or Bootstrap Confidence Intervals). Case of studies. 
Option 3: Simulation studies in Statistics. Case of Study
```

La meva primera proposta es basa en *Option 3*, ja que considero que és la que obre la porta a un treball més versàtil i no tant restringit com la resta. A falta de detallls en què ha de contenir aquesta opció de treball, la meva idea en terms generals es descriu a continuació.

El *Case of Study* és la introducció dels *e-values* com a eina per al contrast d'hipòtesi i la seva comparació ambs els *p-values*. Els *e-values* són una línia d'investigació molt activa actualment ja que corregeixen moltes de les limitacions dels p-valors que s'han anat identificat durant els anys en totes les branques de la literatura acadèmica que els apliquen, tot i ser "statistical cousins" d'aquests mateixos. En particular, destacar les següents propietats (descrites a [1]):

1.  Els *p-values* i els *e-values* com a objectes existeixen exactament sota les mateixes condicions, sigui quina sigui l'estructura del contrast d'hipòtesi. A més a més, un pot transformar d'un a l'altre en un procés que es coneix com a *calibració*.

2. Els *e-values* són flexibles en el disseny i continuació seqüencial d'un experiment: així com la validesa dels *p-valors* només està garantida sota un disseny rigorós **a priori** de l'experiment, els e-valors obren la porta a la flexibilitat en la experimentació, permetent aturar o continuar l'exeperiment en qualsevol instant de temps, tot guarantint el control de l'error de Tipus I; o el que és encara més destacat, permeten modificar el nivell de significació $\alpha$ en base a les dades prèviament observades. 
Observem que tot es duu a terme en una "sequential fashion". 

El treball mantindrà l'estructura *ADMEP* descrita a [2] i trindrà un component teòric que no entrarà en molt detall en els fonaments matemàtics. La idea serioa enumerar una serie de propietats que volem verificar mitjaçant simulació de Monte Carlo.

Proposo que, un cop estiguem alineats amb el projecte, elaborem una proposta formal del projecte i l'enviem al professor de l'assignatura per assegurar-nos que seria una treball vàlid.

## Referències

[1] Morris, T. P., White, I. R., & Crowther, M. J. (2019). Using simulation studies to evaluate statistical methods. Statistics in Medicine, 38(11), 2074-2102. https://doi.org/10.1002/sim.8086

[2] Ramdas, A., & Wang, R. (2025). Hypothesis testing with e-values. Foundations and Trends® in Statistics, 1(1-2), 1-390. https://doi.org/10.1561/3600000002



