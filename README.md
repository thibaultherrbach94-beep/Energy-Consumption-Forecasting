# Energy Consumption Forecasting

## Overview

Ce projet développe un pipeline complet de prévision de consommation électrique combinant :

- statistique en grande dimension ;
- machine learning ;
- deep learning ;
- séries temporelles ;
- quantification de l'incertitude ;
- analyse des événements extrêmes ;
- analyse de drift temporel.

L'objectif principal est de construire un modèle capable de prédire la consommation électrique à partir de l'historique de consommation, de variables calendaires et de données météorologiques.

Une attention particulière est portée à la **validation temporelle** et à l'évaluation sur des données futures totalement indépendantes.

---

# Final Model

Le modèle final est un pipeline hybride composé de quatre niveaux :

```text
Historical consumption
        │
        ▼
High-dimensional feature engineering
        │
        ▼
Ridge Regression
        │
        ▼
Residual MLP
        │
        ▼
Patch Transformer
        │
        ▼
Refined Event Expert
        │
        ▼
Final prediction
```

Le pipeline combine ainsi plusieurs types de modèles ayant chacun un rôle différent.

### Ridge Regression

Le Ridge modélise la structure linéaire principale de la consommation à partir de plusieurs centaines de variables temporelles.

### Residual MLP

Un réseau de neurones apprend ensuite à prédire les résidus laissés par le Ridge.

La prédiction devient :

$$
\hat{y}_{MLP}
=
\hat{y}_{Ridge}
+
\hat{r}_{MLP}
$$

### Patch Transformer

Le Patch Transformer exploite la structure séquentielle des résidus.

Il utilise des fenêtres temporelles de :

$$
336
$$

observations, correspondant à environ une semaine de données à fréquence 30 minutes.

### Refined Event Expert

Un expert spécialisé est utilisé pour certains événements calendaires :

- jours fériés ;
- périodes de fin d'année ;
- lendemain de jour férié.

L'expert n'est activé que sur une petite partie des observations.

---

# Feature Engineering

Le modèle utilise **429 variables explicatives**.

Elles comprennent notamment :

### Historical lags

```text
lag_1
lag_2
...
lag_336
```

ainsi que plusieurs lags saisonniers plus longs.

### Rolling statistics

Exemples :

```text
rolling_mean_6
rolling_mean_48
rolling_mean_336

rolling_std_6
rolling_std_48
rolling_std_336
```

### Trend features

Exemples :

```text
diff_1
diff_day
diff_week

trend_3h
trend_6h
trend_12h
```

### Calendar features

Les cycles temporels sont encodés à l'aide de transformations trigonométriques :

$$
\sin\left(\frac{2\pi h}{24}\right)
$$

et :

$$
\cos\left(\frac{2\pi h}{24}\right)
$$

pour l'heure, le jour de la semaine et le mois.

### Weather features

Le pipeline utilise également plusieurs variables météorologiques :

```text
temp_lag_1h
temp_lag_24h
temp_lag_7d

temp_mean_24h
temp_mean_7d

heating_degree
cooling_degree
```

ainsi que plusieurs interactions non linéaires.

---

# Experimental Protocol

Le projet utilise une séparation temporelle stricte.

Les données 2026 constituent un **jeu de test final indépendant**.

Elles n'ont pas été utilisées pour :

- entraîner les modèles ;
- sélectionner les architectures ;
- optimiser les hyperparamètres ;
- sélectionner les coefficients de combinaison ;
- choisir les règles d'activation des experts.

Le modèle complet est gelé avant l'évaluation finale 2026.

Cette procédure évite d'optimiser les performances directement sur le jeu de test.

---

# Independent 2026 Test

La période finale utilisée est :

```text
2026-01-15
to
2026-09-26
```

Nombre d'observations :

```text
12,239
```

## Final results

| Model | RMSE (MW) | MAE (MW) | MAPE | NRMSE | R² |
|---|---:|---:|---:|---:|---:|
| Ridge | 285.53 | 208.88 | 0.4480% | 0.5899% | 0.9990 |
| Ridge + MLP | 263.31 | 193.06 | 0.4134% | 0.5440% | 0.9991 |
| Ridge + MLP + Patch Transformer | 256.13 | 186.30 | 0.3994% | 0.5292% | 0.9992 |
| **Final model** | **255.91** | **186.11** | **0.3988%** | **0.5287%** | **0.9992** |

The final model improves the RMSE by:

$$
\boxed{10.38\%}
$$

compared with the Ridge baseline.

The mean absolute percentage error is:

$$
\boxed{MAPE = 0.3988\%}
$$

---

# Temporal Generalization

Pour évaluer la stabilité temporelle du modèle, 2025 et 2026 ont été comparés sur exactement la même fenêtre :

```text
15 January
to
26 September
```

Chaque période contient :

```text
12,239 observations
```

Les résultats montrent :

| Metric | 2025 | 2026 | Change |
|---|---:|---:|---:|
| RMSE | 166.69 | 255.91 | +53.53% |
| MAE | 122.17 | 186.11 | +52.33% |
| MAPE | 0.2565% | 0.3988% | +55.48% |
| P99 absolute error | 487.28 | 827.68 | +69.86% |

Le modèle présente donc une **dégradation temporelle significative** en 2026.

---

# Error Analysis

L'analyse des erreurs montre que la dégradation n'est pas uniforme.

Les régimes les plus difficiles sont notamment :

- juillet ;
- août ;
- septembre ;
- les températures élevées ;
- le milieu de journée ;
- les week-ends ;
- certaines transitions calendaires.

Le mois de septembre présente notamment :

$$
RMSE \approx 343\text{ MW}
$$

Les températures supérieures à environ 24 °C présentent un RMSE proche de :

$$
304\text{ MW}
$$

---

# Drift Analysis

Une analyse de drift a été réalisée afin de comprendre la dégradation temporelle.

## Covariate Shift

Sur les **429 features** :

```text
413 Low drift
13 Moderate drift
3 Large drift
```

Seulement environ :

$$
3.73\%
$$

des variables présentent donc un drift modéré ou important.

Les principaux changements concernent notamment :

```text
temp_mean_7d
temp_mean_24h
rolling_mean_336
rolling_std_336
```

La cible reste relativement stable entre les deux périodes.

---

# Conditional Drift

Les erreurs ont également été comparées à l'intérieur de régimes similaires.

Les performances 2026 sont généralement moins bonnes dans les régimes étudiés séparément.

Cependant, un **matching multivarié** entre observations 2025 et 2026 a ensuite été réalisé.

Pour les mutual nearest neighbors :

```text
122 matched pairs
```

les résultats sont :

$$
RMSE_{2025}
=
269.13\text{ MW}
$$

$$
RMSE_{2026}
=
230.20\text{ MW}
$$

La comparaison multivariée ne confirme donc pas une dégradation systématique à caractéristiques similaires.

La conclusion finale est donc :

```text
Strong temporal generalization degradation
+
Localized covariate shift
+
Conditional drift signals
+
No sufficient evidence of confirmed concept drift
```

---

# Final Figures

Les figures principales du projet sont disponibles dans :

```text
figures/
```

### Model progression

```text
figures/01_model_progression_2026.png
```

### 2025 vs 2026 comparison

```text
figures/02_matched_2025_vs_2026.png
```

### Monthly RMSE

```text
figures/03_monthly_rmse_2026.png
```

### Temperature regimes

```text
figures/04_temperature_rmse_2026.png
```

### Extreme errors

```text
figures/05_top1pct_errors_by_month.png
```

---

# Project Structure

```text
Projet energie/
│
├── data_processed/
│
├── models/
│
├── figures/
│
├── scripts/
│
├── notebooks/
│
└── README.md
```

Les objets intermédiaires et modèles coûteux à recalculer sont sauvegardés avec `joblib` ou `torch.save`.

---

# Methods Used

Le projet couvre notamment :

- Linear Regression
- Ridge
- Lasso
- Elastic Net
- PCA
- Hyperparameter tuning
- Ensembling
- Stacking
- Out-of-fold predictions
- Residual learning
- Neural networks
- Patch Transformer
- Weather feature engineering
- Conformal prediction
- Extreme-value analysis
- Monte Carlo methods
- Time-series diagnostics
- Drift detection
- Nearest-neighbor matching
- Independent temporal validation

---

# Technologies

```text
Python
NumPy
Pandas
SciPy
scikit-learn
PyTorch
Matplotlib
Joblib
Jupyter
```

---

# Key Result

The main result of the project is a frozen hybrid forecasting pipeline evaluated on a completely independent future dataset.

On the 2026 test set:

$$
\boxed{RMSE = 255.91\text{ MW}}
$$

$$
\boxed{MAPE = 0.3988\%}
$$

with a:

$$
\boxed{10.38\%}
$$

reduction in RMSE compared with the Ridge baseline.

The project also demonstrates the importance of evaluating time-series models on genuinely future data and analyzing how their performance changes under temporal distribution shift.
