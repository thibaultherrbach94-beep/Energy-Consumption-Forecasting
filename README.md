# Energy Consumption Forecasting

## Overview

Ce projet développe un pipeline complet de prévision de consommation électrique combinant statistique, machine learning, deep learning, séries temporelles et analyse de drift.

L'objectif est de prédire la consommation électrique à partir de l'historique de consommation, de variables calendaires et de données météorologiques.

Une attention particulière est portée à la validation temporelle : le modèle final est évalué sur des données futures de 2026 totalement indépendantes des phases d'entraînement et de sélection.

---

# Final Model

Le modèle final est un pipeline hybride composé de quatre niveaux :

```text
Historical consumption + Weather + Calendar
                  |
                  v
      High-dimensional features
                  |
                  v
          Ridge Regression
                  |
                  v
           Residual MLP
                  |
                  v
        Patch Transformer
                  |
                  v
       Refined Event Expert
                  |
                  v
          Final prediction
