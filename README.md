# IA Culturelles

Dashboard d'analyse des usages et des biais culturels des modèles d'intelligence artificielle générative, construit sur les données de la plateforme [Compar:IA](https://comparia.beta.gouv.fr) (beta.gouv.fr).

Les conversations sont croisées avec deux bases ouvertes du Ministère de la Culture — **Joconde** (580 000 oeuvres) et **Basilic** (86 000 équipements culturels) — pour mesurer les biais temporels, géographiques et par domaine artistique de chaque modèle.

Les données utilisées couvrent la période jusqu'à **février 2026**.

---

## Structure du projet

```
culture_ia/
├── Accueil.py                   # Page d'accueil Streamlit (point d'entrée)
├── pages/
│   ├── Statistiques IA.py       # Cockpit par modèle : winrate, conso, tokens...
│   ├── Analyse de biais.py      # Biais culturels : Joconde + Basilic
│   └── Arena.py                 # Explorateur de conversations
├── data/
│   ├── data_manager.py          # Toutes les fonctions de calcul et d'affichage
│   ├── data_cleaning.py         # Script de nettoyage et préparation des données
│   ├── bias_analysis/
│   │   └── run_bias.py          # Calcul TF-IDF et scores de biais
│   └── cleaned/                 # Parquets générés par data_cleaning.py
└── tests/
    ├── test_data_cleaning.py
    ├── test_data_manager.py
    └── test_run_bias.py
```

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/alexisvincent37/dashboard_IA_culturelles
cd dashboard_IA_culturelles

```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Données

### Données Compar:IA (obligatoires)

Télécharger les trois fichiers suivants et les placer à la racine du dossier `data/` :

| Fichier | Lien |
|---|---|
| `votes.parquet` | https://www.data.gouv.fr/api/1/datasets/r/7651fd0b-f222-43b3-8db8-ed6ae660d313 |
| `reactions.parquet` | https://www.data.gouv.fr/api/1/datasets/r/4ffc86e1-84a4-4fdc-9726-66408e596fef |
| `conversations.parquet` | https://www.data.gouv.fr/api/1/datasets/r/9dd3d51f-4299-4193-ab46-81ae039fe1be |

En ligne de commande :

```bash
cd data
curl -L -o votes.parquet         "https://www.data.gouv.fr/api/1/datasets/r/7651fd0b-f222-43b3-8db8-ed6ae660d313"
curl -L -o reactions.parquet     "https://www.data.gouv.fr/api/1/datasets/r/4ffc86e1-84a4-4fdc-9726-66408e596fef"
curl -L -o conversations.parquet "https://www.data.gouv.fr/api/1/datasets/r/9dd3d51f-4299-4193-ab46-81ae039fe1be"
```

### Données du Ministère de la Culture (pour l'analyse de biais uniquement)

Ces fichiers sont nécessaires uniquement si vous souhaitez relancer le calcul TF-IDF et les scores de biais via `run_bias.py`. Les placer dans `data/bias_analysis/` :

| Fichier | Lien |
|---|---|
| `joconde.csv` | https://www.data.gouv.fr/api/1/datasets/r/7e3307c2-f2ff-455c-bbca-bb6f11aec7bb |
| `basilic.csv` | https://www.data.gouv.fr/api/1/datasets/r/dced78ee-0823-4b61-86e6-57717308d4e4 |

```bash
cd data/bias_analysis
curl -L -o joconde.csv "https://www.data.gouv.fr/api/1/datasets/r/7e3307c2-f2ff-455c-bbca-bb6f11aec7bb"
curl -L -o basilic.csv "https://www.data.gouv.fr/api/1/datasets/r/dced78ee-0823-4b61-86e6-57717308d4e4"
```

---

## Initialisation

Une fois les données Compar:IA en place, lancer le script de nettoyage depuis la racine du projet. Ce script est à exécuter une seule fois, ou à chaque mise à jour des données sources. Il génère les parquets optimisés dans `data/cleaned/`.

```bash
python data/data_cleaning.py
```
Cette étape prend quelques minutes.
---

## Lancer le dashboard

```bash
streamlit run Accueil.py
```

---

## Recalculer les scores de biais (optionnel)

Si vous avez mis à jour les données Compar:IA ou les référentiels Joconde/Basilic, relancer le pipeline TF-IDF et de scoring depuis le dossier `data/bias_analysis/` :

```bash
python data/bias_analysis/run_bias.py
```

Ce script lit `bias_conv.parquet` (généré par `data_cleaning.py`), `joconde.csv` et `basilic.csv`, et produit `data/bias_analysis/bias_scores.parquet` qui alimente la page Analyse de biais.

---

## Tests

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## Sources

- Données de conversations : [Compar:IA](https://comparia.beta.gouv.fr) — beta.gouv.fr
- Referentiel oeuvres : [Joconde](https://www.data.gouv.fr/fr/datasets/joconde-catalogue-des-collections-des-musees-de-france/) — Ministere de la Culture
- Referentiel equipements : [Basilic](https://www.data.gouv.fr/fr/datasets/basilic-equipements-culturels/) — Ministere de la Culture
- Projet realise dans le cadre d'**Open Data University** — Saison 4 — 2025/2026
