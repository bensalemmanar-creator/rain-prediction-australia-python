# Will It Rain Tomorrow? Rain Prediction in Australia

A Python machine learning project that predicts **whether it will rain the next day in Australia**, using daily weather observations (`RainTomorrow`: Yes / No). Built as a team project by **Manar Ben Salem, Wiem Habbassi and Ghofrane Ayari**.

## Files

| File | Description |
|---|---|
| [`Rain_Prediction_Australia.ipynb`](./Rain_Prediction_Australia.ipynb) | Full analysis notebook (Google Colab / Jupyter) |
| [`Rain_Prediction_Presentation.pdf`](./Rain_Prediction_Presentation.pdf) | Key presentation (17 slides) |

## Business question

> Given today's weather, will it rain tomorrow?

A reliable next-day rain warning helps farmers plan irrigation and harvest, event organisers decide on cover or postponement, construction teams schedule outdoor work, and airports prepare staff and equipment.

## Data

- **Source:** `weatherAUS.csv`, daily observations from Australian weather stations (Kaggle "Rain in Australia" dataset)
- **Size:** 145,460 days x 23 columns, 49 locations, Nov 2007 - Jun 2017
- **Variables:** 16 numerical (temperature, rainfall, wind, humidity, pressure, cloud, sunshine) and 6 categorical (location, three wind directions, `RainToday`, target `RainTomorrow`)

The dataset is not included in this repository. Download `weatherAUS.csv` from Kaggle and place it next to the notebook (or upload it when the notebook asks in Colab).

## Approach

1. **Data description and descriptive analysis:** structure, missing values, duplicates, summary statistics, comparison of dry vs rainy next days
2. **Visualization:** class imbalance (only ~22% of days are followed by rain), seasonality by month, boxplots of key variables
3. **Data preparation:** removed duplicates and rows without a target, dropped 4 columns with more than 35% missing values (Sunshine, Evaporation, Cloud9am, Cloud3pm), extracted the month from the date, filled missing values (median for numbers, mode for categories), encoded categories
4. **Feature selection:** removed redundant inputs (Spearman and Cramer's V above 0.80), then kept the 10 most informative features with `SelectKBest` (mutual information)
5. **Balancing:** random undersampling of the training set only, so the validation set keeps the real class mix
6. **Modelling:** five classifiers (Logistic Regression, SVM, Decision Tree, Random Forest, Gradient Boosting) tuned with 5-fold `GridSearchCV`, scored on ROC AUC, with scaling inside the pipeline to avoid leakage
7. **Evaluation and decision making:** accuracy, precision, recall, F1, confusion matrix, ROC curve, feature importance, and recommendations

## Key results

| Model | Validation AUC | Recall (rain) | F1 |
|---|---|---|---|
| **SVM (selected)** | **0.860** | **77.0%** | **0.624** |
| Random Forest | 0.858 | 76.0% | 0.613 |
| Logistic Regression | 0.853 | 74.0% | 0.602 |
| Gradient Boosting | 0.853 | 74.1% | 0.602 |
| Decision Tree | 0.815 | 71.1% | 0.571 |

- **Balancing matters:** with undersampling, the SVM catches 77% of rainy days instead of 48%. Accuracy drops slightly (84% to 79%) because the unbalanced model mostly predicts "no rain", and F1 improves.
- **Main drivers of tomorrow's rain:** 3pm humidity (the strongest signal), 9am pressure, 9am humidity and wind gusts. Rain tends to follow high humidity, lower pressure and stronger gusts.

## Recommendations

1. Deploy the SVM as a daily "rain tomorrow" alert.
2. Tune the probability threshold per user: lower for farmers and airports who cannot afford to miss rain, higher to reduce false alarms.
3. Watch afternoon humidity and falling pressure as the simplest warning signs.
4. Improve the data: sunshine and cloud cover were 40-48% missing, and collecting them could improve accuracy.

## Tools

Python, pandas, NumPy, scikit-learn, SciPy, matplotlib, seaborn, Google Colab

## Team

- Manar Ben Salem
- Wiem Habbassi
- Ghofrane Ayari

## How to run

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn jupyter
jupyter notebook Rain_Prediction_Australia.ipynb
```

The first code cell uses Google Colab's file upload. If you run the notebook locally, replace it with a path to your local `weatherAUS.csv`. Parameters such as the sample size (20,000 rows to keep the SVM and grid search fast), the missing-value limit and the prediction threshold are in the settings cell.
