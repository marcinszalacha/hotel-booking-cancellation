# Hotel Booking Cancellation Prediction

End-to-end machine learning project for predicting hotel booking cancellations, comparing multiple classification approaches, evaluating operating thresholds, and translating predictive results into business interventions and experiment design.

[Open the notebook in Google Colab](https://colab.research.google.com/github/marcinszalacha/hotel-booking-cancellation/blob/main/hotel_booking_cancellation_classification.ipynb)

---

## Business objective

Hotel cancellations create uncertainty in room availability, staffing, pricing, and revenue planning.

The objective of this project is to identify bookings at elevated risk of cancellation while prioritizing **precision** and maintaining useful **recall**.

A precision-first approach is used because false-positive predictions may lead to unnecessary customer interventions, additional operational cost, or customer friction.

The model is intended as a **targeting tool**: it identifies where preventative action may have the highest expected value, rather than assuming that prediction itself will reduce cancellations.

---

## Project workflow

The project follows an end-to-end analytical workflow:

1. Data quality assessment and cleaning
2. Target and class-imbalance analysis
3. Business-oriented exploratory data analysis
4. Correlation funnel analysis
5. Leakage prevention and feature engineering
6. Train / validation / untouched test split
7. Logistic Regression baseline
8. Shallow Decision Tree for interpretable nonlinear patterns
9. Random Forest
10. XGBoost
11. Hyperparameter tuning
12. PyCaret model comparison
13. Blending and stacking
14. Neural-network benchmark
15. Cross-validation stability analysis
16. Precision-focused model selection
17. Threshold sensitivity analysis
18. Final untouched-test evaluation
19. Permutation feature importance
20. Business intervention hypotheses
21. Controlled experiment analysis
22. Power analysis for a future cancellation-reduction experiment

---

## Data

The project uses two datasets stored directly in this repository:

- `data/hotel_cancellations.csv`
- `data/hotel_bookings_geo_experiment.csv`

The main dataset contains hotel booking characteristics and the binary target:

`is_canceled`

The second dataset contains treatment/control geo-level observations used to demonstrate experimental analysis separately from the predictive modelling exercise.

---

## Data leakage prevention

Variables that reveal the final booking outcome are removed before modelling:

- `reservation_status`
- `reservation_status_date`

This ensures that model performance reflects information that could reasonably be available before the final cancellation outcome is known.

---

## Feature engineering

Selected derived features include:

- `total_stay`
- `total_guests`
- numeric arrival month representation

Missing values are handled inside preprocessing pipelines.

Numeric variables use median imputation, while categorical variables use most-frequent imputation and one-hot encoding.

---

## Class imbalance

Approximately **37% of bookings are cancellations**, so the target is moderately rather than severely imbalanced.

SMOTE, oversampling, and undersampling are intentionally not applied.

Instead, the project:

- uses stratified train / validation / test splits
- evaluates precision, recall, F1, and ROC-AUC
- prioritizes precision according to the business objective
- evaluates model stability across cross-validation folds

A model is considered a **stable candidate** when both:

- Precision standard deviation ≤ 1%
- Recall standard deviation ≤ 1%

---

## Models evaluated

The project compares several modelling approaches:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- PyCaret model comparison
- Blended ensemble
- Stacked ensemble
- Neural-network benchmark

A shallow Decision Tree is also used separately as an **interpretation tool** to reveal simple nonlinear relationships and interactions in business-readable units.

---

## Final model

**Random Forest** was selected as the final model because it provided the strongest combination of:

- high precision
- useful recall
- strong ROC-AUC
- cross-validation stability
- robust performance on the validation data

The neural-network benchmark achieved competitive predictive performance but did not satisfy the project's predefined stability rule across repeated runs.

---

## Final test performance

The selected Random Forest was evaluated once on the untouched test set after model and threshold decisions were finalized.

Approximate final performance:

| Metric | Result |
|---|---:|
| Accuracy | 89.4% |
| Precision | 89.4% |
| Recall | 80.9% |
| F1-score | 84.9% |
| ROC-AUC | ~0.96 |

The final operating threshold is **0.50**.

---

## Threshold sensitivity analysis

The project evaluates how changing the classification threshold affects the precision-recall trade-off.

Example validation results:

| Threshold | Precision | Recall | False Positives | False Negatives |
|---:|---:|---:|---:|---:|
| 0.40 | 84.3% | 86.2% | 1,423 | 1,217 |
| 0.50 | 89.2% | 79.8% | 856 | 1,785 |
| 0.521 | 90.2% | 78.1% | 753 | 1,936 |
| 0.60 | 93.0% | 72.5% | 483 | 2,428 |

Lower thresholds favor **recall** and identify more potential cancellations, but create more false-positive interventions.

Higher thresholds favor **precision**, generating fewer but higher-confidence alerts.

The standard **0.50 threshold is retained** because it provides a reasonable balance without imposing an arbitrary business constraint.

---

## Model interpretation

Permutation importance indicates that important predictors include variables such as:

- country
- total number of special requests
- customer type
- market segment
- lead time
- deposit type
- required car-parking spaces

The shallow Decision Tree also shows that cancellation behaviour depends on interactions between booking characteristics rather than a single isolated variable.

These relationships are interpreted as **predictive associations, not causal effects**.

---

## From prediction to intervention

The model does not itself reduce cancellations.

Its role is to identify bookings where preventative action may be most worthwhile.

Potential interventions include:

- confirmation email or SMS before the cancellation deadline
- one-click confirm-or-modify journeys
- payment-method revalidation
- proactive outreach for high-value bookings
- flexible rebooking options
- selective incentives where financially justified
- staged reconfirmation for long-lead bookings

These interventions should be treated as hypotheses and validated experimentally before operational scaling.

---

## Controlled experiment analysis

A separate geo-level experiment dataset is used to demonstrate the difference between **prediction** and **causal evaluation**.

During the active experiment period:

- Treatment observations: 1,393
- Control observations: 1,391
- Treatment mean booking volume: 813.77
- Control mean booking volume: 717.55
- Observed relative lift: **13.41%**
- Welch t-test p-value: **0.0545**

The treatment group performed better descriptively, but the result did not reach the conventional 5% statistical significance threshold.

The experiment therefore provides a **promising but inconclusive** result rather than confirmed evidence of a causal treatment effect.

The estimated business-impact figures are treated as descriptive estimates rather than confirmed causal ROI.

---

## Future experiment planning

A power-analysis example is included for a future cancellation-reduction experiment.

Using illustrative assumptions:

- Control cancellation rate: 60%
- Target treatment cancellation rate: 55%
- Statistical power: 80%
- Significance level: 5%

Estimated required sample size:

- **1,534 bookings per group**
- **3,068 bookings total**

These assumptions are illustrative and should be replaced with actual baseline cancellation rates and the minimum business-relevant effect before running a real experiment.

---

## Business takeaway

The recommended operational workflow is:

**Predict risk → prioritize high-confidence bookings → intervene selectively → measure causal impact → scale only validated actions**

The predictive model supports **targeting**.

Randomized experimentation determines whether a proposed intervention actually changes customer behaviour.

---

## Reproducibility

Both datasets are loaded directly from this public GitHub repository using raw GitHub URLs.

No private Google Sheets authentication or local file upload is required.

The notebook is designed to run from a fresh Google Colab session.

Repository structure:

```text
hotel-booking-cancellation/
│
├── hotel_booking_cancellation_classification.ipynb
├── README.md
├── .gitignore
│
└── data/
    ├── hotel_cancellations.csv
    └── hotel_bookings_geo_experiment.csv
