# Promotion Effectiveness Analysis Using Causal Forest & Uplift Modeling

## Business Problem

Retail companies invest heavily in promotions, but increased sales do not always mean promotions caused the improvement. This project applies **causal inference and machine learning** to measure the true incremental impact of promotions.

The key question:

> "How many additional sales were generated because of promotions?"

The analysis identifies where promotions create the highest impact across countries, products, and customer segments.

---

# Objectives

* Estimate the causal impact of promotions on sales.
* Measure incremental units sold and revenue impact.
* Identify countries with the strongest promotional response.
* Discover high-uplift products and segments.
* Validate results using placebo testing.

---

# Dataset

The project uses FMCG sales data containing approximately **1 million records** across multiple countries.

Key features include:

* Sales data: units sold, promotions, prices
* Product information: SKU, category, brand
* Store information: location and sales channel
* External factors: weather, inventory, and supply chain variables

---

# Methodology

The analysis follows a causal machine learning framework:

1. **Feature Engineering**

   * Encoded categorical variables.
   * Added sales, inventory, weather, and product controls.

2. **Predictive Benchmarking**

   * Linear Regression
   * Ridge Regression

3. **Causal Forest DML**

   * Estimated the true treatment effect of promotions.
   * Calculated Average Treatment Effects (ATE).

4. **Uplift Analysis**

   * Measured promotional lift.
   * Identified high-performing segments.

5. **Heterogeneous Effects**

   * Analyzed differences across countries, products, brands, and channels.

6. **Placebo Validation**

   * Randomized promotion assignments to confirm that observed effects were not due to chance.

---

# Results

| Country | Promo Lift | Incremental Revenue |
| ------- | ---------: | ------------------: |
| Poland  |    100.49% |             $406.24 |
| Italy   |     99.26% |             $439.04 |
| Germany |     98.16% |             $416.53 |
| Austria |     97.14% |             $378.26 |
| Spain   |     96.46% |             $304.49 |
| France  |     95.17% |             $380.04 |

### Key Insights

* Promotions generated significant positive sales impact across all countries.
* Poland achieved the highest promotional lift.
* Italy generated the highest incremental revenue.
* Promotion effectiveness varied by product, category, brand, and sales channel.
* Placebo testing confirmed that results were not driven by random treatment assignment.

---

# Visual Results

### Promotion Lift by Country

![Promotion Lift](images/promotion_lift_by_country.png)

### Incremental Revenue Impact

![Revenue Impact](images/incremental_revenue.png)

### Average Treatment Effect

![ATE](images/ate_comparison.png)

### Feature Importance

![Feature Importance](images/feature_importance.png)

### Placebo Validation

![Placebo](images/placebo_validation.png)

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* EconML
* Causal Forest DML
* Random Forest
* Matplotlib

---

# Project Structure

```text
Promotion-Effectiveness-Causal-Inference/

├── promo_effectiveness_analysis.py
├── visualization.py
├── README.md
├── requirements.txt

├── data/
│   └── sample_data.csv

├── outputs/
│   └── country_results.csv

└── images/
    ├── promotion_lift_by_country.png
    ├── incremental_revenue.png
    ├── ate_comparison.png
    ├── feature_importance.png
    └── placebo_validation.png
```

---

# How To Run

```bash
pip install -r requirements.txt

python promo_effectiveness_analysis.py

python visualization.py
```

---

# Business Recommendations

* Prioritize promotions in markets with high causal uplift.
* Target promotions toward products with strong response.
* Avoid applying promotions uniformly across all products.
* Use causal analysis to optimize promotional spending.

---

# Author

**Alex Barugahara**

* BSc Mathematics
* ACCA Finalist
* Google Advanced Data Analytics Professional Certificate

Data analyst focused on applying statistics, machine learning, and business analytics to solve real-world problems.
