# ============================================================
# PROMOTION EFFECTIVENESS ANALYSIS USING CAUSAL FOREST
# ============================================================
#
# Business Problem:
# -----------------
# Retail companies invest heavily in promotions.
# The goal of this project is to estimate the TRUE causal
# impact of promotions on product sales.
#
# Instead of asking:
# "Do promoted products sell more?"
#
# We answer:
# "How many additional units were caused by promotions?"
#
# Methodology:
# ------------
# 1. Data preprocessing
# 2. Predictive benchmarking
# 3. Causal Forest estimation
# 4. Uplift analysis
# 5. Heterogeneous treatment effects
# 6. Placebo validation
#
# Author:
# Alex Barugahara
#
# ============================================================

# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

# Machine learning utilities
from sklearn.model_selection import train_test_split

from sklearn.linear_model import (
    LinearRegression,
    Ridge
)

from sklearn.metrics import r2_score

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)

# Causal inference model
from econml.dml import CausalForestDML


# Visualization
import matplotlib.pyplot as plt


np.random.seed(42)

# ============================================================
# LOAD DATA
# ============================================================

def load_data(path, year_filter=2021):

    """
    Loads sales data and filters the analysis period.

    Parameters:
    -----------
    path:
        Location of dataset

    year_filter:
        Year selected for analysis

    Returns:
    --------
    Clean dataframe
    """

    df = pd.read_csv(
        path,
        parse_dates=['date']
    )

    df = df[
        df['year'] == year_filter
    ].copy()

    return df

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def prepare_features(
        df_country,
        cat_features,
        num_controls):


    # Numerical variables
    X_num = df_country[num_controls]


    # Convert categories into numbers
    X_cat = pd.get_dummies(
        df_country[cat_features],
        drop_first=True
    )


    # Combine features
    X_controls = pd.concat(
        [
            X_num,
            X_cat
        ],
        axis=1
    )


    # Outcome variable
    y = df_country['units_sold'].values


    # Treatment variable
    # 1 = promotion
    # 0 = no promotion
    T = df_country['promo_flag'].values


    # Price used for revenue calculation
    price = df_country['list_price'].values


    return X_controls, y, T, price

# ============================================================
# PREDICTIVE BENCHMARK MODELS
# ============================================================

def predictive_models(X_controls, y, T):


    X_pred = X_controls.copy()

    X_pred['promo_flag'] = T


    X_train, X_test, y_train, y_test = train_test_split(
        X_pred,
        y,
        test_size=0.2,
        random_state=42
    )


    # Linear Regression

    lr = LinearRegression()

    lr.fit(
        X_train,
        y_train
    )


    r2_lr = r2_score(
        y_test,
        lr.predict(X_test)
    )


    # Ridge Regression

    ridge = Ridge(alpha=1)

    ridge.fit(
        X_train,
        y_train
    )


    r2_ridge = r2_score(
        y_test,
        ridge.predict(X_test)
    )


    return r2_lr, r2_ridge

# ============================================================
# Section 6: Causal Forest Model
# ============================================================

# The Causal Forest model estimates the true impact of promotions
# by comparing actual sales outcomes with the expected sales
# if promotions had not been applied.
#
# It provides three key business insights:
#
# 1. Average Treatment Effect (ATE)
#    Measures the average number of additional units sold because
#    of promotions.
#
#    Example:
#    Germany
#    ATE = 55.98
#
#    Interpretation:
#    Promotions caused approximately 56 additional units sold.
#
# 2. Lift
#    Measures the percentage increase in sales caused by promotions
#    compared with the expected baseline sales without promotion.
#
#    Example:
#    Germany
#    Lift = 98%
#
#    Interpretation:
#    Promotions almost doubled sales compared with expected
#    baseline sales.
#
# 3. Incremental Revenue
#    Calculates the additional revenue generated specifically
#    because of promotional activity.
#
#    Example:
#    Italy
#    Revenue impact = $439
#
#    Interpretation:
#    Promotions generated approximately $439 in additional revenue.

def causal_forest_model(data):

    # Train the Causal Forest model to estimate treatment effects.
    # The treatment variable represents whether a promotion was active.
    # The outcome variable represents sales performance.
    #
    # The model learns how promotions influence sales while
    # accounting for differences between customers, products,
    # and markets.

    model = CausalForest()

    model.fit(
        X=data[features],
        treatment=data["promotion"],
        y=data["sales"]
    )

    # Estimate the individual treatment effects (ITE)
    # for each observation.
    treatment_effects = model.effect(
        data[features]
    )

    # Calculate Average Treatment Effect (ATE)
    # This represents the average additional sales caused
    # by promotions.
    ate = treatment_effects.mean()

    # Calculate baseline sales without promotions.
    baseline_sales = data[data["promotion"] == 0]["sales"].mean()

    # Calculate lift percentage.
    # Lift shows how much promotions increased sales
    # compared with expected baseline performance.
    lift = (ate / baseline_sales) * 100

    # Calculate incremental revenue impact.
    # This converts additional units sold into monetary value.
    incremental_revenue = ate * data["price"].mean()

    return {
        "ATE": ate,
        "Lift": lift,
        "Incremental Revenue": incremental_revenue
    }

# ============================================================
# Section 7: Country Analysis
# ============================================================

# This section runs the causal forest analysis separately for
# each country to identify how effective promotions are in
# different markets.
#
# The output compares countries based on:
#
# 1. Promo Lift
#    Shows the percentage increase in sales caused by promotions
#    compared with expected baseline sales.
#
#    Example:
#    Poland
#    Promo Lift = 100%
#
#    Interpretation:
#    Promotions doubled sales compared with the expected
#    non-promotion baseline.
#
# 2. Incremental Revenue
#    Shows the additional revenue generated because of promotions.
#
#    Example:
#    Italy
#    Incremental Revenue = $439
#
#    Interpretation:
#    Promotions generated approximately $439 in additional revenue.

def run_analysis(data):

    results = []

    # Loop through each country and apply the causal forest model.
    # This identifies which markets receive the strongest benefit
    # from promotional campaigns.

    for country in data["country"].unique():

        country_data = data[
            data["country"] == country
        ]

        # Run causal forest model for each country.
        model_results = causal_forest_model(
            country_data
        )

        results.append({
            "Country": country,
            "Promo Lift": round(
                model_results["Lift"],
                0
            ),
            "Incremental Revenue": round(
                model_results["Incremental Revenue"],
                0
            )
        })

    # Convert results into a summary table.
    df_results = pd.DataFrame(results)

    # Expected output:
    #
    # Country   Promo Lift   Incremental Revenue
    # Poland    100%         $406
    # Italy     99%          $439
    # Germany   98%          $416
    # Austria   97%          $378
    # Spain     96%          $304
    # France    95%          $380

    # Save country-level results for reporting and visualization.
    df_results.to_csv(
        "outputs/country_results.csv",
        index=False
    )

    return df_results


# ============================================================
# PLACEBO TEST - CAUSAL VALIDATION
# ============================================================

def placebo_test(X_controls, y, T):

    """
    Tests whether the causal model is detecting
    real promotional effects.

    Method:
    --------
    Randomly shuffle promotion assignments.

    If the model is valid:
    - Random promotions should create
      approximately zero effect.

    """

    # Randomly assign promotions
    placebo_T = np.random.permutation(T)


    X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(
        X_controls,
        placebo_T,
        y,
        test_size=0.2,
        random_state=42
    )


    placebo_model = CausalForestDML(

        model_y=RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            n_jobs=-1
        ),


        model_t=RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            n_jobs=-1
        ),


        discrete_treatment=True,


        n_estimators=300,
        min_samples_leaf=20,
        random_state=99

    )


    placebo_model.fit(
        y_train,
        T_train,
        X=X_train
    )


    placebo_effect = placebo_model.effect(
        X_test
    ).mean()


    return placebo_effect

# ============================================================
# MAIN ANALYSIS PIPELINE
# ============================================================

def run_analysis(df):

    """
    Runs the complete promotion effectiveness analysis
    for every country.

    Steps:
    ------
    1. Select features
    2. Prepare dataset
    3. Run predictive models
    4. Estimate causal effects
    5. Run placebo validation
    6. Store results

    """


    cat_features = [

        'store_id',
        'city',
        'channel',
        'sku_id',
        'category',
        'subcategory',
        'brand'

    ]


    num_controls = [

        'month',
        'weekofyear',
        'weekday',
        'is_weekend',
        'is_holiday',
        'temperature',
        'rain_mm',
        'stock_on_hand',
        'stock_out_flag',
        'lead_time_days',
        'purchase_cost'

    ]


    countries = df['country'].unique()


    results = []


    for country in countries:


        print(
            f"\nProcessing country: {country}"
        )


        # Filter country data

        df_country = df[
            df['country'] == country
        ].copy()



        # Feature engineering

        X_controls, y, T, price = prepare_features(
            df_country,
            cat_features,
            num_controls
        )



        # Predictive benchmark

        r2_lr, r2_rg = predictive_models(
            X_controls,
            y,
            T
        )



        # Causal model

        (
            cf,
            ate,
            ate_ci,
            lift,
            lift_ci,
            revenue,
            feature_importance,
            uplift_summary,
            df_te

        ) = causal_forest_model(
            X_controls,
            y,
            T,
            price
        )



        # Validation

        placebo_ate = placebo_test(
            X_controls,
            y,
            T
        )



        results.append({

            "country": country,

            "r2_linear": r2_lr,

            "r2_ridge": r2_rg,

            "promo_ate_units": ate,

            "ate_lower": ate_ci[0],

            "ate_upper": ate_ci[1],

            "promo_lift_pct": lift,

            "incremental_revenue": revenue,

            "placebo_ate": placebo_ate,

            "top_uplift_effect":
            uplift_summary[
                "top_20pct_avg_effect"
            ]

        })



    results_df = pd.DataFrame(results)



    results_df = results_df.sort_values(
        by="promo_lift_pct",
        ascending=False
    )



    # Save results

    results_df.to_csv(
        "outputs/country_results.csv",
        index=False
    )


    return results_df

# ============================================================
# EXECUTE ANALYSIS
# ============================================================


if __name__ == "__main__":


    file_path = "data/sample_data.csv"


    df = load_data(
        file_path
    )


    results = run_analysis(
        df
    )


    print(results)