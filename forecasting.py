"""
forecasting.py
--------------
Time-series forecasting of quarterly/monthly drug-category demand using the
REAL "Pharma Drug Sales" Kaggle dataset (2014-01 to 2019-10, monthly).

Drug categories (ATC codes) act as our "product lines":
  M01AB - Antiinflammatory/antirheumatic, non-steroid (acetic acid derivatives)
  M01AE - Antiinflammatory/antirheumatic, non-steroid (propionic acid derivatives)
  N02BA - Analgesics/antipyretics (salicylic acid derivatives)
  N02BE - Analgesics/antipyretics (pyrazolones/anilides, e.g. paracetamol-class)
  N05B  - Anxiolytics
  N05C  - Hypnotics/sedatives
  R03   - Drugs for obstructive airway disease (asthma/COPD)
  R06   - Systemic antihistamines

For each product line we:
  1. Hold out the final 6 months as a test set.
  2. Fit an ARIMA model (order auto-selected via pmdarima) on the training set.
  3. Forecast the holdout period and compute MAPE against actuals.
  4. Refit on the full series and forecast the next quarter (3 months) forward.

Outputs:
  outputs/forecast_results.csv   - one row per product with holdout MAPE + next-quarter forecast
  outputs/forecast_detail.csv    - month-by-month actual vs. fitted vs. forecast, all products
"""

import os
import warnings
import numpy as np
import pandas as pd
import pmdarima as pm

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "salesmonthly.csv")
OUT_RESULTS = os.path.join(BASE_DIR, "outputs", "forecast_results.csv")
OUT_DETAIL = os.path.join(BASE_DIR, "outputs", "forecast_detail.csv")

PRODUCT_COLS = ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]
HOLDOUT_MONTHS = 6
FORECAST_HORIZON = 3  # next quarter


def mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def load_series() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["datum"] = pd.to_datetime(df["datum"])
    df = df.set_index("datum").asfreq("ME")
    # Drop the last row if it's a partial/incomplete month (data quality check against daily file)
    daily = pd.read_csv(DATA_PATH.replace("salesmonthly", "salesdaily"))
    daily["datum"] = pd.to_datetime(daily["datum"])
    last_month_days = daily[daily["datum"].dt.to_period("M") == df.index[-1].to_period("M")].shape[0]
    if last_month_days < 25:  # incomplete month
        print(f"Dropping {df.index[-1].strftime('%Y-%m')}: only {last_month_days} days recorded (partial month)")
        df = df.iloc[:-1]
    return df


def forecast_product(series: pd.Series, product: str) -> dict:
    # log1p stabilizes variance for these multiplicative, right-skewed, zero-inclusive series
    log_series = np.log1p(series)

    train = log_series.iloc[:-HOLDOUT_MONTHS]
    test_actual = series.iloc[-HOLDOUT_MONTHS:]

    # Auto-select (p,d,q)(P,D,Q)m via stepwise AIC search; monthly seasonality m=12
    model = pm.auto_arima(
        train,
        seasonal=True,
        m=12,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=3, max_q=3, max_P=2, max_Q=2, max_D=1,
        n_fits=50,
    )

    # Holdout validation (invert log1p)
    holdout_pred_log = model.predict(n_periods=HOLDOUT_MONTHS)
    holdout_pred = np.expm1(np.array(holdout_pred_log))
    holdout_pred = np.clip(holdout_pred, 0, None)
    holdout_mape = mape(test_actual.values, holdout_pred)

    # Refit on full history, forecast next quarter
    final_model = pm.auto_arima(
        log_series,
        seasonal=True,
        m=12,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=3, max_q=3, max_P=2, max_Q=2, max_D=1,
        n_fits=50,
    )
    future_log = final_model.predict(n_periods=FORECAST_HORIZON)
    future = np.clip(np.expm1(np.array(future_log)), 0, None)
    future_index = pd.date_range(series.index[-1] + pd.offsets.MonthEnd(1), periods=FORECAST_HORIZON, freq="ME")

    holdout_pred_arr = holdout_pred
    detail_rows = []
    for idx_pos, (d, actual) in enumerate(test_actual.items()):
        detail_rows.append({"product": product, "date": d, "actual": actual,
                             "forecast": holdout_pred_arr[idx_pos],
                             "type": "holdout_test"})
    for d, f in zip(future_index, future):
        detail_rows.append({"product": product, "date": d, "actual": np.nan,
                             "forecast": f, "type": "next_quarter_forecast"})

    return {
        "product": product,
        "order": str(model.order),
        "seasonal_order": str(model.seasonal_order),
        "holdout_mape_pct": round(holdout_mape, 2),
        "next_quarter_total_units": round(float(np.sum(future)), 1),
        "next_quarter_avg_monthly_units": round(float(np.mean(future)), 1),
    }, detail_rows


def run():
    df = load_series()
    results, all_detail = [], []

    for product in PRODUCT_COLS:
        print(f"Fitting ARIMA for {product} ...")
        res, detail = forecast_product(df[product], product)
        results.append(res)
        all_detail.extend(detail)

    results_df = pd.DataFrame(results).sort_values("holdout_mape_pct")
    detail_df = pd.DataFrame(all_detail)

    results_df.to_csv(OUT_RESULTS, index=False)
    detail_df.to_csv(OUT_DETAIL, index=False)

    print("\n=== Holdout Accuracy (MAPE %) by Product ===")
    print(results_df[["product", "order", "holdout_mape_pct", "next_quarter_total_units"]].to_string(index=False))

    overall_mape = results_df["holdout_mape_pct"].mean()
    print(f"\nAverage MAPE across all product lines: {overall_mape:.2f}%")
    print(f"Implied average forecast accuracy: {100 - overall_mape:.2f}%")


if __name__ == "__main__":
    run()
