import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set style
plt.rcParams.update({'font.size': 12})
sns.set(style="darkgrid")

# ========== Helper Functions ==========

def create_lag_features(series: pd.Series, window: int):
    """
    Create lagged features from a pandas Series.
    Returns (X, y) where X shape = (n_samples, window) and y is next day's value.
    """
    values = series.values
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i-window:i])
        y.append(values[i])
    return np.array(X), np.array(y)

def iterative_forecast(model, last_window, days):
    """
    Iteratively forecast 'days' future steps using the model and the last observed window.
    last_window: shape (window,)
    """
    preds = []
    window = last_window.copy().tolist()
    for _ in range(days):
        X_in = np.array(window[-len(last_window):]).reshape(1, -1)
        p = model.predict(X_in)[0]
        preds.append(p)
        window.append(p)
    return np.array(preds)

# ========== Main Script ==========

def main(args):
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"❌ CSV file not found: {csv_path}")

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    if "Close" not in df.columns:
        raise ValueError("CSV must contain a 'Close' column.")

    df = df.sort_values("Date").reset_index(drop=True)
    df = df[["Date", "Close"]].dropna().reset_index(drop=True)

    # Configuration
    window = args.window
    forecast_days = args.forecast_days

    # Create lag features
    X, y = create_lag_features(df["Close"], window=window)
    dates = df["Date"].iloc[window:].reset_index(drop=True)

    # Train/test split (time-based)
    n_samples = X.shape[0]
    split_idx = int(n_samples * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Save metrics summary
    summary_path = results_dir / "metrics_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Metrics Summary\n")
        f.write("-----------------\n")
        f.write(f"window: {window}\n")
        f.write(f"n_train: {len(y_train)}\n")
        f.write(f"n_test: {len(y_test)}\n")
        f.write(f"rmse: {rmse:.4f}\n")
        f.write(f"r2: {r2:.4f}\n")

    print(f"✅ Metrics saved to: {summary_path}")
    print(f"📊 RMSE: {rmse:.4f}   R²: {r2:.4f}")

    # Forecast next N days
    last_window = df["Close"].values[-window:]
    forecast_values = iterative_forecast(model, last_window, forecast_days)
    last_date = df["Date"].iloc[-1]
    forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days, freq='B')

    # Save forecast
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecast_Close": forecast_values})
    forecast_df.to_csv(results_dir / f"forecast_next_{forecast_days}days.csv", index=False)
    print(f"📈 Forecast CSV saved to: results/forecast_next_{forecast_days}days.csv")

    # ========== Interactive Visualization Menu ==========
    print("\n🔍 Choose an option to view:")
    print("1️⃣  Actual vs Predicted (Test Data)")
    print("2️⃣  Future Forecast (Next 30 Days)")
    print("3️⃣  Combined View (All in one graph)")
    choice = input("👉 Enter choice (1 / 2 / 3): ")

    plt.figure(figsize=(10,5))

    if choice == "1":
        # Actual vs Predicted
        plt.plot(dates_test, y_test, label="Actual", linewidth=2)
        plt.plot(dates_test, y_pred, label="Predicted", linewidth=2, linestyle='--')
        plt.title("Actual vs Predicted Stock Prices", fontsize=16)

    elif choice == "2":
        # Forecast Only
        plt.plot(df["Date"], df["Close"], label="Historical", linewidth=2)
        plt.plot(forecast_df["Date"], forecast_df["Forecast_Close"], label="Forecast", linewidth=2, linestyle='--')
        plt.axvline(x=df["Date"].iloc[-1], color='grey', linestyle=':')
        plt.title(f"Next {forecast_days} Days Forecast", fontsize=16)

    elif choice == "3":
        # Combined
        plt.plot(df["Date"], df["Close"], label="Historical", linewidth=2)
        plt.plot(forecast_df["Date"], forecast_df["Forecast_Close"], label="Forecast", linewidth=2, linestyle='--')
        plt.plot(dates_test, y_pred, label="Predicted (Test)", linewidth=2)
        plt.axvline(x=df["Date"].iloc[-1], color='grey', linestyle=':')
        plt.title("Combined: Historical, Predicted, and Forecast", fontsize=16)

    else:
        print("⚠️ Invalid input. Showing Actual vs Predicted by default.")
        plt.plot(dates_test, y_test, label="Actual", linewidth=2)
        plt.plot(dates_test, y_pred, label="Predicted", linewidth=2, linestyle='--')
        plt.title("Actual vs Predicted Stock Prices", fontsize=16)

    plt.xlabel("Date", fontsize=13)
    plt.ylabel("Close Price", fontsize=13)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ========== Entry Point ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="📈 Stock Price Prediction (Linear Regression)")
    parser.add_argument("--csv", type=str, default=r"C:\Users\sonid\python\stock_data.csv", help="CSV file path")
    parser.add_argument("--forecast_days", type=int, default=30, help="Number of days to forecast")
    parser.add_argument("--window", type=int, default=5, help="Number of lag days used as features")
    args = parser.parse_args()
    main(args)

