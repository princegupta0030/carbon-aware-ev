import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(filepath="data/processed_hourly.csv"):
    df = pd.read_csv(filepath, parse_dates=['DATE_TIME'])
    df.sort_values('DATE_TIME', inplace=True)
    return df

def feature_engineering(df):
    """
    Creates temporal features that might help the model alongside weather data.
    """
    df['HOUR'] = df['DATE_TIME'].dt.hour
    df['DAY_OF_YEAR'] = df['DATE_TIME'].dt.dayofyear
    # Important features for Solar: IRRADIATION, TEMPERATURE, HOUR
    features = ['IRRADIATION', 'MODULE_TEMPERATURE', 'AMBIENT_TEMPERATURE', 'HOUR']
    target = 'AC_POWER'
    return df, features, target

def train_and_evaluate(df, features, target, model_path="src/models"):
    """
    Splits data chronologically (first 80% train, last 20% test).
    Trains a Random Forest Regressor and outputs metrics.
    """
    # Chronological Split (No Data Leakage)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    logger.info(f"Training set: {train_df['DATE_TIME'].min()} to {train_df['DATE_TIME'].max()} ({len(train_df)} samples)")
    logger.info(f"Test set: {test_df['DATE_TIME'].min()} to {test_df['DATE_TIME'].max()} ({len(test_df)} samples)")

    # Initialize and Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    logger.info(f"Evaluation Metrics -> MAE: {mae:.2f} kW | RMSE: {rmse:.2f} kW")

    # Save Model
    os.makedirs(model_path, exist_ok=True)
    model_filepath = os.path.join(model_path, "rf_solar_model.joblib")
    joblib.dump(model, model_filepath)
    logger.info(f"Model saved to {model_filepath}")

    # Return metrics for dashboard usage
    metrics = {'MAE': mae, 'RMSE': rmse}

    # Save a subset of predictions vs actual for visualization later
    results_df = test_df[['DATE_TIME', target]].copy()
    results_df['PREDICTED_AC_POWER'] = predictions
    results_df.to_csv(os.path.join(model_path, "test_predictions.csv"), index=False)

    return model, metrics, results_df

def run_forecasting_pipeline():
    logger.info("Starting Forecasting Pipeline...")
    if not os.path.exists("data/processed_hourly.csv"):
        logger.error("Processed data not found. Please run data_processing.py first.")
        return

    df = load_data()
    df, features, target = feature_engineering(df)
    train_and_evaluate(df, features, target)

if __name__ == "__main__":
    run_forecasting_pipeline()
