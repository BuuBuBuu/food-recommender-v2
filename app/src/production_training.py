# production_training.py
import os
import pickle
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from .path_config import DATA_DIR, MERGED_DATA_PATH, TRAINING_DATA_PATH


def train_production_model(X, y, data_folder):
    print("\nStarting production training...")
    start_time = time.time()

    # 1. Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training shapes: X={X_train.shape}, y={y_train.shape}")

    # 2. Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test)

    # 3. Set model parameters
    params = {
        "objective": "regression",  # We choose regression due to the nature of the target variable which is a continuous variable and to better interpret scores
        "metric": "rmse",  # RMSE provides an interpretable error metric in te same units as the target
        "boosting_type": "gbdt",  # Gradient Boosting Decision Trees effectively capture complex non-linear relationships and interactions among features
        "num_leaves": 31,  # This limits the complexity of each tree, helping to prevent overfitting
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "min_data_in_leaf": 100,
        "max_bin": 255,
    }

    # 4. Train model with callbacks for early stopping
    print("\nTraining model...")
    callbacks = [
        lgb.early_stopping(
            stopping_rounds=50
        ),  # Halt training if no improvement is observed for 50 consecutive rounds, also provides overfitting prevention.
        lgb.log_evaluation(
            period=100
        ),  # Evaluation metrics are logged every 100 iterations
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,  # Model is trained for up to 1000 boosting rounds
        valid_sets=[valid_data],
        callbacks=callbacks,
    )

    # 5. Evaluate
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # RMSE Provides a measure of the average prediction error
    r2 = r2_score(y_test, y_pred) # R² Score Indicates the proportion of variance in the target variable that is explained by the model
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R² score: {r2:.4f}")

    # 6. Feature importance analysis
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importance('gain') # Compute using gain metric to identify which features most significantly reduce prediction error
    })
    importance = importance.sort_values('importance', ascending=False)
    print("\nTop 10 most important features:")
    print(importance.head(10)) # Top 10 features are highlighted

    # 7. Save model and metadata
    print("\nSaving model...")
    model_data = {
        'model': model,
        'feature_names': X.columns.tolist(),
        'model_params': params,
        'feature_importance': importance.to_dict(),
        'metrics': {
            'rmse': rmse,
            'r2': r2
        }
    }

    model_filename = os.path.join(data_folder, "production_ranking_model_enhanced.pkl")
    with open(model_filename, 'wb') as f:
        pickle.dump(model_data, f)

    end_time = time.time()
    print(f"\nTotal training time: {(end_time - start_time)/60:.2f} minutes")
    return model_data

if __name__ == "__main__":
    print("Starting script...")

    # Load existing training data
    data_folder = DATA_DIR
    data_filename = os.path.join(data_folder, "training_data.pkl")

    print(f"\nLoading data from {data_filename}...")
    with open(data_filename, 'rb') as f:
        X, y = pickle.load(f)
    print(f"Data loaded. Shape: {X.shape}")

    # Train improved model
    model_data = train_production_model(X, y, data_folder)
    print("\nProduction training completed successfully!")
