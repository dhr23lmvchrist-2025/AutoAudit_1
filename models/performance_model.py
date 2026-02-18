"""
Machine Learning Model for Performance Prediction and Analysis
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os


class PerformancePredictor:
    """
    ML model to predict AutoAudit performance based on system characteristics
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}

    def prepare_features(self, df):
        """Prepare features for model training"""

        # Create feature matrix
        features = pd.DataFrame()

        # Encode categorical variables
        for col in ['system_type', 'data_volume', 'principle', 'category']:
            if col in df.columns:
                le = LabelEncoder()
                features[f'{col}_encoded'] = le.fit_transform(df[col])
                self.label_encoders[col] = le

        # Add numeric features
        numeric_cols = ['development_year']
        for col in numeric_cols:
            if col in df.columns:
                features[col] = df[col]

        # Add interaction features
        if 'system_type_encoded' in features.columns and 'principle_encoded' in features.columns:
            features['type_principle_interaction'] = (
                    features['system_type_encoded'] * features['principle_encoded']
            )

        return features

    def train_accuracy_model(self, df):
        """Train model to predict proposed accuracy"""

        # Prepare features and target
        X = self.prepare_features(df)
        y = df['proposed_accuracy']

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['accuracy'] = scaler

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=self.random_state
        )

        # Train Random Forest model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)

        # Train Gradient Boosting model
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=self.random_state
        )
        gb_model.fit(X_train, y_train)

        # Store models
        self.models['accuracy_rf'] = rf_model
        self.models['accuracy_gb'] = gb_model

        # Evaluate
        y_pred_rf = rf_model.predict(X_test)
        y_pred_gb = gb_model.predict(X_test)

        results = {
            'rf': {
                'r2': r2_score(y_test, y_pred_rf),
                'mse': mean_squared_error(y_test, y_pred_rf),
                'mae': mean_absolute_error(y_test, y_pred_rf)
            },
            'gb': {
                'r2': r2_score(y_test, y_pred_gb),
                'mse': mean_squared_error(y_test, y_pred_gb),
                'mae': mean_absolute_error(y_test, y_pred_gb)
            }
        }

        # Feature importance
        self.feature_importance['accuracy'] = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        return results

    def train_improvement_model(self, df):
        """Train model to predict improvement (proposed - baseline)"""

        # Prepare features
        X = self.prepare_features(df)
        y = df['improvement']

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['improvement'] = scaler

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=self.random_state
        )

        # Train model
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            random_state=self.random_state
        )
        model.fit(X_train, y_train)

        self.models['improvement'] = model

        # Evaluate
        y_pred = model.predict(X_test)

        results = {
            'r2': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred)
        }

        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        results['cv_r2_mean'] = cv_scores.mean()
        results['cv_r2_std'] = cv_scores.std()

        # Feature importance
        self.feature_importance['improvement'] = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        return results

    def predict_performance(self, system_data):
        """Predict performance for new systems"""

        # Prepare features
        X = self.prepare_features(system_data)

        predictions = {}

        # Make predictions with all models
        if 'accuracy_rf' in self.models:
            X_scaled = self.scalers['accuracy'].transform(X)
            predictions['accuracy_rf'] = self.models['accuracy_rf'].predict(X_scaled)
            predictions['accuracy_gb'] = self.models['accuracy_gb'].predict(X_scaled)
            predictions['accuracy_ensemble'] = (
                                                       predictions['accuracy_rf'] + predictions['accuracy_gb']
                                               ) / 2

        if 'improvement' in self.models:
            X_scaled = self.scalers['improvement'].transform(X)
            predictions['improvement'] = self.models['improvement'].predict(X_scaled)

        return predictions

    def save_models(self, path='./models/trained_model.pkl'):
        """Save trained models to disk"""

        os.makedirs(os.path.dirname(path), exist_ok=True)

        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'label_encoders': self.label_encoders,
            'feature_importance': self.feature_importance,
            'random_state': self.random_state
        }

        joblib.dump(model_data, path)
        print(f"Models saved to {path}")

    def load_models(self, path='./models/trained_model.pkl'):
        """Load trained models from disk"""

        if os.path.exists(path):
            model_data = joblib.load(path)
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.label_encoders = model_data['label_encoders']
            self.feature_importance = model_data['feature_importance']
            self.random_state = model_data['random_state']
            print(f"Models loaded from {path}")
        else:
            print(f"No model found at {path}")