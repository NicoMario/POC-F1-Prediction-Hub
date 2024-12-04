import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import logging
from typing import Dict, List, Union, Optional, Tuple
import joblib
from pathlib import Path


class F1Model:
    def __init__(self):
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.feature_importances = None
        self.preprocessor = None
        self.is_trained = False
        self.used_features = None
        
    def get_model_state(self) -> dict:
        """Get the current state of the model"""
        return {
            'is_trained': self.is_trained,
            'feature_importances': self.feature_importances,
            'used_features': self.used_features,
            'model': self.model,
            'preprocessor': self.preprocessor
        }
        
    def set_model_state(self, state: dict):
        """Restore the model state"""
        self.is_trained = state['is_trained']
        self.feature_importances = state['feature_importances']
        self.used_features = state['used_features']
        self.model = state['model']
        self.preprocessor = state['preprocessor']
        
    def train(self, training_data: pd.DataFrame, selected_features: Dict[str, bool]) -> dict:
        """Train the model to predict race positions using selected features"""
        try:
            if training_data.empty:
                raise ValueError("No training data provided")
            
            # Initialize feature lists
            numerical_features = ['grid']
            categorical_features = []
            
            # Add selected features
            if selected_features.get('Constructor', False):
                categorical_features.append('constructor')
            if selected_features.get('Circuit', False):
                categorical_features.append('circuit')
            if selected_features.get('Round', False):
                numerical_features.append('round')
            
            # Store used features
            self.used_features = {'numerical': numerical_features, 'categorical': categorical_features}
            
            # Create preprocessing steps
            numeric_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            
            # Create preprocessor
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ],
                remainder='drop'
            )
            
            # Create pipeline
            self.model = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            
            # Prepare features and target
            X = training_data[numerical_features + categorical_features]
            y = training_data['position']
            
            # Train model
            self.model.fit(X, y)
            self.is_trained = True
            
            # Calculate scores
            train_score = self.model.score(X, y)
            test_size = 0.2
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            test_score = self.model.score(X_test, y_test)
            
            return {
                'training_score': train_score,
                'test_score': test_score,
                'data_points': len(training_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error in training: {str(e)}")
            self.is_trained = False
            raise
            
    def predict(self, input_data: dict) -> dict:
        """Make a prediction for a single race entry"""
        if not self.is_trained or self.model is None:
            raise ValueError("Please train the model first")
            
        try:
            # Create features dictionary based on used features
            features = {}
            
            # Add numerical features
            for feature in self.used_features['numerical']:
                if feature == 'grid':
                    features[feature] = input_data['grid_position']
                else:
                    features[feature] = input_data.get(feature, 0)
                    
            # Add categorical features
            for feature in self.used_features['categorical']:
                features[feature] = input_data.get(feature, 'Unknown')
            
            # Convert to DataFrame
            features_df = pd.DataFrame([features])
            
            # Make prediction
            prediction = self.model.predict(features_df)[0]
            
            return {
                'predicted_position': round(prediction, 1),
                'confidence': 0.8,
                'input_features': features
            }
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise

    def _calculate_confidence(self, X: pd.DataFrame) -> float:
        """
        Calculate prediction confidence based on various factors
        
        Args:
            X: Input features DataFrame
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Get prediction probabilities
            prediction_proba = self.model.named_steps['regressor'].predict_proba(
                self.preprocessor.transform(X)
            ) if hasattr(self.model.named_steps['regressor'], 'predict_proba') else None
            
            if prediction_proba is not None:
                # Use the maximum probability as the base confidence
                base_confidence = np.max(prediction_proba[0])
            else:
                # Fallback to a simpler confidence metric
                base_confidence = 0.8
            
            # Adjust confidence based on feature importance weights
            feature_weights = []
            for feature in X.columns:
                if feature in self.feature_importances:
                    feature_weights.append(self.feature_importances[feature])
            
            # Calculate weighted confidence
            if feature_weights:
                weighted_confidence = base_confidence * np.mean(feature_weights)
                return min(max(weighted_confidence, 0), 1)  # Ensure between 0 and 1
            
            return base_confidence
            
        except Exception as e:
            self.logger.warning(f"Error calculating confidence: {str(e)}")
            return 0.7  # Return default confidence

    def _get_prediction_interval(self, prediction: float, confidence: float) -> Dict[str, float]:
        """
        Calculate prediction interval based on confidence
        
        Args:
            prediction: The predicted value
            confidence: Confidence score
            
        Returns:
            Dictionary with lower and upper bounds
        """
        # Calculate interval width based on confidence
        interval_width = (1 - confidence) * 5  # Adjust multiplier as needed
        
        return {
            'lower_bound': max(1, round(prediction - interval_width, 1)),  # Position can't be less than 1
            'upper_bound': min(20, round(prediction + interval_width, 1))  # Position can't be more than 20
        }

    def _save_model(self, filename: str = 'latest_model.joblib'):
        """
        Save the trained model and preprocessor to disk
        
        Args:
            filename: Name of the file to save the model to
        """
        try:
            model_path = self.model_dir / filename
            model_data = {
                'model': self.model,
                'feature_importances': self.feature_importances,
                'preprocessor': self.preprocessor
            }
            joblib.dump(model_data, model_path)
            self.logger.info(f"Model saved successfully to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")

    def load_model(self, filename: str = 'latest_model.joblib') -> bool:
        """
        Load a trained model from disk
        
        Args:
            filename: Name of the file to load the model from
            
        Returns:
            Boolean indicating if loading was successful
        """
        try:
            model_path = self.model_dir / filename
            if not model_path.exists():
                self.logger.warning(f"No saved model found at {model_path}")
                return False
            
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.feature_importances = model_data['feature_importances']
            self.preprocessor = model_data['preprocessor']
            
            self.logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False

    def get_feature_importance_analysis(self) -> Dict[str, float]:
        """
        Get detailed feature importance analysis
        
        Returns:
            Dictionary of features and their importance scores
        """
        if self.feature_importances is None:
            raise ValueError("Model not trained yet")
        
        # Sort features by importance
        sorted_features = dict(sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return sorted_features

    def get_model_diagnostics(self) -> Dict:
        """
        Get detailed model diagnostics and performance metrics
        
        Returns:
            Dictionary containing various model diagnostics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        try:
            regressor = self.model.named_steps['regressor']
            
            diagnostics = {
                'model_type': type(regressor).__name__,
                'n_features': len(self.feature_importances),
                'parameters': regressor.get_params(),
                'feature_importances': self.get_feature_importance_analysis()
            }
            
            # Add GBM-specific diagnostics if applicable
            if isinstance(regressor, GradientBoostingRegressor):
                diagnostics.update({
                    'n_estimators': regressor.n_estimators,
                    'learning_rate': regressor.learning_rate,
                    'max_depth': regressor.max_depth,
                    'train_score': regressor.train_score_.tolist()  # Training deviance
                })
            
            return diagnostics
            
        except Exception as e:
            self.logger.error(f"Error getting model diagnostics: {str(e)}")
            return {}

    def evaluate_model_performance(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model performance on test data
        
        Args:
            X_test: Test features
            y_test: True test values
            
        Returns:
            Dictionary containing various performance metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        try:
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate various metrics
            metrics = {
                'r2_score': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': np.mean(np.abs(y_test - y_pred)),
                'median_ae': np.median(np.abs(y_test - y_pred)),
                'position_accuracy': np.mean(np.abs(y_test - y_pred) <= 1)  # Within 1 position
            }
            
            # Calculate position-specific accuracies
            for threshold in [1, 2, 3]:
                metrics[f'within_{threshold}_positions'] = \
                    np.mean(np.abs(y_test - y_pred) <= threshold)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model performance: {str(e)}")
            return {}

    def analyze_prediction_errors(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Analyze prediction errors to identify patterns
        
        Args:
            X_test: Test features
            y_test: True test values
            
        Returns:
            Dictionary containing error analysis
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        try:
            # Make predictions
            y_pred = self.model.predict(X_test)
            errors = y_test - y_pred
            
            # Analyze errors
            analysis = {
                'error_distribution': {
                    'mean': np.mean(errors),
                    'std': np.std(errors),
                    'skew': pd.Series(errors).skew(),
                    'kurtosis': pd.Series(errors).kurtosis()
                },
                'error_percentiles': {
                    '25%': np.percentile(errors, 25),
                    '50%': np.percentile(errors, 50),
                    '75%': np.percentile(errors, 75)
                },
                'outliers': len(errors[np.abs(errors) > 2 * np.std(errors)])
            }
            
            # Add feature-specific error analysis
            feature_errors = {}
            for feature in X_test.columns:
                if feature in self.feature_importances:
                    correlation = np.corrcoef(X_test[feature], errors)[0, 1]
                    feature_errors[feature] = correlation
            
            analysis['feature_error_correlations'] = feature_errors
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing prediction errors: {str(e)}")
            return {}