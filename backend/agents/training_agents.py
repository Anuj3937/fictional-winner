from google.adk.agents import LlmAgent, ParallelAgent
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, r2_score, classification_report, mean_squared_error
import joblib
from pathlib import Path
import json
import asyncio

async def train_xgboost_model(feature_result: Dict[str, Any]) -> Dict[str, Any]:
    """Train optimized XGBoost model"""
    return await _train_model_async("xgboost", feature_result)

async def train_lightgbm_model(feature_result: Dict[str, Any]) -> Dict[str, Any]:
    """Train optimized LightGBM model"""
    return await _train_model_async("lightgbm", feature_result)

async def train_random_forest_model(feature_result: Dict[str, Any]) -> Dict[str, Any]:
    """Train optimized Random Forest model"""
    return await _train_model_async("random_forest", feature_result)

async def _train_model_async(model_type: str, feature_result: Dict[str, Any]) -> Dict[str, Any]:
    """Async model training with optimized parameters"""
    
    try:
        # Load engineered data
        data_path = feature_result.get("engineered_data_path")
        if not data_path:
            return {"status": "error", "message": "No engineered data path provided"}
            
        df = pd.read_csv(data_path)
        target_col = feature_result.get("target_column")
        
        if not target_col or target_col not in df.columns:
            return {"status": "error", "message": "Target column not found"}
        
        # Prepare features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Determine task type
        is_classification = len(y.unique()) <= 20 and y.dtype in ['object', 'int64', 'bool']
        task_type = "classification" if is_classification else "regression"
        
        # Split data with stratification for classification
        stratify = y if is_classification else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
        
        # Initialize model with optimized parameters
        model = _get_optimized_model(model_type, task_type, X_train.shape)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy' if is_classification else 'r2')
        
        # Additional metrics
        y_pred = model.predict(X_test)
        additional_metrics = _calculate_metrics(y_test, y_pred, task_type)
        
        # Feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_.tolist()))
        
        # Save model
        model_path = Path("models") / f"{model_type}_{task_type}_optimized.pkl"
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump(model, model_path)
        
        return {
            "status": "success",
            "model_type": model_type,
            "task_type": task_type,
            "train_score": float(train_score),
            "test_score": float(test_score),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "additional_metrics": additional_metrics,
            "feature_importance": feature_importance,
            "model_path": str(model_path),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "features_used": len(X.columns)
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "model_type": model_type,
            "message": f"Model training failed: {str(e)}"
        }

def _get_optimized_model(model_type: str, task_type: str, data_shape: tuple):
    """Get model with optimized hyperparameters based on data characteristics"""
    
    n_samples, n_features = data_shape
    
    if model_type == "xgboost":
        params = {
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # Optimize based on data size
        if n_samples < 1000:
            params.update({
                'n_estimators': 100,
                'max_depth': 4,
                'learning_rate': 0.1
            })
        elif n_samples < 10000:
            params.update({
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05
            })
        else:
            params.update({
                'n_estimators': 300,
                'max_depth': 8,
                'learning_rate': 0.03
            })
        
        if task_type == "classification":
            params['eval_metric'] = 'logloss'
            return xgb.XGBClassifier(**params)
        else:
            return xgb.XGBRegressor(**params)
            
    elif model_type == "lightgbm":
        params = {
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'force_col_wise': True  # Optimize for speed
        }
        
        if n_samples < 1000:
            params.update({
                'n_estimators': 100,
                'max_depth': 4,
                'learning_rate': 0.1
            })
        else:
            params.update({
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05
            })
        
        if task_type == "classification":
            return lgb.LGBMClassifier(**params)
        else:
            return lgb.LGBMRegressor(**params)
            
    else:  # random_forest
        params = {
            'random_state': 42,
            'n_jobs': -1
        }
        
        if n_samples < 1000:
            params.update({
                'n_estimators': 100,
                'max_depth': 10
            })
        else:
            params.update({
                'n_estimators': 200,
                'max_depth': 15
            })
        
        if task_type == "classification":
            return RandomForestClassifier(**params)
        else:
            return RandomForestRegressor(**params)

def _calculate_metrics(y_true, y_pred, task_type: str) -> Dict[str, float]:
    """Calculate additional performance metrics"""
    
    metrics = {}
    
    try:
        if task_type == "classification":
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            # Handle binary vs multiclass
            if len(np.unique(y_true)) == 2:
                avg_method = 'binary'
            else:
                avg_method = 'weighted'
            
            metrics = {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, average=avg_method, zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, average=avg_method, zero_division=0)),
                'f1_score': float(f1_score(y_true, y_pred, average=avg_method, zero_division=0))
            }
        else:
            from sklearn.metrics import mean_absolute_error
            
            metrics = {
                'r2_score': float(r2_score(y_true, y_pred)),
                'mse': float(mean_squared_error(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'mae': float(mean_absolute_error(y_true, y_pred))
            }
            
    except Exception as e:
        metrics['error'] = str(e)
    
    return metrics

# Individual Model Training Agents
xgboost_trainer = LlmAgent(
    name="xgboost_trainer",
    model="gemini-2.0-flash",
    instruction="""You are an XGBoost specialist. Train optimized XGBoost models:

1. Use optimized hyperparameters based on dataset size
2. Handle both classification and regression tasks
3. Implement proper cross-validation
4. Extract feature importance
5. Save trained models for later use

Use train_xgboost_model tool to train high-performance XGBoost models.""",
    
    tools=[train_xgboost_model]
)

lightgbm_trainer = LlmAgent(
    name="lightgbm_trainer", 
    model="gemini-2.0-flash",
    instruction="""You are a LightGBM specialist. Train optimized LightGBM models:

1. Use efficient LightGBM parameters for speed and accuracy
2. Handle both classification and regression tasks  
3. Optimize for large datasets with fast training
4. Extract feature importance and performance metrics
5. Save trained models with proper versioning

Use train_lightgbm_model tool to train high-performance LightGBM models.""",
    
    tools=[train_lightgbm_model]
)

random_forest_trainer = LlmAgent(
    name="random_forest_trainer",
    model="gemini-2.0-flash", 
    instruction="""You are a Random Forest specialist. Train robust Random Forest models:

1. Use ensemble approach with optimal tree parameters
2. Handle both classification and regression robustly
3. Provide stable performance across different datasets
4. Extract feature importance and out-of-bag scores
5. Save trained models with ensemble metadata

Use train_random_forest_model tool to train reliable Random Forest models.""",
    
    tools=[train_random_forest_model]
)

# Parallel Model Training Agent
parallel_model_training = ParallelAgent(
    name="parallel_model_training",
    sub_agents=[xgboost_trainer, lightgbm_trainer, random_forest_trainer]
)
