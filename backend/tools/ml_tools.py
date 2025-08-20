import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, classification_report, confusion_matrix, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, List
import json
import asyncio
import optuna
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

async def train_model_with_monitoring(dataset_path: str, model_type: str, task_type: str, hyperparams: Dict = None) -> Dict[str, Any]:
    """Train model with comprehensive monitoring and validation"""
    
    try:
        logger.info(f"Starting {model_type} training for {task_type}")
        
        # Load and prepare data
        df = pd.read_csv(dataset_path)
        target_col = detect_target_column(df)
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Validate data
        if X.empty or y.empty:
            raise ValueError("Empty features or target data")
        
        # Auto-detect task type if not provided
        if task_type == "auto":
            task_type = "classification" if len(y.unique()) <= 20 and y.dtype in ['object', 'int64', 'bool'] else "regression"
        
        # Split data with stratification for classification
        stratify = y if task_type == "classification" and len(y.unique()) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
        
        # Initialize model with hyperparameters
        model = _initialize_model(model_type, task_type, X_train.shape, hyperparams)
        
        # Train model with error handling
        training_start = pd.Timestamp.now()
        model.fit(X_train, y_train)
        training_time = (pd.Timestamp.now() - training_start).total_seconds()
        
        # Comprehensive evaluation
        evaluation_results = await _evaluate_model_comprehensive(
            model, X_train, X_test, y_train, y_test, task_type
        )
        
        # Feature importance analysis
        feature_importance = _extract_feature_importance(model, X.columns.tolist())
        
        # Model validation
        validation_results = await _validate_model_quality(evaluation_results, task_type)
        
        # Save model
        model_filename = f"{model_type}_{task_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        model_path = Path("models") / model_filename
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump(model, model_path)
        
        # Compile results
        results = {
            "status": "success",
            "model_info": {
                "type": model_type,
                "task_type": task_type,
                "algorithm": model.__class__.__name__,
                "hyperparameters": hyperparams or "default",
                "training_time_seconds": training_time
            },
            "data_info": {
                "total_samples": len(df),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "features": len(X.columns),
                "target_distribution": y.value_counts().to_dict() if task_type == "classification" else {"mean": float(y.mean()), "std": float(y.std())}
            },
            "performance": evaluation_results,
            "feature_importance": feature_importance,
            "validation": validation_results,
            "model_path": str(model_path),
            "artifacts": {
                "confusion_matrix": evaluation_results.get("confusion_matrix"),
                "classification_report": evaluation_results.get("classification_report"),
                "cv_scores": evaluation_results.get("cv_scores"),
                "prediction_examples": evaluation_results.get("prediction_examples")
            }
        }
        
        logger.info(f"{model_type} training completed successfully. Score: {evaluation_results['test_score']:.3f}")
        return results
        
    except Exception as e:
        logger.error(f"{model_type} training failed: {str(e)}")
        return {
            "status": "error",
            "model_type": model_type,
            "task_type": task_type,
            "error": str(e),
            "message": f"Model training failed: {str(e)}"
        }

def _initialize_model(model_type: str, task_type: str, data_shape: Tuple[int, int], hyperparams: Dict = None):
    """Initialize model with optimal parameters"""
    
    n_samples, n_features = data_shape
    params = hyperparams or {}
    
    # Base parameters for all models
    base_params = {
        'random_state': 42,
        'n_jobs': -1
    }
    
    if model_type == "xgboost":
        xgb_params = {
            **base_params,
            'verbosity': 0,
            'eval_metric': 'logloss' if task_type == "classification" else 'rmse'
        }
        
        # Auto-scale parameters based on data size
        if n_samples < 1000:
            xgb_params.update({'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1})
        elif n_samples < 10000:
            xgb_params.update({'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05})
        else:
            xgb_params.update({'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.03})
        
        xgb_params.update(params)  # Override with provided hyperparams
        
        if task_type == "classification":
            return xgb.XGBClassifier(**xgb_params)
        else:
            return xgb.XGBRegressor(**xgb_params)
            
    elif model_type == "lightgbm":
        lgb_params = {
            **base_params,
            'verbose': -1,
            'force_col_wise': True
        }
        
        if n_samples < 1000:
            lgb_params.update({'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1})
        else:
            lgb_params.update({'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05})
        
        lgb_params.update(params)
        
        if task_type == "classification":
            return lgb.LGBMClassifier(**lgb_params)
        else:
            return lgb.LGBMRegressor(**lgb_params)
            
    elif model_type == "random_forest":
        rf_params = {**base_params}
        
        if n_samples < 1000:
            rf_params.update({'n_estimators': 100, 'max_depth': 10})
        else:
            rf_params.update({'n_estimators': 200, 'max_depth': 15})
        
        rf_params.update(params)
        
        if task_type == "classification":
            return RandomForestClassifier(**rf_params)
        else:
            return RandomForestRegressor(**rf_params)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

async def _evaluate_model_comprehensive(model, X_train, X_test, y_train, y_test, task_type: str) -> Dict[str, Any]:
    """Comprehensive model evaluation with multiple metrics"""
    
    try:
        # Basic predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Cross-validation
        cv_folds = 5 if len(X_train) >= 100 else 3
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) if task_type == "classification" else KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='accuracy' if task_type == "classification" else 'r2')
        
        # Basic scores
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        evaluation = {
            "train_score": float(train_score),
            "test_score": float(test_score),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "cv_scores": cv_scores.tolist(),
            "overfitting_score": float(train_score - test_score)  # Positive means overfitting
        }
        
        if task_type == "classification":
            # Classification-specific metrics
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            
            # Handle binary vs multiclass
            average = 'binary' if len(np.unique(y_test)) == 2 else 'weighted'
            
            evaluation.update({
                "accuracy": float(accuracy_score(y_test, y_pred_test)),
                "precision": float(precision_score(y_test, y_pred_test, average=average, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred_test, average=average, zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred_test, average=average, zero_division=0)),
                "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
                "classification_report": classification_report(y_test, y_pred_test, output_dict=True)
            })
            
            # ROC AUC for binary classification
            if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    evaluation["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))
                except:
                    pass
        else:
            # Regression-specific metrics
            evaluation.update({
                "r2_score": float(r2_score(y_test, y_pred_test)),
                "mse": float(mean_squared_error(y_test, y_pred_test)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                "mae": float(mean_absolute_error(y_test, y_pred_test)),
                "mape": float(np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100) if (y_test != 0).all() else None
            })
        
        # Prediction examples for inspection
        n_examples = min(10, len(X_test))
        indices = np.random.choice(len(X_test), n_examples, replace=False)
        
        evaluation["prediction_examples"] = [
            {
                "actual": float(y_test.iloc[i]) if hasattr(y_test, 'iloc') else float(y_test[i]),
                "predicted": float(y_pred_test[i]),
                "index": int(indices[j])
            }
            for j, i in enumerate(indices)
        ]
        
        return evaluation
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        return {
            "train_score": 0.0,
            "test_score": 0.0,
            "cv_mean": 0.0,
            "cv_std": 1.0,
            "error": str(e)
        }

def _extract_feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
    """Extract and normalize feature importance"""
    
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            # Normalize to sum to 1
            importance_normalized = importance / importance.sum() if importance.sum() > 0 else importance
            
            # Create sorted dictionary
            importance_dict = dict(zip(feature_names, importance_normalized))
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            return {k: float(v) for k, v in sorted_importance.items()}
        
        elif hasattr(model, 'coef_'):  # Linear models
            importance = np.abs(model.coef_).flatten()
            importance_normalized = importance / importance.sum() if importance.sum() > 0 else importance
            importance_dict = dict(zip(feature_names, importance_normalized))
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            return {k: float(v) for k, v in sorted_importance.items()}
        
        else:
            return {}
            
    except Exception as e:
        logger.warning(f"Could not extract feature importance: {str(e)}")
        return {}

async def _validate_model_quality(evaluation: Dict[str, Any], task_type: str) -> Dict[str, Any]:
    """Validate model quality against industry standards"""
    
    validation = {
        "is_acceptable": False,
        "quality_score": 0.0,
        "issues": [],
        "recommendations": []
    }
    
    try:
        test_score = evaluation.get("test_score", 0)
        cv_mean = evaluation.get("cv_mean", 0)
        cv_std = evaluation.get("cv_std", 1)
        overfitting_score = evaluation.get("overfitting_score", 0)
        
        quality_score = 0.0
        
        # Score thresholds
        if task_type == "classification":
            if test_score >= 0.9:
                quality_score += 0.4
            elif test_score >= 0.8:
                quality_score += 0.3
            elif test_score >= 0.7:
                quality_score += 0.2
            else:
                validation["issues"].append(f"Low accuracy ({test_score:.3f})")
                validation["recommendations"].append("Try different algorithms or feature engineering")
        else:  # regression
            if test_score >= 0.9:
                quality_score += 0.4
            elif test_score >= 0.7:
                quality_score += 0.3
            elif test_score >= 0.5:
                quality_score += 0.2
            else:
                validation["issues"].append(f"Low R² score ({test_score:.3f})")
                validation["recommendations"].append("Consider feature engineering or different algorithms")
        
        # Cross-validation stability
        if cv_std <= 0.05:
            quality_score += 0.3
        elif cv_std <= 0.1:
            quality_score += 0.2
        else:
            validation["issues"].append(f"High CV variance ({cv_std:.3f})")
            validation["recommendations"].append("Model is unstable - try regularization or more data")
        
        # Overfitting check
        if abs(overfitting_score) <= 0.05:
            quality_score += 0.3
        elif abs(overfitting_score) <= 0.1:
            quality_score += 0.1
        else:
            validation["issues"].append(f"Potential overfitting (train-test gap: {overfitting_score:.3f})")
            validation["recommendations"].append("Apply regularization or cross-validation")
        
        validation["quality_score"] = quality_score
        validation["is_acceptable"] = quality_score >= 0.6
        
        if validation["is_acceptable"]:
            validation["recommendations"].append("Model quality is acceptable for production use")
        
        return validation
        
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        validation["issues"].append(f"Validation error: {str(e)}")
        return validation

def detect_target_column(df: pd.DataFrame) -> str:
    """Intelligent target column detection"""
    
    # Priority list of common target column names
    target_candidates = [
        'target', 'label', 'class', 'y', 'outcome', 'prediction', 'result',
        'response', 'dependent', 'output', 'score', 'rating', 'price', 'value'
    ]
    
    # Check for exact matches (case-insensitive)
    for col in df.columns:
        if col.lower() in target_candidates:
            return col
    
    # Check for partial matches
    for col in df.columns:
        col_lower = col.lower()
        for candidate in target_candidates:
            if candidate in col_lower or col_lower in candidate:
                return col
    
    # If no matches, return the last column (common convention)
    return df.columns[-1] if len(df.columns) > 0 else None

async def optimize_hyperparameters_advanced(dataset_path: str, model_type: str, task_type: str, n_trials: int = 50) -> Dict[str, Any]:
    """Advanced hyperparameter optimization using Optuna"""
    
    try:
        logger.info(f"Starting hyperparameter optimization for {model_type}")
        
        # Load data
        df = pd.read_csv(dataset_path)
        target_col = detect_target_column(df)
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Split for optimization
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        def objective(trial):
            try:
                if model_type == "xgboost":
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                        'gamma': trial.suggest_float('gamma', 0, 5),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                        'random_state': 42,
                        'n_jobs': -1,
                        'verbosity': 0
                    }
                    
                    if task_type == "classification":
                        params['eval_metric'] = 'logloss'
                        model = xgb.XGBClassifier(**params)
                    else:
                        params['eval_metric'] = 'rmse'
                        model = xgb.XGBRegressor(**params)
                        
                elif model_type == "lightgbm":
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                        'random_state': 42,
                        'n_jobs': -1,
                        'verbose': -1
                    }
                    
                    if task_type == "classification":
                        model = lgb.LGBMClassifier(**params)
                    else:
                        model = lgb.LGBMRegressor(**params)
                        
                elif model_type == "random_forest":
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 5, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                        'random_state': 42,
                        'n_jobs': -1
                    }
                    
                    if task_type == "classification":
                        model = RandomForestClassifier(**params)
                    else:
                        model = RandomForestRegressor(**params)
                
                # Train and evaluate
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                
                return score
                
            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                return 0.0
        
        # Run optimization
        study = optuna.create_study(direction='maximize', study_name=f"{model_type}_optimization")
        study.optimize(objective, n_trials=n_trials, timeout=1800)  # 30 minutes max
        
        # Train final model with best parameters
        best_params = study.best_params
        final_model = _initialize_model(model_type, task_type, X.shape, best_params)
        final_model.fit(X_train, y_train)
        final_score = final_model.score(X_val, y_val)
        
        # Save optimized model
        model_filename = f"{model_type}_{task_type}_optimized.pkl"
        model_path = Path("models") / model_filename
        joblib.dump(final_model, model_path)
        
        return {
            "status": "success",
            "model_type": model_type,
            "optimization_trials": len(study.trials),
            "best_params": best_params,
            "best_score": study.best_value,
            "final_score": final_score,
            "improvement": final_score - study.trials[0].value if study.trials else 0,
            "model_path": str(model_path),
            "optimization_history": [trial.value for trial in study.trials if trial.value is not None]
        }
        
    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {str(e)}")
        return {
            "status": "error",
            "model_type": model_type,
            "error": str(e)
        }
