from google.adk.agents import LlmAgent
from typing import Dict, Any, List
import optuna
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import asyncio
import json

async def optimize_hyperparameters(training_results: Dict[str, Any]) -> Dict[str, Any]:
    """Advanced hyperparameter optimization using Optuna"""
    
    try:
        # Find the best performing model to optimize
        best_model_info = _find_best_model(training_results)
        if not best_model_info:
            return {"status": "error", "message": "No successful models found for optimization"}
        
        model_type = best_model_info["model_type"]
        model_path = best_model_info["model_path"]
        
        # Get dataset info
        base_path = Path(model_path).parent.parent / "temp" / "engineered_dataset.csv"
        if not base_path.exists():
            return {"status": "error", "message": "Engineered dataset not found for optimization"}
        
        df = pd.read_csv(base_path)
        target_col = _detect_target_column(df)
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Determine task type
        is_classification = len(y.unique()) <= 20
        task_type = "classification" if is_classification else "regression"
        
        # Run optimization
        optimization_result = await _optimize_model_async(model_type, X, y, task_type)
        
        # Retrain model with best parameters
        retrained_result = await _retrain_with_best_params(
            model_type, X, y, task_type, optimization_result["best_params"]
        )
        
        return {
            "status": "success",
            "optimized_model": model_type,
            "original_score": best_model_info["test_score"],
            "optimized_score": retrained_result["test_score"],
            "improvement": retrained_result["test_score"] - best_model_info["test_score"],
            "best_params": optimization_result["best_params"],
            "optimization_trials": optimization_result["n_trials"],
            "optimized_model_path": retrained_result["model_path"]
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Hyperparameter optimization failed: {str(e)}"}

async def _optimize_model_async(model_type: str, X, y, task_type: str) -> Dict[str, Any]:
    """Async hyperparameter optimization using Optuna"""
    
    from sklearn.model_selection import cross_val_score
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    def objective(trial):
        try:
            if model_type == "xgboost":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42,
                    'n_jobs': -1
                }
                
                if task_type == "classification":
                    model = xgb.XGBClassifier(**params, eval_metric='logloss')
                else:
                    model = xgb.XGBRegressor(**params)
                    
            elif model_type == "lightgbm":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1
                }
                
                if task_type == "classification":
                    model = lgb.LGBMClassifier(**params)
                else:
                    model = lgb.LGBMRegressor(**params)
                    
            else:  # random_forest
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': 42,
                    'n_jobs': -1
                }
                
                if task_type == "classification":
                    model = RandomForestClassifier(**params)
                else:
                    model = RandomForestRegressor(**params)
            
            # Cross-validation score
            scoring = 'accuracy' if task_type == "classification" else 'r2'
            scores = cross_val_score(model, X, y, cv=3, scoring=scoring)
            
            return scores.mean()
            
        except Exception as e:
            return 0.0  # Return poor score on error
    
    # Create and run study
    study = optuna.create_study(direction='maximize')
    
    # Optimize with timeout to prevent hanging
    study.optimize(objective, n_trials=30, timeout=600)  # 10 minutes max
    
    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
        "n_trials": len(study.trials)
    }

async def _retrain_with_best_params(model_type: str, X, y, task_type: str, best_params: Dict) -> Dict[str, Any]:
    """Retrain model with optimized parameters"""
    
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create model with best parameters
    if model_type == "xgboost":
        best_params.update({'random_state': 42, 'n_jobs': -1})
        if task_type == "classification":
            best_params['eval_metric'] = 'logloss'
            model = xgb.XGBClassifier(**best_params)
        else:
            model = xgb.XGBRegressor(**best_params)
            
    elif model_type == "lightgbm":
        best_params.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})
        if task_type == "classification":
            model = lgb.LGBMClassifier(**best_params)
        else:
            model = lgb.LGBMRegressor(**best_params)
            
    else:  # random_forest
        best_params.update({'random_state': 42, 'n_jobs': -1})
        if task_type == "classification":
            model = RandomForestClassifier(**best_params)
        else:
            model = RandomForestRegressor(**best_params)
    
    # Train and evaluate
    model.fit(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    # Save optimized model
    model_path = Path("models") / f"{model_type}_optimized_{task_type}.pkl"
    joblib.dump(model, model_path)
    
    return {
        "test_score": test_score,
        "model_path": str(model_path)
    }

async def create_ensemble_model(training_results: Dict[str, Any], optimization_result: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create ensemble model from trained models"""
    
    try:
        # Collect successful models
        successful_models = []
        model_paths = []
        model_scores = []
        
        # Add training results
        if isinstance(training_results, dict):
            for agent_result in training_results.values():
                if isinstance(agent_result, dict) and agent_result.get("status") == "success":
                    successful_models.append(agent_result)
                    model_paths.append(agent_result["model_path"])
                    model_scores.append(agent_result["test_score"])
        
        # Add optimized model if available
        if optimization_result and optimization_result.get("status") == "success":
            optimized_path = optimization_result.get("optimized_model_path")
            optimized_score = optimization_result.get("optimized_score")
            if optimized_path and optimized_score:
                model_paths.append(optimized_path)
                model_scores.append(optimized_score)
        
        if len(model_paths) < 2:
            return {"status": "error", "message": "Need at least 2 models for ensemble"}
        
        # Load dataset for ensemble evaluation
        base_path = Path("temp") / "engineered_dataset.csv"
        df = pd.read_csv(base_path)
        target_col = _detect_target_column(df)
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Create voting ensemble
        ensemble_result = await _create_voting_ensemble(model_paths, X, y)
        
        return {
            "status": "success",
            "ensemble_type": "voting",
            "base_models": len(model_paths),
            "individual_scores": model_scores,
            "ensemble_score": ensemble_result["score"],
            "improvement_over_best": ensemble_result["score"] - max(model_scores),
            "ensemble_model_path": ensemble_result["model_path"]
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Ensemble creation failed: {str(e)}"}

async def _create_voting_ensemble(model_paths: List[str], X, y) -> Dict[str, Any]:
    """Create voting ensemble from model paths"""
    
    from sklearn.ensemble import VotingClassifier, VotingRegressor
    from sklearn.model_selection import train_test_split
    
    # Load models
    models = []
    model_names = []
    
    for i, path in enumerate(model_paths):
        model = joblib.load(path)
        model_name = f"model_{i}"
        models.append((model_name, model))
        model_names.append(model_name)
    
    # Determine task type
    is_classification = len(y.unique()) <= 20
    
    # Create ensemble
    if is_classification:
        ensemble = VotingClassifier(estimators=models, voting='soft')
    else:
        ensemble = VotingRegressor(estimators=models)
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    score = ensemble.score(X_test, y_test)
    
    # Save ensemble
    ensemble_path = Path("models") / "ensemble_voting.pkl"
    joblib.dump(ensemble, ensemble_path)
    
    return {
        "score": score,
        "model_path": str(ensemble_path)
    }

def _find_best_model(training_results: Dict[str, Any]) -> Dict[str, Any]:
    """Find best performing model from training results"""
    
    best_model = None
    best_score = -float('inf')
    
    if isinstance(training_results, dict):
        for result in training_results.values():
            if isinstance(result, dict) and result.get("status") == "success":
                score = result.get("test_score", 0)
                if score > best_score:
                    best_score = score
                    best_model = result
    
    return best_model

def _detect_target_column(df: pd.DataFrame) -> str:
    """Detect target column"""
    possible_targets = ['target', 'label', 'class', 'y', 'outcome']
    
    for col in df.columns:
        if col.lower() in possible_targets:
            return col
    
    return df.columns[-1]

# Hyperparameter Optimization Agent
hyperparameter_optimizer = LlmAgent(
    name="hyperparameter_optimizer",
    model="gemini-2.0-flash",
    instruction="""You are a hyperparameter optimization expert using Optuna:

1. Identify the best performing model from training results
2. Use Bayesian optimization to find optimal hyperparameters
3. Retrain the model with optimized parameters
4. Compare performance improvements
5. Save the optimized model

Use optimize_hyperparameters tool to improve model performance through systematic parameter tuning.""",
    
    tools=[optimize_hyperparameters]
)

# Ensemble Creation Agent
ensemble_creator = LlmAgent(
    name="ensemble_creator",
    model="gemini-2.0-flash", 
    instruction="""You are an ensemble modeling expert:

1. Combine multiple trained models into powerful ensembles
2. Create voting ensembles for robust predictions
3. Compare ensemble performance with individual models
4. Select the best ensemble strategy
5. Save final ensemble models

Use create_ensemble_model tool to build high-performance ensemble models that outperform individual models.""",
    
    tools=[create_ensemble_model]
)
