from google.adk.agents import LlmAgent
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

async def evaluate_ml_models(training_results: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive evaluation of trained ML models"""
    
    try:
        # Extract successful models
        successful_models = []
        
        if isinstance(training_results, dict):
            for agent_name, result in training_results.items():
                if isinstance(result, dict) and result.get("status") == "success":
                    successful_models.append({
                        "agent_name": agent_name,
                        "model_type": result.get("model_type", "unknown"),
                        "test_score": result.get("test_score", 0),
                        "cv_mean": result.get("cv_mean", 0),
                        "cv_std": result.get("cv_std", 0),
                        "additional_metrics": result.get("additional_metrics", {}),
                        "feature_importance": result.get("feature_importance", {}),
                        "model_path": result.get("model_path", "")
                    })
        
        if not successful_models:
            return {
                "status": "error",
                "message": "No successful models found for evaluation"
            }
        
        # Rank models by performance
        ranked_models = sorted(successful_models, key=lambda x: x["test_score"], reverse=True)
        best_model = ranked_models[0]
        
        # Calculate ensemble metrics if multiple models
        ensemble_performance = None
        if len(successful_models) > 1:
            ensemble_performance = {
                "average_score": np.mean([m["test_score"] for m in successful_models]),
                "score_std": np.std([m["test_score"] for m in successful_models]),
                "model_count": len(successful_models),
                "score_range": {
                    "min": min([m["test_score"] for m in successful_models]),
                    "max": max([m["test_score"] for m in successful_models])
                }
            }
        
        # Generate performance insights
        performance_insights = generate_performance_insights(successful_models, best_model)
        
        # Create evaluation report
        evaluation_report = {
            "status": "success",
            "evaluation_summary": {
                "total_models_evaluated": len(successful_models),
                "best_model": {
                    "name": best_model["model_type"],
                    "test_score": best_model["test_score"],
                    "cv_score": f"{best_model['cv_mean']:.3f} ± {best_model['cv_std']:.3f}",
                    "model_path": best_model["model_path"]
                },
                "model_rankings": [
                    {
                        "rank": i + 1,
                        "model_type": model["model_type"],
                        "test_score": model["test_score"],
                        "cv_mean": model["cv_mean"]
                    }
                    for i, model in enumerate(ranked_models)
                ]
            },
            "ensemble_performance": ensemble_performance,
            "performance_insights": performance_insights,
            "recommendations": generate_recommendations(successful_models, best_model),
            "model_comparison": create_model_comparison(successful_models)
        }
        
        return evaluation_report
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Model evaluation failed: {str(e)}"
        }

def generate_performance_insights(models: list, best_model: dict) -> list:
    """Generate insights about model performance"""
    
    insights = []
    
    # Best model insight
    insights.append(f"🏆 Best performing model: {best_model['model_type']} with {best_model['test_score']:.1%} accuracy")
    
    # Performance distribution
    scores = [m["test_score"] for m in models]
    if len(scores) > 1:
        score_std = np.std(scores)
        if score_std < 0.05:
            insights.append("📊 All models show consistent performance (low variance)")
        else:
            insights.append("📊 Models show varied performance - ensemble might be beneficial")
    
    # Cross-validation insights
    cv_scores = [m["cv_mean"] for m in models if m["cv_mean"] > 0]
    if cv_scores:
        avg_cv = np.mean(cv_scores)
        if avg_cv > 0.8:
            insights.append("✅ Strong cross-validation performance indicates robust models")
        elif avg_cv > 0.6:
            insights.append("⚠️ Moderate cross-validation performance - consider more data or feature engineering")
        else:
            insights.append("❌ Low cross-validation performance - models may need significant improvement")
    
    # Feature importance insights
    if best_model.get("feature_importance"):
        top_features = sorted(
            best_model["feature_importance"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        insights.append(f"🎯 Top predictive features: {', '.join([f for f in top_features])}")
    
    return insights

def generate_recommendations(models: list, best_model: dict) -> list:
    """Generate recommendations for model improvement"""
    
    recommendations = []
    
    best_score = best_model["test_score"]
    
    if best_score >= 0.9:
        recommendations.append("🎉 Excellent model performance! Ready for production deployment.")
        recommendations.append("Consider A/B testing to validate real-world performance.")
    elif best_score >= 0.8:
        recommendations.append("✅ Good model performance. Consider hyperparameter fine-tuning for improvement.")
        recommendations.append("Monitor model performance in production and retrain periodically.")
    elif best_score >= 0.7:
        recommendations.append("⚠️ Moderate performance. Consider:")
        recommendations.append("- Feature engineering to create more predictive features")
        recommendations.append("- Collecting more training data")
        recommendations.append("- Trying advanced algorithms or ensemble methods")
    else:
        recommendations.append("❌ Low performance indicates significant issues:")
        recommendations.append("- Review data quality and feature relevance")
        recommendations.append("- Consider different modeling approaches")
        recommendations.append("- Validate that the problem is learnable from available data")
    
    # Ensemble recommendations
    if len(models) > 1:
        scores = [m["test_score"] for m in models]
        if np.std(scores) > 0.05:  # High variance
            recommendations.append("🔄 Consider ensemble methods to combine diverse model predictions")
    
    return recommendations

def create_model_comparison(models: list) -> Dict[str, Any]:
    """Create detailed model comparison"""
    
    comparison = {
        "performance_metrics": {},
        "model_characteristics": {},
        "training_efficiency": {}
    }
    
    for model in models:
        model_name = model["model_type"]
        
        # Performance metrics
        comparison["performance_metrics"][model_name] = {
            "test_score": model["test_score"],
            "cv_mean": model["cv_mean"],
            "cv_std": model["cv_std"],
            "additional_metrics": model["additional_metrics"]
        }
        
        # Model characteristics
        comparison["model_characteristics"][model_name] = {
            "interpretability": get_interpretability_score(model_name),
            "training_speed": get_training_speed(model_name),
            "prediction_speed": get_prediction_speed(model_name),
            "memory_usage": get_memory_usage(model_name)
        }
    
    return comparison

def get_interpretability_score(model_type: str) -> str:
    """Get interpretability score for model type"""
    interpretability_map = {
        "random_forest": "High",
        "xgboost": "Medium",
        "lightgbm": "Medium",
        "catboost": "Medium",
        "linear_regression": "High",
        "logistic_regression": "High",
        "neural_network": "Low"
    }
    return interpretability_map.get(model_type.lower(), "Medium")

def get_training_speed(model_type: str) -> str:
    """Get training speed for model type"""
    speed_map = {
        "random_forest": "Medium",
        "xgboost": "Medium",
        "lightgbm": "Fast",
        "catboost": "Medium",
        "linear_regression": "Fast",
        "logistic_regression": "Fast",
        "neural_network": "Slow"
    }
    return speed_map.get(model_type.lower(), "Medium")

def get_prediction_speed(model_type: str) -> str:
    """Get prediction speed for model type"""
    speed_map = {
        "random_forest": "Medium",
        "xgboost": "Fast",
        "lightgbm": "Fast",
        "catboost": "Fast",
        "linear_regression": "Fast",
        "logistic_regression": "Fast",
        "neural_network": "Medium"
    }
    return speed_map.get(model_type.lower(), "Fast")

def get_memory_usage(model_type: str) -> str:
    """Get memory usage for model type"""
    memory_map = {
        "random_forest": "High",
        "xgboost": "Medium",
        "lightgbm": "Low",
        "catboost": "Medium",
        "linear_regression": "Low",
        "logistic_regression": "Low",
        "neural_network": "High"
    }
    return memory_map.get(model_type.lower(), "Medium")

# Model Evaluation Agent
evaluation_agent = LlmAgent(
    name="model_evaluator",
    model="gemini-2.0-flash",
    instruction="""You are a comprehensive model evaluation expert.

Your responsibilities:
1. Evaluate all trained models comprehensively
2. Rank models by performance metrics
3. Generate detailed performance insights
4. Provide actionable recommendations for improvement
5. Create model comparison reports
6. Assess production readiness

Use the evaluate_ml_models tool to perform thorough model evaluation.
Focus on both statistical performance and practical deployment considerations.""",
    
    tools=[evaluate_ml_models]
)
