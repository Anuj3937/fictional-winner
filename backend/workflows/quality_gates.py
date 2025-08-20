from typing import Dict, Any, List
import asyncio
import json
from pathlib import Path
from loguru import logger

class QualityGate:
    """Base class for quality gates with reiteration logic"""
    
    def __init__(self, name: str, max_iterations: int = 3):
        self.name = name
        self.max_iterations = max_iterations
        self.current_iteration = 0
    
    async def evaluate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate quality and determine if reiteration is needed"""
        raise NotImplementedError
    
    async def suggest_improvements(self, issues: List[str]) -> List[str]:
        """Suggest improvements based on identified issues"""
        raise NotImplementedError

class DataQualityGate(QualityGate):
    """Quality gate for data validation"""
    
    def __init__(self):
        super().__init__("DataQualityGate", max_iterations=3)
        self.quality_thresholds = {
            "min_quality_score": 0.6,
            "max_missing_ratio": 0.3,
            "min_samples": 100,
            "min_features": 2,
            "max_outlier_ratio": 0.1
        }
    
    async def evaluate(self, data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate data quality against thresholds"""
        try:
            self.current_iteration += 1
            
            # Extract metrics from data analysis
            quality_score = data_analysis.get("quality_score", 0)
            shape = data_analysis.get("shape", (0, 0))
            missing_values = data_analysis.get("missing_values", {})
            
            # Calculate quality metrics
            missing_ratio = sum(missing_values.values()) / (shape[0] * shape[1]) if shape * shape[1] > 0 else 1
            sample_count = shape
            feature_count = shape[1]
            
            # Evaluate against thresholds
            issues = []
            should_reiterate = False
            
            if quality_score < self.quality_thresholds["min_quality_score"]:
                issues.append(f"Quality score ({quality_score:.2f}) below threshold ({self.quality_thresholds['min_quality_score']})")
                should_reiterate = True
            
            if missing_ratio > self.quality_thresholds["max_missing_ratio"]:
                issues.append(f"Too many missing values ({missing_ratio:.1%})")
                should_reiterate = True
            
            if sample_count < self.quality_thresholds["min_samples"]:
                issues.append(f"Insufficient samples ({sample_count} < {self.quality_thresholds['min_samples']})")
                should_reiterate = True
            
            if feature_count < self.quality_thresholds["min_features"]:
                issues.append(f"Insufficient features ({feature_count} < {self.quality_thresholds['min_features']})")
                should_reiterate = True
            
            # Don't reiterate if we've reached max iterations
            if self.current_iteration >= self.max_iterations:
                should_reiterate = False
                if issues:
                    issues.append(f"Maximum iterations ({self.max_iterations}) reached - proceeding with current data")
            
            suggestions = await self.suggest_improvements(issues) if issues else []
            
            result = {
                "gate_name": self.name,
                "iteration": self.current_iteration,
                "quality_score": quality_score,
                "is_acceptable": not should_reiterate,
                "should_reiterate": should_reiterate,
                "issues": issues,
                "suggestions": suggestions,
                "metrics": {
                    "quality_score": quality_score,
                    "missing_ratio": missing_ratio,
                    "sample_count": sample_count,
                    "feature_count": feature_count
                }
            }
            
            logger.info(f"Data quality gate: {'PASS' if not should_reiterate else 'FAIL'} (iteration {self.current_iteration})")
            return result
            
        except Exception as e:
            logger.error(f"Data quality gate evaluation failed: {e}")
            return {
                "gate_name": self.name,
                "iteration": self.current_iteration,
                "is_acceptable": False,
                "should_reiterate": False,
                "error": str(e)
            }
    
    async def suggest_improvements(self, issues: List[str]) -> List[str]:
        """Generate improvement suggestions for data quality issues"""
        suggestions = []
        
        for issue in issues:
            if "quality score" in issue.lower():
                suggestions.extend([
                    "Search for higher quality datasets online",
                    "Apply data cleaning techniques",
                    "Consider synthetic data generation with better parameters"
                ])
            elif "missing values" in issue.lower():
                suggestions.extend([
                    "Find datasets with more complete data",
                    "Apply advanced imputation techniques",
                    "Consider removing features with excessive missing values"
                ])
            elif "insufficient samples" in issue.lower():
                suggestions.extend([
                    "Search for larger datasets",
                    "Apply data augmentation techniques",
                    "Generate synthetic data to supplement real data"
                ])
            elif "insufficient features" in issue.lower():
                suggestions.extend([
                    "Find datasets with richer feature sets",
                    "Apply feature engineering early",
                    "Consider feature extraction techniques"
                ])
        
        return list(set(suggestions))  # Remove duplicates

class ModelQualityGate(QualityGate):
    """Quality gate for model performance validation"""
    
    def __init__(self):
        super().__init__("ModelQualityGate", max_iterations=3)
        self.performance_thresholds = {
            "min_accuracy": 0.7,  # For classification
            "min_r2": 0.6,       # For regression
            "max_cv_std": 0.1,   # Cross-validation stability
            "min_models": 2      # Minimum successful models
        }
    
    async def evaluate(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate model performance against thresholds"""
        try:
            self.current_iteration += 1
            
            # Extract model results
            successful_models = []
            if isinstance(model_results, dict):
                for result in model_results.values():
                    if isinstance(result, dict) and result.get("status") == "success":
                        successful_models.append(result)
            
            if not successful_models:
                return {
                    "gate_name": self.name,
                    "iteration": self.current_iteration,
                    "is_acceptable": False,
                    "should_reiterate": self.current_iteration < self.max_iterations,
                    "issues": ["No models trained successfully"],
                    "suggestions": ["Retry model training with different parameters"]
                }
            
            # Calculate performance metrics
            test_scores = [m.get("test_score", 0) for m in successful_models]
            cv_stds = [m.get("cv_std", 0) for m in successful_models]
            
            best_score = max(test_scores)
            avg_score = sum(test_scores) / len(test_scores)
            max_cv_std = max(cv_stds) if cv_stds else 0
            
            # Determine task type from first model
            task_type = successful_models[0].get("task_type", "classification")
            
            # Set appropriate threshold
            min_score = (self.performance_thresholds["min_accuracy"] 
                        if task_type == "classification" 
                        else self.performance_thresholds["min_r2"])
            
            # Evaluate against thresholds
            issues = []
            should_reiterate = False
            
            if best_score < min_score:
                issues.append(f"Best model score ({best_score:.3f}) below threshold ({min_score})")
                should_reiterate = True
            
            if len(successful_models) < self.performance_thresholds["min_models"]:
                issues.append(f"Only {len(successful_models)} successful model(s), expected {self.performance_thresholds['min_models']}")
                should_reiterate = True
            
            if max_cv_std > self.performance_thresholds["max_cv_std"]:
                issues.append(f"High cross-validation variance ({max_cv_std:.3f})")
                # Don't reiterate for CV variance alone
            
            # Don't reiterate if we've reached max iterations
            if self.current_iteration >= self.max_iterations:
                should_reiterate = False
                if issues:
                    issues.append(f"Maximum iterations ({self.max_iterations}) reached - proceeding with best model")
            
            suggestions = await self.suggest_improvements(issues) if issues else []
            
            result = {
                "gate_name": self.name,
                "iteration": self.current_iteration,
                "is_acceptable": not should_reiterate,
                "should_reiterate": should_reiterate,
                "issues": issues,
                "suggestions": suggestions,
                "metrics": {
                    "best_score": best_score,
                    "average_score": avg_score,
                    "max_cv_std": max_cv_std,
                    "successful_models": len(successful_models),
                    "task_type": task_type
                }
            }
            
            logger.info(f"Model quality gate: {'PASS' if not should_reiterate else 'FAIL'} (iteration {self.current_iteration})")
            return result
            
        except Exception as e:
            logger.error(f"Model quality gate evaluation failed: {e}")
            return {
                "gate_name": self.name,
                "iteration": self.current_iteration,
                "is_acceptable": False,
                "should_reiterate": False,
                "error": str(e)
            }
    
    async def suggest_improvements(self, issues: List[str]) -> List[str]:
        """Generate improvement suggestions for model performance issues"""
        suggestions = []
        
        for issue in issues:
            if "model score" in issue.lower():
                suggestions.extend([
                    "Try hyperparameter optimization",
                    "Apply more advanced feature engineering",
                    "Consider ensemble methods",
                    "Try different algorithms"
                ])
            elif "successful model" in issue.lower():
                suggestions.extend([
                    "Adjust model parameters",
                    "Check data preprocessing",
                    "Try simpler algorithms first",
                    "Increase training time limits"
                ])
            elif "variance" in issue.lower():
                suggestions.extend([
                    "Apply regularization techniques",
                    "Increase training data",
                    "Use more stable algorithms",
                    "Apply cross-validation tuning"
                ])
        
        return list(set(suggestions))

# Quality gate instances
data_quality_gate = DataQualityGate()
model_quality_gate = ModelQualityGate()

async def evaluate_data_quality(data_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate data quality using the data quality gate"""
    return await data_quality_gate.evaluate(data_analysis)

async def evaluate_model_quality(model_results: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate model quality using the model quality gate"""
    return await model_quality_gate.evaluate(model_results)
