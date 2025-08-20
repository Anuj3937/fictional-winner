import pandas as pd
import numpy as np
from typing import Dict, Any, List
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

async def assess_data_quality_comprehensive(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive data quality assessment with detailed metrics"""
    
    try:
        if analysis_result.get("status") != "success":
            return {
                "status": "error",
                "message": "Invalid analysis result provided"
            }
        
        # Extract key metrics
        quality_score = analysis_result.get("quality_score", 0)
        shape = analysis_result.get("shape", (0, 0))
        missing_values = analysis_result.get("missing_values", {})
        duplicates = analysis_result.get("duplicates", {})
        
        # Calculate derived metrics
        total_cells = shape[0] * shape[1] if shape * shape[1] > 0 else 1
        missing_ratio = sum(missing_values.values()) / total_cells
        duplicate_ratio = duplicates.get("percentage", 0) / 100
        
        # Quality thresholds
        thresholds = {
            "min_quality_score": 0.6,
            "max_missing_ratio": 0.2,
            "max_duplicate_ratio": 0.05,
            "min_samples": 200,
            "min_features": 3,
            "max_high_cardinality_ratio": 0.1
        }
        
        # Detailed quality assessment
        quality_checks = []
        issues = []
        recommendations = []
        should_reiterate = False
        
        # Overall quality score check
        quality_check = {
            "check_name": "overall_quality_score",
            "threshold": thresholds["min_quality_score"],
            "actual_value": quality_score,
            "passed": quality_score >= thresholds["min_quality_score"],
            "severity": "high"
        }
        quality_checks.append(quality_check)
        
        if not quality_check["passed"]:
            issues.append(f"Overall quality score ({quality_score:.3f}) below acceptable threshold ({thresholds['min_quality_score']})")
            recommendations.extend([
                "Consider finding a higher quality dataset",
                "Apply comprehensive data cleaning",
                "Validate data source and collection methods"
            ])
            should_reiterate = True
        
        # Missing values check
        missing_check = {
            "check_name": "missing_values",
            "threshold": thresholds["max_missing_ratio"],
            "actual_value": missing_ratio,
            "passed": missing_ratio <= thresholds["max_missing_ratio"],
            "severity": "medium" if missing_ratio <= 0.3 else "high"
        }
        quality_checks.append(missing_check)
        
        if not missing_check["passed"]:
            issues.append(f"High missing value ratio ({missing_ratio:.1%}) exceeds threshold ({thresholds['max_missing_ratio']:.1%})")
            recommendations.extend([
                "Find datasets with more complete data",
                "Apply sophisticated imputation techniques",
                "Consider removing columns with excessive missing values"
            ])
            if missing_ratio > 0.4:
                should_reiterate = True
        
        # Sample size check
        sample_check = {
            "check_name": "sample_size",
            "threshold": thresholds["min_samples"],
            "actual_value": shape[0],
            "passed": shape >= thresholds["min_samples"],
            "severity": "high"
        }
        quality_checks.append(sample_check)
        
        if not sample_check["passed"]:
            issues.append(f"Insufficient samples ({shape[0]}) for reliable modeling (minimum {thresholds['min_samples']})")
            recommendations.extend([
                "Collect more data samples",
                "Use data augmentation techniques",
                "Generate synthetic data to supplement real data"
            ])
            if shape[0] < 100:
                should_reiterate = True
        
        # Feature count check
        feature_check = {
            "check_name": "feature_count",
            "threshold": thresholds["min_features"],
            "actual_value": shape[1],
            "passed": shape[1] >= thresholds["min_features"],
            "severity": "medium"
        }
        quality_checks.append(feature_check)
        
        if not feature_check["passed"]:
            issues.append(f"Insufficient features ({shape[1]}) for comprehensive modeling")
            recommendations.extend([
                "Apply feature engineering techniques",
                "Find datasets with richer feature sets",
                "Create derived features from existing ones"
            ])
        
        # Duplicate check
        duplicate_check = {
            "check_name": "duplicates",
            "threshold": thresholds["max_duplicate_ratio"],
            "actual_value": duplicate_ratio,
            "passed": duplicate_ratio <= thresholds["max_duplicate_ratio"],
            "severity": "low" if duplicate_ratio <= 0.1 else "medium"
        }
        quality_checks.append(duplicate_check)
        
        if not duplicate_check["passed"]:
            issues.append(f"High duplicate ratio ({duplicate_ratio:.1%}) may indicate data collection issues")
            recommendations.append("Remove duplicate records before modeling")
        
        # Data distribution checks
        numeric_stats = analysis_result.get("numeric_stats", {})
        categorical_stats = analysis_result.get("categorical_stats", {})
        
        # Check for extreme skewness in numeric columns
        skewed_features = []
        for feature, stats_dict in numeric_stats.items():
            skewness = stats_dict.get("skewness")
            if skewness is not None and abs(skewness) > 2:
                skewed_features.append(feature)
        
        if skewed_features:
            issues.append(f"Highly skewed features detected: {', '.join(skewed_features[:3])}{'...' if len(skewed_features) > 3 else ''}")
            recommendations.append("Consider applying transformations (log, sqrt) to highly skewed features")
        
        # Check for high cardinality categorical features
        high_cardinality_features = []
        for feature, stats_dict in categorical_stats.items():
            cardinality_ratio = stats_dict.get("unique_count", 0) / shape[0] if shape[0] > 0 else 0
            if cardinality_ratio > thresholds["max_high_cardinality_ratio"]:
                high_cardinality_features.append(feature)
        
        if high_cardinality_features:
            issues.append(f"High cardinality categorical features: {', '.join(high_cardinality_features[:3])}")
            recommendations.append("Consider feature encoding or grouping for high cardinality categorical features")
        
        # Calculate overall quality assessment
        passed_checks = sum(1 for check in quality_checks if check["passed"])
        quality_percentage = (passed_checks / len(quality_checks)) * 100
        
        # Final quality determination
        overall_quality = "excellent" if quality_percentage >= 90 else \
                         "good" if quality_percentage >= 70 else \
                         "fair" if quality_percentage >= 50 else "poor"
        
        return {
            "status": "success",
            "quality_assessment": {
                "overall_score": quality_score,
                "overall_quality": overall_quality,
                "quality_percentage": quality_percentage,
                "is_acceptable": not should_reiterate,
                "should_reiterate": should_reiterate,
                "passed_checks": passed_checks,
                "total_checks": len(quality_checks)
            },
            "detailed_checks": quality_checks,
            "issues": issues,
            "recommendations": recommendations,
            "metrics": {
                "missing_ratio": missing_ratio,
                "duplicate_ratio": duplicate_ratio,
                "sample_count": shape[0],
                "feature_count": shape[1],
                "skewed_features_count": len(skewed_features),
                "high_cardinality_features_count": len(high_cardinality_features)
            },
            "next_action": "proceed" if not should_reiterate else "improve_data_quality"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Quality assessment failed: {str(e)}"
        }

async def assess_model_ensemble_quality_advanced(model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Advanced ensemble quality assessment with model diversity analysis"""
    
    try:
        if not model_results:
            return {
                "status": "error",
                "message": "No model results provided",
                "quality_assessment": {
                    "is_acceptable": False,
                    "should_reiterate": True
                }
            }
        
        # Filter successful models
        successful_models = [r for r in model_results if r.get("status") == "success"]
        
        if not successful_models:
            return {
                "status": "success",
                "quality_assessment": {
                    "overall_score": 0.0,
                    "is_acceptable": False,
                    "should_reiterate": True,
                    "issues": ["No models trained successfully"],
                    "recommendations": ["Retry model training with adjusted parameters"]
                }
            }
        
        # Extract performance metrics
        performance_metrics = []
        for model in successful_models:
            perf = model.get("performance", {})
            performance_metrics.append({
                "model_type": model.get("model_info", {}).get("type", "unknown"),
                "test_score": perf.get("test_score", 0),
                "cv_mean": perf.get("cv_mean", 0),
                "cv_std": perf.get("cv_std", 1),
                "train_score": perf.get("train_score", 0),
                "overfitting": perf.get("overfitting_score", 0)
            })
        
        # Calculate ensemble statistics
        test_scores = [m["test_score"] for m in performance_metrics]
        cv_means = [m["cv_mean"] for m in performance_metrics]
        cv_stds = [m["cv_std"] for m in performance_metrics]
        
        best_score = max(test_scores)
        avg_score = np.mean(test_scores)
        score_std = np.std(test_scores)
        avg_cv_std = np.mean(cv_stds)
        
        # Quality thresholds
        thresholds = {
            "min_best_score": 0.75,
            "min_avg_score": 0.65,
            "max_score_variance": 0.15,
            "max_avg_cv_std": 0.1,
            "min_models": 2,
            "max_overfitting": 0.1
        }
        
        # Detailed quality checks
        quality_checks = []
        issues = []
        recommendations = []
        
        # Best model performance check
        best_score_check = {
            "check_name": "best_model_performance",
            "threshold": thresholds["min_best_score"],
            "actual_value": best_score,
            "passed": best_score >= thresholds["min_best_score"],
            "severity": "high"
        }
        quality_checks.append(best_score_check)
        
        if not best_score_check["passed"]:
            issues.append(f"Best model score ({best_score:.3f}) below performance threshold")
            recommendations.extend([
                "Apply hyperparameter optimization",
                "Try advanced feature engineering",
                "Consider ensemble methods or different algorithms"
            ])
        
        # Average performance check
        avg_score_check = {
            "check_name": "average_performance",
            "threshold": thresholds["min_avg_score"],
            "actual_value": avg_score,
            "passed": avg_score >= thresholds["min_avg_score"],
            "severity": "medium"
        }
        quality_checks.append(avg_score_check)
        
        if not avg_score_check["passed"]:
            issues.append(f"Average model performance ({avg_score:.3f}) is suboptimal")
            recommendations.append("Improve overall model training strategy")
        
        # Model count check
        model_count_check = {
            "check_name": "model_diversity",
            "threshold": thresholds["min_models"],
            "actual_value": len(successful_models),
            "passed": len(successful_models) >= thresholds["min_models"],
            "severity": "high"
        }
        quality_checks.append(model_count_check)
        
        if not model_count_check["passed"]:
            issues.append(f"Only {len(successful_models)} successful model(s), insufficient for robust ensemble")
            recommendations.append("Ensure multiple algorithms train successfully")
        
        # Performance consistency check
        consistency_check = {
            "check_name": "performance_consistency",
            "threshold": thresholds["max_score_variance"],
            "actual_value": score_std,
            "passed": score_std <= thresholds["max_score_variance"],
            "severity": "medium"
        }
        quality_checks.append(consistency_check)
        
        if not consistency_check["passed"]:
            issues.append(f"High variance in model performance ({score_std:.3f})")
            recommendations.append("Investigate why models perform very differently")
        
        # Cross-validation stability check
        stability_check = {
            "check_name": "cv_stability",
            "threshold": thresholds["max_avg_cv_std"],
            "actual_value": avg_cv_std,
            "passed": avg_cv_std <= thresholds["max_avg_cv_std"],
            "severity": "medium"
        }
        quality_checks.append(stability_check)
        
        if not stability_check["passed"]:
            issues.append(f"Models show high cross-validation variance ({avg_cv_std:.3f})")
            recommendations.append("Apply regularization or gather more training data")
        
        # Overfitting analysis
        overfitting_scores = [abs(m["overfitting"]) for m in performance_metrics]
        avg_overfitting = np.mean(overfitting_scores)
        
        overfitting_check = {
            "check_name": "overfitting_control",
            "threshold": thresholds["max_overfitting"],
            "actual_value": avg_overfitting,
            "passed": avg_overfitting <= thresholds["max_overfitting"],
            "severity": "medium"
        }
        quality_checks.append(overfitting_check)
        
        if not overfitting_check["passed"]:
            issues.append(f"Models show signs of overfitting (avg gap: {avg_overfitting:.3f})")
            recommendations.append("Apply regularization techniques")
        
        # Calculate overall quality score
        passed_checks = sum(1 for check in quality_checks if check["passed"])
        quality_percentage = (passed_checks / len(quality_checks)) * 100
        
        # Weight the quality score by performance
        performance_weight = min(1.0, best_score / thresholds["min_best_score"])
        overall_score = (quality_percentage / 100) * performance_weight
        
        # Determine if reiteration is needed
        critical_failures = sum(1 for check in quality_checks 
                               if not check["passed"] and check["severity"] == "high")
        should_reiterate = critical_failures > 0 or overall_score < 0.6
        
        # Model diversity analysis
        model_types = set(m["model_type"] for m in performance_metrics)
        diversity_score = len(model_types) / max(1, len(successful_models))
        
        return {
            "status": "success",
            "quality_assessment": {
                "overall_score": overall_score,
                "quality_percentage": quality_percentage,
                "is_acceptable": not should_reiterate,
                "should_reiterate": should_reiterate,
                "confidence_level": "high" if overall_score >= 0.8 else "medium" if overall_score >= 0.6 else "low"
            },
            "performance_summary": {
                "best_score": best_score,
                "average_score": avg_score,
                "score_std": score_std,
                "avg_cv_std": avg_cv_std,
                "successful_models": len(successful_models),
                "model_diversity": diversity_score,
                "avg_overfitting": avg_overfitting
            },
            "detailed_checks": quality_checks,
            "issues": issues,
            "recommendations": recommendations,
            "model_rankings": sorted(performance_metrics, key=lambda x: x["test_score"], reverse=True),
            "next_action": "proceed_to_ensemble" if not should_reiterate else "optimize_models"
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Model quality assessment failed: {str(e)}"
        }

def calculate_model_diversity(model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate diversity metrics for model ensemble"""
    
    try:
        diversity_metrics = {
            "algorithm_diversity": 0.0,
            "performance_diversity": 0.0,
            "prediction_diversity": 0.0
        }
        
        if len(model_results) < 2:
            return diversity_metrics
        
        # Algorithm diversity
        model_types = [r.get("model_info", {}).get("type", "unknown") for r in model_results]
        unique_types = len(set(model_types))
        diversity_metrics["algorithm_diversity"] = unique_types / len(model_types)
        
        # Performance diversity (coefficient of variation)
        test_scores = [r.get("performance", {}).get("test_score", 0) for r in model_results]
        if len(test_scores) > 1 and np.mean(test_scores) > 0:
            cv = np.std(test_scores) / np.mean(test_scores)
            diversity_metrics["performance_diversity"] = min(1.0, cv)
        
        return diversity_metrics
        
    except Exception as e:
        return {
            "algorithm_diversity": 0.0,
            "performance_diversity": 0.0,
            "prediction_diversity": 0.0,
            "error": str(e)
        }

async def generate_quality_improvement_plan(assessment: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a comprehensive quality improvement plan"""
    
    try:
        issues = assessment.get("issues", [])
        recommendations = assessment.get("recommendations", [])
        
        # Categorize issues by severity and type
        improvement_plan = {
            "priority_actions": [],
            "secondary_actions": [],
            "monitoring_actions": [],
            "estimated_impact": {}
        }
        
        # Map issues to actions
        issue_action_map = {
            "quality score": "priority_actions",
            "missing value": "priority_actions", 
            "insufficient sample": "priority_actions",
            "model score": "priority_actions",
            "overfitting": "secondary_actions",
            "variance": "secondary_actions",
            "duplicate": "secondary_actions",
            "skewed": "monitoring_actions",
            "cardinality": "monitoring_actions"
        }
        
        for issue in issues:
            action_category = "secondary_actions"  # default
            for keyword, category in issue_action_map.items():
                if keyword in issue.lower():
                    action_category = category
                    break
            
            if action_category not in improvement_plan:
                improvement_plan[action_category] = []
            
            # Generate specific action based on issue
            action = generate_specific_action(issue)
            if action not in improvement_plan[action_category]:
                improvement_plan[action_category].append(action)
        
        # Add recommendations to appropriate categories
        for rec in recommendations:
            if any(word in rec.lower() for word in ["find", "collect", "search", "dataset"]):
                improvement_plan["priority_actions"].append(rec)
            elif any(word in rec.lower() for word in ["regularization", "optimization", "tuning"]):
                improvement_plan["secondary_actions"].append(rec)
            else:
                improvement_plan["monitoring_actions"].append(rec)
        
        # Remove duplicates and limit actions
        for category in improvement_plan:
            if isinstance(improvement_plan[category], list):
                improvement_plan[category] = list(set(improvement_plan[category]))[:5]
        
        return {
            "status": "success",
            "improvement_plan": improvement_plan,
            "total_actions": sum(len(actions) for actions in improvement_plan.values() if isinstance(actions, list))
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to generate improvement plan: {str(e)}"
        }

def generate_specific_action(issue: str) -> str:
    """Generate specific actionable recommendations based on issue"""
    
    issue_lower = issue.lower()
    
    if "quality score" in issue_lower:
        return "Search for higher quality datasets from reputable sources"
    elif "missing value" in issue_lower:
        return "Apply advanced imputation techniques (KNN, iterative)"
    elif "insufficient sample" in issue_lower:
        return "Increase dataset size through data collection or augmentation"
    elif "model score" in issue_lower:
        return "Apply hyperparameter optimization and feature engineering"
    elif "overfitting" in issue_lower:
        return "Implement regularization and cross-validation"
    elif "variance" in issue_lower:
        return "Increase training data stability and apply ensemble methods"
    elif "duplicate" in issue_lower:
        return "Remove duplicate records and validate data collection process"
    else:
        return "Apply general data quality improvement techniques"
