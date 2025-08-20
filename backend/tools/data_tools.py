import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import asyncio
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

async def analyze_dataset(file_path: str) -> Dict[str, Any]:
    """Comprehensive dataset analysis with quality scoring"""
    try:
        # Load dataset efficiently
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, low_memory=False)
        elif file_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_path.suffix.lower() == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Basic information
        shape = df.shape
        columns = df.columns.tolist()
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Missing values analysis
        missing_values = df.isnull().sum().to_dict()
        missing_percentage = (df.isnull().sum() / len(df) * 100).to_dict()
        
        # Duplicates
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100
        
        # Memory usage
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        
        # Statistical summary for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_stats = {}
        for col in numeric_columns:
            numeric_stats[col] = {
                'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                'std': float(df[col].std()) if not df[col].isna().all() else None,
                'min': float(df[col].min()) if not df[col].isna().all() else None,
                'max': float(df[col].max()) if not df[col].isna().all() else None,
                'median': float(df[col].median()) if not df[col].isna().all() else None,
                'skewness': float(df[col].skew()) if not df[col].isna().all() else None,
                'kurtosis': float(df[col].kurtosis()) if not df[col].isna().all() else None
            }
        
        # Categorical analysis
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        categorical_stats = {}
        for col in categorical_columns:
            categorical_stats[col] = {
                'unique_count': int(df[col].nunique()),
                'most_frequent': df[col].mode().iloc[0] if not df[col].empty and len(df[col].mode()) > 0 else None,
                'frequency_distribution': df[col].value_counts().head(10).to_dict()
            }
        
        # Outlier detection for numeric columns
        outliers = {}
        for col in numeric_columns:
            if not df[col].isna().all():
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outliers[col] = {
                    'count': int(outlier_count),
                    'percentage': float(outlier_count / len(df) * 100)
                }
        
        # Data quality score calculation
        quality_score = calculate_quality_score(df)
        
        # Generate recommendations
        recommendations = generate_data_recommendations(df, quality_score)
        
        # Sample data (first 5 rows)
        sample_data = df.head(5).to_dict('records')
        
        return {
            "status": "success",
            "file_info": {
                "path": str(file_path),
                "size_mb": memory_usage,
                "format": file_path.suffix
            },
            "shape": shape,
            "columns": columns,
            "dtypes": dtypes,
            "missing_values": missing_values,
            "missing_percentage": missing_percentage,
            "duplicates": {
                "count": int(duplicate_count),
                "percentage": float(duplicate_percentage)
            },
            "memory_usage_mb": memory_usage,
            "numeric_stats": numeric_stats,
            "categorical_stats": categorical_stats,
            "outliers": outliers,
            "quality_score": quality_score,
            "recommendations": recommendations,
            "sample_data": sample_data,
            "data_types": {
                "numeric_columns": list(numeric_columns),
                "categorical_columns": list(categorical_columns),
                "datetime_columns": list(df.select_dtypes(include=['datetime64']).columns)
            }
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Dataset analysis failed: {str(e)}"}

def calculate_quality_score(df: pd.DataFrame) -> float:
    """Calculate comprehensive data quality score (0-1)"""
    score = 1.0
    
    # Penalize missing values
    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    score -= missing_ratio * 0.4
    
    # Penalize duplicates
    duplicate_ratio = df.duplicated().sum() / len(df)
    score -= duplicate_ratio * 0.2
    
    # Penalize insufficient data
    if len(df) < 100:
        score -= 0.3
    elif len(df) < 500:
        score -= 0.1
    
    # Penalize insufficient features
    if df.shape[1] < 3:
        score -= 0.2
    elif df.shape[1] < 5:
        score -= 0.1
    
    # Bonus for good data variety
    numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
    categorical_cols = df.select_dtypes(include=['object', 'category']).shape[1]
    
    if numeric_cols > 0 and categorical_cols > 0:
        score += 0.1  # Mixed data types are good
    
    # Penalize high cardinality categorical variables
    for col in df.select_dtypes(include=['object', 'category']).columns:
        cardinality_ratio = df[col].nunique() / len(df)
        if cardinality_ratio > 0.5:  # High cardinality
            score -= 0.05
    
    return max(0.0, min(1.0, score))

def generate_data_recommendations(df: pd.DataFrame, quality_score: float) -> List[str]:
    """Generate actionable recommendations for data improvement"""
    recommendations = []
    
    # Missing values recommendations
    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    if missing_ratio > 0.1:
        recommendations.append("Consider imputation strategies for missing values")
        if missing_ratio > 0.3:
            recommendations.append("High missing values - consider finding a more complete dataset")
    
    # Duplicates recommendations
    duplicate_ratio = df.duplicated().sum() / len(df)
    if duplicate_ratio > 0.05:
        recommendations.append("Remove duplicate rows to improve data quality")
    
    # Sample size recommendations
    if len(df) < 1000:
        recommendations.append("Consider collecting more data for better model performance")
    
    # Feature recommendations
    if df.shape[1] < 5:
        recommendations.append("Consider feature engineering to create more predictive features")
    
    # High cardinality recommendations
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if df[col].nunique() / len(df) > 0.1:
            recommendations.append(f"Consider encoding or grouping high-cardinality feature: {col}")
    
    # Outlier recommendations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if not df[col].isna().all():
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_ratio = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum() / len(df)
            if outlier_ratio > 0.05:
                recommendations.append(f"Consider handling outliers in column: {col}")
    
    # Overall quality recommendations
    if quality_score < 0.5:
        recommendations.append("Data quality is low - consider finding a higher quality dataset")
    elif quality_score < 0.7:
        recommendations.append("Data quality is moderate - preprocessing will be important")
    
    return recommendations

async def search_datasets_online(domain: str, task_type: str) -> Dict[str, Any]:
    """Search for suitable datasets online (mock implementation)"""
    try:
        # Mock dataset suggestions based on domain and task type
        datasets = {
            "healthcare": {
                "classification": [
                    {"name": "Heart Disease Dataset", "source": "UCI", "samples": 303, "features": 14},
                    {"name": "Diabetes Dataset", "source": "Kaggle", "samples": 768, "features": 8},
                    {"name": "Breast Cancer Dataset", "source": "sklearn", "samples": 569, "features": 30}
                ],
                "regression": [
                    {"name": "Medical Cost Dataset", "source": "Kaggle", "samples": 1338, "features": 7},
                    {"name": "Life Expectancy Dataset", "source": "WHO", "samples": 2938, "features": 22}
                ]
            },
            "finance": {
                "classification": [
                    {"name": "Credit Card Fraud", "source": "Kaggle", "samples": 284807, "features": 31},
                    {"name": "Loan Default Dataset", "source": "Lending Club", "samples": 887383, "features": 74}
                ],
                "regression": [
                    {"name": "House Prices Dataset", "source": "Kaggle", "samples": 1460, "features": 79},
                    {"name": "Stock Price Dataset", "source": "Yahoo Finance", "samples": 10000, "features": 6}
                ]
            }
        }
        
        domain_datasets = datasets.get(domain, datasets["finance"])  # Fallback to finance
        task_datasets = domain_datasets.get(task_type, domain_datasets["classification"])  # Fallback to classification
        
        return {
            "status": "success",
            "domain": domain,
            "task_type": task_type,
            "suggested_datasets": task_datasets,
            "message": f"Found {len(task_datasets)} suitable datasets for {domain} {task_type}"
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Dataset search failed: {str(e)}"}

async def validate_dataset_format(file_path: str) -> Dict[str, Any]:
    """Validate dataset format and structure"""
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"valid": False, "error": "File does not exist"}
        
        # Check file size (limit to 500MB)
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > 500:
            return {"valid": False, "error": f"File too large ({size_mb:.1f}MB), maximum 500MB allowed"}
        
        # Check file format
        supported_formats = ['.csv', '.parquet', '.xlsx', '.xls', '.json']
        if file_path.suffix.lower() not in supported_formats:
            return {"valid": False, "error": f"Unsupported format {file_path.suffix}, supported: {supported_formats}"}
        
        # Try to load a small sample to validate format
        try:
            if file_path.suffix.lower() == '.csv':
                pd.read_csv(file_path, nrows=5)
            elif file_path.suffix.lower() == '.parquet':
                pd.read_parquet(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                pd.read_excel(file_path, nrows=5)
            elif file_path.suffix.lower() == '.json':
                pd.read_json(file_path, nrows=5)
        except Exception as e:
            return {"valid": False, "error": f"Invalid file format: {str(e)}"}
        
        return {
            "valid": True,
            "format": file_path.suffix.lower(),
            "size_mb": size_mb,
            "message": "Dataset format is valid"
        }
        
    except Exception as e:
        return {"valid": False, "error": f"Validation failed: {str(e)}"}
