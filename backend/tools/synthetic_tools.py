import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import json
from scipy import stats
from loguru import logger
from tools.search_tools import search_engine
import asyncio

@dataclass
class SyntheticDataConfig:
    n_samples: int
    n_features: int
    task_type: str
    domain: str
    feature_names: List[str]
    statistical_params: Dict[str, Any]
    target_correlation: float = 0.3

class HighPerformanceSyntheticGenerator:
    """Ultra-fast synthetic data generation with statistical accuracy"""
    
    def __init__(self):
        self.rng = np.random.default_rng(42)  # Use modern random generator for speed
        
    async def generate_research_based_dataset(self, requirements: Dict[str, Any]) -> str:
        """Generate synthetic dataset based on web research - optimized for speed"""
        
        try:
            # Extract requirements
            n_samples = min(requirements.get("n_samples", 1000), 100000)  # Cap for performance
            n_features = min(requirements.get("n_features", 5), 50)  # Cap for performance
            task_type = requirements.get("task_type", "classification")
            domain = requirements.get("domain", "general")
            
            logger.info(f"Generating {n_samples} samples with {n_features} features for {domain} {task_type}")
            
            # Research statistical parameters concurrently
            statistical_params = await search_engine.search_statistical_data(domain, task_type)
            
            # Generate feature names
            feature_names = self._generate_feature_names(domain, n_features)
            
            # Create configuration
            config = SyntheticDataConfig(
                n_samples=n_samples,
                n_features=n_features,
                task_type=task_type,
                domain=domain,
                feature_names=feature_names,
                statistical_params=statistical_params
            )
            
            # Generate dataset using vectorized operations for speed
            df = self._generate_optimized_dataset(config)
            
            # Save dataset
            output_path = self._save_dataset(df, config)
            
            logger.info(f"Generated synthetic dataset: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Synthetic data generation failed: {e}")
            raise
    
    def _generate_feature_names(self, domain: str, n_features: int) -> List[str]:
        """Generate domain-appropriate feature names - O(1) lookup"""
        
        domain_features = {
            "healthcare": [
                "age", "bmi", "blood_pressure_systolic", "blood_pressure_diastolic",
                "heart_rate", "cholesterol", "glucose", "weight", "height", "temperature"
            ],
            "finance": [
                "income", "credit_score", "debt_ratio", "age", "savings",
                "loan_amount", "employment_years", "assets", "monthly_expenses", "dependents"
            ],
            "retail": [
                "price", "quantity", "discount", "rating", "reviews_count",
                "category_id", "brand_score", "shipping_cost", "return_rate", "profit_margin"
            ],
            "transportation": [
                "distance", "duration", "speed", "fuel_consumption", "cost",
                "vehicle_age", "capacity", "efficiency", "maintenance_score", "safety_rating"
            ]
        }
        
        base_features = domain_features.get(domain, [f"feature_{i}" for i in range(20)])
        
        # Return requested number of features, cycling if needed
        if len(base_features) >= n_features:
            return base_features[:n_features]
        else:
            # Extend with generic features if needed
            extended_features = base_features.copy()
            for i in range(len(base_features), n_features):
                extended_features.append(f"feature_{i}")
            return extended_features
    
    def _generate_optimized_dataset(self, config: SyntheticDataConfig) -> pd.DataFrame:
        """Generate dataset using vectorized operations for maximum speed"""
        
        logger.info("Generating features with vectorized operations")
        
        # Pre-allocate data array for speed - O(1) allocation
        data = np.empty((config.n_samples, config.n_features), dtype=np.float32)
        
        # Get statistical parameters
        feature_ranges = config.statistical_params.get("feature_ranges", {})
        distribution_type = config.statistical_params.get("distributions", {}).get("primary", "normal")
        typical_values = config.statistical_params.get("typical_values", {"mean": 50, "std": 15})
        
        # Generate all features using vectorized operations
        for i, feature_name in enumerate(config.feature_names):
            data[:, i] = self._generate_feature_vector(
                feature_name, config.n_samples, feature_ranges, 
                distribution_type, typical_values
            )
        
        # Create DataFrame - much faster than iterative construction
        df = pd.DataFrame(data, columns=config.feature_names)
        
        # Generate target variable with realistic correlations
        target = self._generate_target_variable(df, config)
        df["target"] = target
        
        # Add some realistic correlations between features (vectorized)
        df = self._add_feature_correlations(df, config)
        
        logger.info(f"Generated dataset shape: {df.shape}")
        return df
    
    def _generate_feature_vector(self, feature_name: str, n_samples: int, 
                               feature_ranges: Dict, distribution_type: str, 
                               typical_values: Dict) -> np.ndarray:
        """Generate single feature vector using optimized distributions"""
        
        # Get range for this feature
        if feature_name in feature_ranges:
            min_val = feature_ranges[feature_name]["min"]
            max_val = feature_ranges[feature_name]["max"]
        else:
            # Use typical values with some spread
            mean_val = typical_values.get("mean", 50)
            std_val = typical_values.get("std", 15)
            min_val = max(0, mean_val - 3 * std_val)
            max_val = mean_val + 3 * std_val
        
        # Generate based on distribution type using vectorized operations
        if distribution_type == "uniform":
            return self.rng.uniform(min_val, max_val, n_samples)
        
        elif distribution_type == "exponential":
            # Generate exponential and scale to range
            scale = (max_val - min_val) / 3
            exp_data = self.rng.exponential(scale, n_samples)
            return np.clip(exp_data + min_val, min_val, max_val)
        
        elif distribution_type == "log-normal":
            # Generate log-normal and scale to range
            mu = np.log((min_val + max_val) / 2)
            sigma = 0.5
            lognorm_data = self.rng.lognormal(mu, sigma, n_samples)
            return np.clip(lognorm_data, min_val, max_val)
        
        else:  # normal distribution (default)
            mean = (min_val + max_val) / 2
            std = (max_val - min_val) / 6  # 99.7% of data within range
            normal_data = self.rng.normal(mean, std, n_samples)
            return np.clip(normal_data, min_val, max_val)
    
    def _generate_target_variable(self, df: pd.DataFrame, config: SyntheticDataConfig) -> np.ndarray:
        """Generate realistic target variable with correlations - vectorized"""
        
        feature_data = df[config.feature_names].values
        n_samples = len(df)
        
        if config.task_type == "classification":
            # Create linear combination of features for logistic relationship
            # Use first 3 features for speed
            n_features_to_use = min(3, len(config.feature_names))
            weights = self.rng.normal(0, 1, n_features_to_use)
            
            # Normalize features for stable computation
            features_subset = feature_data[:, :n_features_to_use]
            features_norm = (features_subset - features_subset.mean(axis=0)) / (features_subset.std(axis=0) + 1e-8)
            
            # Create linear combination
            linear_combination = features_norm @ weights
            
            # Apply logistic function for probabilities
            probabilities = 1 / (1 + np.exp(-linear_combination))
            
            # Generate binary target
            return (probabilities > 0.5).astype(int)
        
        else:  # regression
            # Create linear relationship with noise
            n_features_to_use = min(3, len(config.feature_names))
            weights = self.rng.normal(0, 2, n_features_to_use)
            
            features_subset = feature_data[:, :n_features_to_use]
            target = features_subset @ weights
            
            # Add noise
            noise_std = target.std() * 0.1
            noise = self.rng.normal(0, noise_std, n_samples)
            
            return target + noise
    
    def _add_feature_correlations(self, df: pd.DataFrame, config: SyntheticDataConfig) -> pd.DataFrame:
        """Add realistic correlations between features - optimized"""
        
        # Only add correlations if we have enough features to avoid performance issues
        if len(config.feature_names) < 2:
            return df
        
        # Add some domain-specific correlations
        feature_names = config.feature_names
        
        if config.domain == "healthcare" and len(feature_names) >= 2:
            # BMI correlation with weight/height if present
            if "bmi" in feature_names and "weight" in feature_names:
                weight_col = df["weight"]
                df["bmi"] = 0.7 * weight_col / 70 + self.rng.normal(0, 2, len(df))  # Approximate BMI
        
        elif config.domain == "finance" and len(feature_names) >= 2:
            # Income correlation with credit score
            if "income" in feature_names and "credit_score" in feature_names:
                income_col = df["income"] 
                income_norm = (income_col - income_col.mean()) / income_col.std()
                df["credit_score"] = 650 + 50 * income_norm + self.rng.normal(0, 30, len(df))
                df["credit_score"] = np.clip(df["credit_score"], 300, 850)
        
        return df
    
    def _save_dataset(self, df: pd.DataFrame, config: SyntheticDataConfig) -> str:
        """Save dataset efficiently"""
        
        output_dir = Path("uploads")
        output_dir.mkdir(exist_ok=True)
        
        filename = f"synthetic_{config.domain}_{config.task_type}_{config.n_samples}samples.csv"
        output_path = output_dir / filename
        
        # Use optimized CSV writing
        df.to_csv(output_path, index=False, float_format='%.3f')
        
        # Save metadata for reference
        metadata = {
            "config": {
                "n_samples": config.n_samples,
                "n_features": config.n_features,
                "task_type": config.task_type,
                "domain": config.domain,
                "feature_names": config.feature_names
            },
            "statistical_params": config.statistical_params,
            "file_info": {
                "size_mb": output_path.stat().st_size / (1024 * 1024),
                "shape": list(df.shape)
            }
        }
        
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return str(output_path)

# Global synthetic generator
synthetic_generator = HighPerformanceSyntheticGenerator()
