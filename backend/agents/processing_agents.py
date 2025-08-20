from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from pathlib import Path
import joblib

async def preprocess_dataset(data_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Intelligent preprocessing based on data profile with optimized operations"""
    try:
        dataset_path = data_profile.get("path")
        if not dataset_path:
            return {"status": "error", "message": "No dataset path in profile"}
            
        # Load dataset efficiently
        df = pd.read_csv(dataset_path, low_memory=False)
        original_shape = df.shape
        
        # Store preprocessing steps for reproducibility
        preprocessing_steps = []
        transformers = {}
        
        # 1. Handle missing values (vectorized operations)
        missing_columns = df.columns[df.isnull().any()].tolist()
        if missing_columns:
            for col in missing_columns:
                if df[col].dtype in ['object', 'category']:
                    # Mode imputation for categorical
                    mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                    df[col].fillna(mode_value, inplace=True)
                    preprocessing_steps.append(f"Imputed missing {col} with mode: {mode_value}")
                else:
                    # Median imputation for numerical (robust to outliers)
                    median_value = df[col].median()
                    df[col].fillna(median_value, inplace=True)
                    preprocessing_steps.append(f"Imputed missing {col} with median: {median_value:.2f}")
        
        # 2. Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            removed_dupes = initial_rows - len(df)
            preprocessing_steps.append(f"Removed {removed_dupes} duplicate rows")
        
        # 3. Handle outliers using IQR method (vectorized)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != detect_target_column(df):  # Don't remove outliers from target
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers_count = outliers_mask.sum()
                
                if outliers_count > 0 and outliers_count < len(df) * 0.05:  # Remove if < 5% outliers
                    df = df[~outliers_mask]
                    preprocessing_steps.append(f"Removed {outliers_count} outliers from {col}")
        
        # 4. Encode categorical variables
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        target_col = detect_target_column(df)
        
        for col in categorical_columns:
            if col != target_col:
                unique_values = df[col].nunique()
                
                if unique_values <= 10:  # One-hot encode low cardinality
                    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                    encoded_data = encoder.fit_transform(df[[col]])
                    feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]  # Skip first (dropped)
                    
                    # Replace original column with encoded columns
                    encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)
                    df = df.drop(columns=[col])
                    df = pd.concat([df, encoded_df], axis=1)
                    
                    transformers[f"{col}_encoder"] = encoder
                    preprocessing_steps.append(f"One-hot encoded {col} ({unique_values} categories)")
                    
                else:  # Label encode high cardinality
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col].astype(str))
                    transformers[f"{col}_encoder"] = encoder
                    preprocessing_steps.append(f"Label encoded {col} ({unique_values} categories)")
        
        # 5. Scale numerical features (excluding target)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col != target_col]
        
        if feature_columns:
            scaler = StandardScaler()
            df[feature_columns] = scaler.fit_transform(df[feature_columns])
            transformers['scaler'] = scaler
            preprocessing_steps.append(f"Standardized {len(feature_columns)} numerical features")
        
        # Save preprocessed dataset
        output_path = Path("temp") / "preprocessed_dataset.csv"
        output_path.parent.mkdir(exist_ok=True)
        df.to_csv(output_path, index=False)
        
        # Save transformers
        transformers_path = Path("models") / "preprocessing_transformers.pkl"
        transformers_path.parent.mkdir(exist_ok=True)
        joblib.dump(transformers, transformers_path)
        
        return {
            "status": "success",
            "original_shape": original_shape,
            "processed_shape": df.shape,
            "preprocessing_steps": preprocessing_steps,
            "transformers_saved": str(transformers_path),
            "processed_data_path": str(output_path),
            "columns_after_preprocessing": df.columns.tolist(),
            "target_column": target_col,
            "feature_columns": [col for col in df.columns if col != target_col]
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Preprocessing failed: {str(e)}"}

async def engineer_features(preprocessing_result: Dict[str, Any]) -> Dict[str, Any]:
    """Advanced feature engineering with domain-specific optimizations"""
    try:
        processed_data_path = preprocessing_result.get("processed_data_path")
        if not processed_data_path:
            return {"status": "error", "message": "No processed data path provided"}
            
        df = pd.read_csv(processed_data_path)
        target_col = preprocessing_result.get("target_column")
        feature_columns = preprocessing_result.get("feature_columns", [])
        
        engineering_steps = []
        new_features = []
        
        # 1. Polynomial features for numerical columns (limited to avoid explosion)
        numeric_features = df.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if col != target_col][:5]  # Limit to 5 features
        
        if len(numeric_features) >= 2:
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            
            # Only create interactions to avoid feature explosion
            poly_features = poly.fit_transform(df[numeric_features[:3]])  # Use only top 3 for speed
            feature_names = poly.get_feature_names_out(numeric_features[:3])
            
            # Add only the interaction terms (skip original features)
            interaction_terms = feature_names[len(numeric_features[:3]):]
            interaction_data = poly_features[:, len(numeric_features[:3]):]
            
            for i, name in enumerate(interaction_terms):
                df[f"poly_{name}"] = interaction_data[:, i]
                new_features.append(f"poly_{name}")
            
            engineering_steps.append(f"Created {len(interaction_terms)} polynomial interaction features")
        
        # 2. Statistical features (rolling statistics for time-like data)
        if len(numeric_features) > 0:
            for col in numeric_features[:3]:  # Limit for performance
                # Rolling statistics (if data has some ordering)
                window_size = min(5, len(df) // 10)
                if window_size >= 2:
                    df[f"{col}_rolling_mean"] = df[col].rolling(window=window_size, min_periods=1).mean()
                    df[f"{col}_rolling_std"] = df[col].rolling(window=window_size, min_periods=1).std().fillna(0)
                    new_features.extend([f"{col}_rolling_mean", f"{col}_rolling_std"])
            
            engineering_steps.append(f"Created rolling statistics for {len(numeric_features[:3])} features")
        
        # 3. Binning of numerical features
        for col in numeric_features[:3]:  # Limit for performance
            try:
                # Create quantile-based bins
                df[f"{col}_binned"], bin_edges = pd.cut(df[col], bins=5, retbins=True, labels=False)
                new_features.append(f"{col}_binned")
                engineering_steps.append(f"Created 5 bins for {col}")
            except:
                continue
        
        # 4. Feature selection based on correlation with target
        if target_col and len(df.columns) > 20:  # Only if we have many features
            correlations = df.corr()[target_col].abs().sort_values(ascending=False)
            
            # Keep top 20 features + target
            top_features = correlations.head(21).index.tolist()  # 20 + target
            df = df[top_features]
            
            removed_features = len(feature_columns) + len(new_features) - len(top_features) + 1
            engineering_steps.append(f"Selected top 20 features, removed {removed_features} low-correlation features")
        
        # Save engineered dataset
        output_path = Path("temp") / "engineered_dataset.csv"
        df.to_csv(output_path, index=False)
        
        return {
            "status": "success",
            "engineered_data_path": str(output_path),
            "original_features": len(feature_columns),
            "new_features_created": len(new_features),
            "final_features": len([col for col in df.columns if col != target_col]),
            "engineering_steps": engineering_steps,
            "feature_names": [col for col in df.columns if col != target_col],
            "target_column": target_col
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Feature engineering failed: {str(e)}"}

def detect_target_column(df: pd.DataFrame) -> str:
    """Detect target column efficiently"""
    possible_targets = ['target', 'label', 'class', 'y', 'outcome', 'prediction']
    
    for col in df.columns:
        if col.lower() in possible_targets:
            return col
    
    # Return last column as fallback
    return df.columns[-1] if len(df.columns) > 0 else None

# Preprocessing Agent
preprocessing_agent = LlmAgent(
    name="preprocessing_specialist",
    model="gemini-2.0-flash",
    instruction="""You are a data preprocessing expert. Handle data cleaning and preparation:

1. Handle missing values intelligently based on data types
2. Remove duplicates and outliers appropriately
3. Encode categorical variables (one-hot or label encoding)
4. Scale numerical features
5. Prepare data for machine learning

Use preprocess_dataset tool to clean and prepare the data efficiently.
Always preserve data integrity while optimizing for ML performance.""",
    
    tools=[preprocess_dataset]
)

# Feature Engineering Agent
feature_engineering_agent = LlmAgent(
    name="feature_engineer", 
    model="gemini-2.0-flash",
    instruction="""You are a feature engineering expert. Create valuable features:

1. Generate polynomial and interaction features
2. Create statistical features (rolling statistics)
3. Apply feature binning and discretization
4. Select the most relevant features
5. Optimize feature set for model performance

Use engineer_features tool to create meaningful features that improve model performance.
Balance feature creation with computational efficiency.""",
    
    tools=[engineer_features, google_search]
)
