import pandas as pd
import numpy as np
from pathlib import Path
import json
import datetime
from typing import Dict, Any, List
import textwrap
from loguru import logger

async def generate_complete_project(
    model_results: Dict[str, Any], 
    requirements: Dict[str, Any],
    dataset_info: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Generate complete ML project with all necessary files"""
    
    try:
        logger.info("Starting complete project generation")
        
        # Extract best model information
        best_model_info = _select_best_model(model_results)
        if not best_model_info:
            return {"status": "error", "message": "No successful models found"}
        
        # Create project structure
        project_name = f"{requirements.get('domain', 'general')}_{requirements.get('task_type', 'ml')}_project"
        project_dir = Path("generated") / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all project files
        generated_files = {}
        
        # 1. Model inference code
        inference_code = await _generate_inference_code(best_model_info, requirements, dataset_info)
        inference_path = project_dir / "model_inference.py"
        inference_path.write_text(inference_code)
        generated_files["inference_code"] = str(inference_path)
        
        # 2. Streamlit application
        streamlit_code = await _generate_streamlit_app(best_model_info, requirements, dataset_info)
        streamlit_path = project_dir / "streamlit_app.py"
        streamlit_path.write_text(streamlit_code)
        generated_files["streamlit_app"] = str(streamlit_path)
        
        # 3. FastAPI application
        fastapi_code = await _generate_fastapi_app(best_model_info, requirements)
        fastapi_path = project_dir / "api_server.py"
        fastapi_path.write_text(fastapi_code)
        generated_files["api_server"] = str(fastapi_path)
        
        # 4. Requirements file
        requirements_txt = await _generate_requirements_file(best_model_info, requirements)
        req_path = project_dir / "requirements.txt"
        req_path.write_text(requirements_txt)
        generated_files["requirements"] = str(req_path)
        
        # 5. Docker configuration
        dockerfile, docker_compose = await _generate_docker_files(best_model_info, requirements)
        dockerfile_path = project_dir / "Dockerfile"
        dockerfile_path.write_text(dockerfile)
        generated_files["dockerfile"] = str(dockerfile_path)
        
        compose_path = project_dir / "docker-compose.yml"
        compose_path.write_text(docker_compose)
        generated_files["docker_compose"] = str(compose_path)
        
        # 6. Configuration files
        config_code = await _generate_config_files(best_model_info, requirements)
        config_path = project_dir / "config.py"
        config_path.write_text(config_code)
        generated_files["config"] = str(config_path)
        
        # 7. Testing suite
        test_code = await _generate_test_suite(best_model_info, requirements)
        test_dir = project_dir / "tests"
        test_dir.mkdir(exist_ok=True)
        test_path = test_dir / "test_model.py"
        test_path.write_text(test_code)
        generated_files["tests"] = str(test_path)
        
        # 8. Deployment scripts
        deploy_script = await _generate_deployment_script(best_model_info, requirements)
        deploy_path = project_dir / "deploy.sh"
        deploy_path.write_text(deploy_script)
        deploy_path.chmod(0o755)  # Make executable
        generated_files["deploy_script"] = str(deploy_path)
        
        # 9. Comprehensive README
        readme_content = await _generate_comprehensive_readme(best_model_info, requirements, generated_files)
        readme_path = project_dir / "README.md"
        readme_path.write_text(readme_content)
        generated_files["readme"] = str(readme_path)
        
        # 10. Model monitoring code
        monitoring_code = await _generate_monitoring_code(best_model_info, requirements)
        monitoring_path = project_dir / "model_monitor.py"
        monitoring_path.write_text(monitoring_code)
        generated_files["monitoring"] = str(monitoring_path)
        
        # 11. Data preprocessing pipeline
        preprocessing_code = await _generate_preprocessing_pipeline(dataset_info, requirements)
        preprocessing_path = project_dir / "preprocessing_pipeline.py"
        preprocessing_path.write_text(preprocessing_code)
        generated_files["preprocessing"] = str(preprocessing_path)
        
        # Create additional directories
        (project_dir / "models").mkdir(exist_ok=True)
        (project_dir / "data").mkdir(exist_ok=True)
        (project_dir / "logs").mkdir(exist_ok=True)
        (project_dir / "static").mkdir(exist_ok=True)
        
        # Generate project metadata
        metadata = {
            "project_name": project_name,
            "generated_at": datetime.datetime.now().isoformat(),
            "model_info": best_model_info,
            "requirements": requirements,
            "dataset_info": dataset_info,
            "generated_files": generated_files,
            "project_structure": _get_project_structure(project_dir)
        }
        
        metadata_path = project_dir / "project_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
        
        logger.info(f"Complete project generated successfully: {project_dir}")
        
        return {
            "status": "success",
            "project_directory": str(project_dir),
            "project_name": project_name,
            "generated_files": generated_files,
            "metadata": metadata,
            "file_count": len(generated_files),
            "project_size_mb": _calculate_directory_size(project_dir)
        }
        
    except Exception as e:
        logger.error(f"Project generation failed: {str(e)}")
        return {
            "status": "error",
            "message": f"Project generation failed: {str(e)}"
        }

def _select_best_model(model_results: Dict[str, Any]) -> Dict[str, Any]:
    """Select the best performing model from results"""
    
    best_model = None
    best_score = -1
    
    if isinstance(model_results, dict):
        for model_name, result in model_results.items():
            if isinstance(result, dict) and result.get("status") == "success":
                performance = result.get("performance", {})
                score = performance.get("test_score", 0)
                
                if score > best_score:
                    best_score = score
                    best_model = result.copy()
                    best_model["model_name"] = model_name
    
    return best_model

async def _generate_inference_code(model_info: Dict[str, Any], requirements: Dict[str, Any], dataset_info: Dict[str, Any] = None) -> str:
    """Generate production-ready model inference code"""
    
    model_type = model_info.get("model_info", {}).get("type", "unknown")
    task_type = model_info.get("model_info", {}).get("task_type", "classification")
    domain = requirements.get("domain", "general")
    
    # Extract feature information
    feature_info = model_info.get("data_info", {})
    features = feature_info.get("features", 10)
    
    code = f'''"""
Production ML Model Inference Module
Generated automatically by ML Automation Platform

Model: {model_type.title()}
Task: {task_type.title()}
Domain: {domain.title()}
Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLModelInference:
    """
    Production-ready ML model inference class for {domain} {task_type}
    
    Features:
    - Robust error handling and validation
    - Automatic preprocessing pipeline
    - Comprehensive logging
    - Model performance monitoring
    - Batch and single prediction support
    """
    
    def __init__(self, model_path: str, preprocessing_pipeline_path: Optional[str] = None):
        """
        Initialize the ML model inference
        
        Args:
            model_path: Path to the trained model file
            preprocessing_pipeline_path: Path to preprocessing pipeline (optional)
        """
        self.model_path = Path(model_path)
        self.preprocessing_pipeline_path = Path(preprocessing_pipeline_path) if preprocessing_pipeline_path else None
        self.model = None
        self.preprocessing_pipeline = None
        self.model_info = {{
            "type": "{model_type}",
            "task_type": "{task_type}",
            "domain": "{domain}",
            "expected_features": {features}
        }}
        
        self._load_model()
        self._load_preprocessing_pipeline()
        
        logger.info(f"MLModelInference initialized for {{self.model_info['type']}} {{self.model_info['task_type']}}")
    
    def _load_model(self):
        """Load the trained model"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {{self.model_path}}")
            
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from {{self.model_path}}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {{str(e)}}")
            raise
    
    def _load_preprocessing_pipeline(self):
        """Load preprocessing pipeline if available"""
        try:
            if self.preprocessing_pipeline_path and self.preprocessing_pipeline_path.exists():
                self.preprocessing_pipeline = joblib.load(self.preprocessing_pipeline_path)
                logger.info(f"Preprocessing pipeline loaded from {{self.preprocessing_pipeline_path}}")
            else:
                logger.warning("No preprocessing pipeline provided - using raw input")
                
        except Exception as e:
            logger.warning(f"Failed to load preprocessing pipeline: {{str(e)}}")
    
    def _validate_input(self, data: Union[Dict, List, pd.DataFrame]) -> pd.DataFrame:
        """Validate and convert input data to DataFrame"""
        try:
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                if len(data) == 0:
                    raise ValueError("Empty input list")
                if isinstance(data[0], dict):
                    df = pd.DataFrame(data)
                else:
                    # Assume single feature vector
                    df = pd.DataFrame([data])
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                raise ValueError(f"Unsupported input type: {{type(data)}}")
            
            # Basic validation
            if df.empty:
                raise ValueError("Empty DataFrame provided")
            
            expected_features = self.model_info["expected_features"]
            if len(df.columns) != expected_features:
                logger.warning(f"Expected {{expected_features}} features, got {{len(df.columns)}}")
            
            return df
            
        except Exception as e:
            logger.error(f"Input validation failed: {{str(e)}}")
            raise
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing pipeline to data"""
        try:
            if self.preprocessing_pipeline:
                # Apply preprocessing pipeline
                processed_df = self.preprocessing_pipeline.transform(df)
                if isinstance(processed_df, np.ndarray):
                    processed_df = pd.DataFrame(processed_df, columns=df.columns)
                return processed_df
            else:
                # Basic preprocessing
                # Handle missing values
                df_processed = df.copy()
                
                # Fill missing values with median for numeric, mode for categorical
                for col in df_processed.columns:
                    if df_processed[col].dtype in ['object', 'category']:
                        mode_value = df_processed[col].mode()
                        if len(mode_value) > 0:
                            df_processed[col].fillna(mode_value.iloc[0], inplace=True)
                        else:
                            df_processed[col].fillna("unknown", inplace=True)
                    else:
                        df_processed[col].fillna(df_processed[col].median(), inplace=True)
                
                return df_processed
                
        except Exception as e:
            logger.error(f"Preprocessing failed: {{str(e)}}")
            raise
    
    def predict(self, data: Union[Dict, List, pd.DataFrame], return_probabilities: bool = False) -> Dict[str, Any]:
        """
        Make predictions on new data
        
        Args:
            data: Input data (dict, list, or DataFrame)
            return_probabilities: Whether to return prediction probabilities (classification only)
            
        Returns:
            Dictionary containing predictions and metadata
        """
        try:
            prediction_start = datetime.now()
            
            # Validate and convert input
            df = self._validate_input(data)
            input_shape = df.shape
            
            # Preprocess data
            df_processed = self._preprocess_data(df)
            
            # Make predictions
            predictions = self.model.predict(df_processed)
            
            result = {{
                "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
                "input_shape": input_shape,
                "prediction_time": (datetime.now() - prediction_start).total_seconds(),
                "model_type": self.model_info["type"],
                "task_type": self.model_info["task_type"]
            }}
            
            # Add probabilities for classification
            if self.model_info["task_type"] == "classification" and (return_probabilities or hasattr(self.model, 'predict_proba')):
                try:
                    if hasattr(self.model, 'predict_proba'):
                        probabilities = self.model.predict_proba(df_processed)
                        result["probabilities"] = probabilities.tolist()
                        result["confidence"] = np.max(probabilities, axis=1).tolist()
                    
                    # Add class labels if available
                    if hasattr(self.model, 'classes_'):
                        result["class_labels"] = self.model.classes_.tolist()
                        
                except Exception as e:
                    logger.warning(f"Could not get probabilities: {{str(e)}}")
            
            # Add prediction intervals for regression
            elif self.model_info["task_type"] == "regression":
                try:
                    # Simple prediction interval estimation (assuming normal distribution)
                    if hasattr(self.model, 'predict'):
                        # This is a simple approximation - in production, use proper methods
                        pred_std = np.std(predictions) if len(predictions) > 1 else 0.1
                        lower_bound = predictions - 1.96 * pred_std
                        upper_bound = predictions + 1.96 * pred_std
                        
                        result["prediction_intervals"] = {{
                            "lower_bound": lower_bound.tolist() if isinstance(lower_bound, np.ndarray) else [lower_bound],
                            "upper_bound": upper_bound.tolist() if isinstance(upper_bound, np.ndarray) else [upper_bound],
                            "confidence_level": 0.95
                        }}
                except Exception as e:
                    logger.warning(f"Could not calculate prediction intervals: {{str(e)}}")
            
            logger.info(f"Prediction completed for {{len(df)}} samples in {{result['prediction_time']:.3f}}s")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {{str(e)}}")
            return {{
                "error": str(e),
                "predictions": None,
                "model_type": self.model_info["type"],
                "task_type": self.model_info["task_type"]
            }}
    
    def batch_predict(self, data_list: List[Union[Dict, List]], batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Efficient batch prediction processing
        
        Args:
            data_list: List of input samples
            batch_size: Number of samples to process in each batch
            
        Returns:
            List of prediction results
        """
        try:
            results = []
            total_samples = len(data_list)
            
            logger.info(f"Starting batch prediction for {{total_samples}} samples")
            
            for i in range(0, total_samples, batch_size):
                batch = data_list[i:i + batch_size]
                batch_result = self.predict(batch)
                
                if "error" not in batch_result:
                    # Split batch results into individual predictions
                    predictions = batch_result["predictions"]
                    probabilities = batch_result.get("probabilities", [None] * len(batch))
                    confidence = batch_result.get("confidence", [None] * len(batch))
                    
                    for j, pred in enumerate(predictions):
                        individual_result = {{
                            "prediction": pred,
                            "probability": probabilities[j] if j < len(probabilities) else None,
                            "confidence": confidence[j] if j < len(confidence) else None,
                            "batch_index": i + j
                        }}
                        results.append(individual_result)
                else:
                    # Handle batch error
                    for j in range(len(batch)):
                        results.append({{"error": batch_result["error"], "batch_index": i + j}})
                
                # Progress logging
                if (i + batch_size) % (batch_size * 10) == 0:
                    logger.info(f"Processed {{min(i + batch_size, total_samples)}}/{{total_samples}} samples")
            
            logger.info(f"Batch prediction completed for {{total_samples}} samples")
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {{str(e)}}")
            return [{{"error": str(e)}}] * len(data_list)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        info = self.model_info.copy()
        
        try:
            # Add model-specific information
            if hasattr(self.model, 'feature_importances_'):
                info["has_feature_importance"] = True
            
            if hasattr(self.model, 'classes_'):
                info["classes"] = self.model.classes_.tolist()
                info["n_classes"] = len(self.model.classes_)
            
            if hasattr(self.model, 'n_features_in_'):
                info["n_features_in"] = self.model.n_features_in_
            
            # Model parameters
            if hasattr(self.model, 'get_params'):
                info["model_parameters"] = self.model.get_params()
            
            info["model_loaded"] = self.model is not None
            info["preprocessing_available"] = self.preprocessing_pipeline is not None
            
        except Exception as e:
            logger.warning(f"Could not get full model info: {{str(e)}}")
        
        return info
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                # This assumes features are named as feature_0, feature_1, etc.
                # In practice, you'd want to use actual feature names
                n_features = len(self.model.feature_importances_)
                feature_names = [f"feature_{{i}}" for i in range(n_features)]
                
                importance_dict = dict(zip(feature_names, self.model.feature_importances_))
                # Sort by importance
                sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
                
                return sorted_importance
            else:
                logger.warning("Model does not have feature importance")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get feature importance: {{str(e)}}")
            return None

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    try:
        # Initialize model (replace with actual model path)
        model_path = "models/trained_model.pkl"
        inference = MLModelInference(model_path)
        
        # Example prediction
        sample_data = {{
            # Add sample feature values here based on your model
            "feature_0": 1.0,
            "feature_1": 2.0,
            # ... more features
        }}
        
        # Single prediction
        result = inference.predict(sample_data, return_probabilities=True)
        print("Prediction result:", json.dumps(result, indent=2))
        
        # Get model info
        model_info = inference.get_model_info()
        print("Model info:", json.dumps(model_info, indent=2))
        
        # Get feature importance
        feature_importance = inference.get_feature_importance()
        if feature_importance:
            print("Feature importance:", json.dumps(feature_importance, indent=2))
        
    except Exception as e:
        print(f"Example execution failed: {{str(e)}}")
'''
    
    return textwrap.dedent(code)

async def _generate_streamlit_app(model_info: Dict[str, Any], requirements: Dict[str, Any], dataset_info: Dict[str, Any] = None) -> str:
    """Generate comprehensive Streamlit application"""
    
    model_type = model_info.get("model_info", {}).get("type", "unknown")
    task_type = model_info.get("model_info", {}).get("task_type", "classification")
    domain = requirements.get("domain", "general")
    
    performance = model_info.get("performance", {})
    test_score = performance.get("test_score", 0)
    
    code = f'''"""
Interactive ML Model Application
Built with Streamlit

Model: {model_type.title()}
Task: {task_type.title()}
Domain: {domain.title()}
Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime
from pathlib import Path
import sys

# Add current directory to path to import our model
sys.path.append(str(Path(__file__).parent))

try:
    from model_inference import MLModelInference
except ImportError:
    st.error("Could not import MLModelInference. Make sure model_inference.py is in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="{domain.title()} {task_type.title()} Model",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {{
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    padding: 1rem;
    background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
    border-radius: 10px;
}}

.metric-card {{
    background: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
}}

.prediction-result {{
    font-size: 1.5rem;
    font-weight: bold;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}}

.success-result {{
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}}

.warning-result {{
    background-color: #fff3cd;
    color: #856404;
    border: 1px solid #ffeaa7;
}}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the ML model with caching"""
    try:
        model_path = "models/trained_model.pkl"
        if not Path(model_path).exists():
            st.error(f"Model file not found: {{model_path}}")
            return None
        
        model = MLModelInference(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {{str(e)}}")
        return None

def create_sample_data():
    """Create sample data for demonstration"""
    # This would be customized based on your actual features
    sample_data = {{
        "feature_0": 1.0,
        "feature_1": 2.0,
        "feature_2": 3.0,
        # Add more features based on your model
    }}
    return sample_data

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        🤖 {domain.title()} {task_type.title()} Model
        <br><small>AI-Powered {task_type.title()} Solution</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Sidebar with model information
    with st.sidebar:
        st.header("📊 Model Information")
        
        model_info = model.get_model_info()
        
        st.metric("Model Type", "{model_type.title()}")
        st.metric("Task Type", "{task_type.title()}")
        st.metric("Domain", "{domain.title()}")
        st.metric("Test Score", f"{test_score:.3f}")
        
        if model_info.get("n_classes"):
            st.metric("Classes", model_info["n_classes"])
        
        st.markdown("---")
        
        # Model performance section
        st.subheader("🎯 Performance Metrics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Accuracy" if "{task_type}" == "classification" else "R² Score",
                f"{test_score:.3f}",
                delta=f"{test_score - 0.5:.3f}" if test_score > 0.5 else None
            )
        
        with col2:
            cv_mean = {performance.get('cv_mean', 0):.3f}
            st.metric("CV Score", f"{cv_mean}")
        
        # Feature importance
        feature_importance = model.get_feature_importance()
        if feature_importance:
            st.subheader("📈 Feature Importance")
            
            # Convert to DataFrame for plotting
            importance_df = pd.DataFrame([
                {{"feature": k, "importance": v}}
                for k, v in list(feature_importance.items())[:10]  # Top 10
            ])
            
            fig = px.bar(
                importance_df,
                x="importance",
                y="feature",
                orientation="h",
                title="Top 10 Important Features"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Make Predictions", 
        "📊 Batch Processing", 
        "📈 Model Analytics", 
        "🔧 Model Testing"
    ])
    
    with tab1:
        st.header("Make Individual Predictions")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Input Features")
            
            # Create input fields based on model features
            # This is a simplified example - you'd customize based on actual features
            input_data = {{}}
            
            # Example feature inputs (customize based on your model)
            feature_cols = st.columns(3)
            
            with feature_cols[0]:
                input_data["feature_0"] = st.number_input("Feature 1", value=0.0, step=0.1)
                input_data["feature_1"] = st.number_input("Feature 2", value=0.0, step=0.1)
            
            with feature_cols[1]:
                input_data["feature_2"] = st.number_input("Feature 3", value=0.0, step=0.1)
                input_data["feature_3"] = st.number_input("Feature 4", value=0.0, step=0.1)
            
            with feature_cols[2]:
                input_data["feature_4"] = st.number_input("Feature 5", value=0.0, step=0.1)
                
                # Add sample data button
                if st.button("Use Sample Data", type="secondary"):
                    sample = create_sample_data()
                    st.session_state.update(sample)
                    st.rerun()
            
            # Prediction button
            if st.button("🚀 Make Prediction", type="primary"):
                with st.spinner("Making prediction..."):
                    result = model.predict(input_data, return_probabilities=True)
                
                if "error" in result:
                    st.error(f"Prediction failed: {{result['error']}}")
                else:
                    prediction = result["predictions"][0]
                    
                    if "{task_type}" == "classification":
                        st.markdown(f"""
                        <div class="prediction-result success-result">
                            Predicted Class: {{prediction}}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if "probabilities" in result:
                            probs = result["probabilities"][0]
                            confidence = max(probs)
                            
                            st.metric("Confidence", f"{{confidence:.1%}}")
                            
                            # Probability visualization
                            if "class_labels" in result:
                                classes = result["class_labels"]
                                prob_df = pd.DataFrame({{
                                    "Class": classes,
                                    "Probability": probs
                                }})
                                
                                fig = px.bar(
                                    prob_df,
                                    x="Class",
                                    y="Probability",
                                    title="Class Probabilities"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    
                    else:  # regression
                        st.markdown(f"""
                        <div class="prediction-result success-result">
                            Predicted Value: {{prediction:.3f}}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if "prediction_intervals" in result:
                            intervals = result["prediction_intervals"]
                            lower = intervals["lower_bound"][0]
                            upper = intervals["upper_bound"]
                            
                            st.metric(
                                "Prediction Interval (95%)",
                                f"[{{lower:.3f}}, {{upper:.3f}}]"
                            )
        
        with col2:
            st.subheader("Prediction Details")
            
            if "result" in locals():
                st.json({{
                    "prediction_time": f"{{result.get('prediction_time', 0):.4f}}s",
                    "model_type": result.get("model_type"),
                    "input_shape": result.get("input_shape")
                }})
    
    with tab2:
        st.header("Batch Prediction Processing")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file for batch predictions",
            type="csv",
            help="Upload a CSV file with the same features as training data"
        )
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {{len(batch_df)}} rows for prediction")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(batch_df.head(), use_container_width=True)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    batch_size = st.slider("Batch Size", min_value=10, max_value=1000, value=100)
                
                with col2:
                    if st.button("🚀 Run Batch Predictions", type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        start_time = time.time()
                        
                        # Convert DataFrame to list of dictionaries
                        data_list = batch_df.to_dict('records')
                        
                        # Run batch prediction
                        batch_results = model.batch_predict(data_list, batch_size=batch_size)
                        
                        processing_time = time.time() - start_time
                        
                        progress_bar.progress(1.0)
                        status_text.success(f"Completed {{len(batch_results)}} predictions in {{processing_time:.2f}}s")
                        
                        # Process results
                        predictions = [r.get("prediction") for r in batch_results]
                        batch_df["prediction"] = predictions
                        
                        if "{task_type}" == "classification":
                            confidences = [r.get("confidence") for r in batch_results]
                            batch_df["confidence"] = confidences
                        
                        # Show results
                        st.subheader("Prediction Results")
                        st.dataframe(batch_df, use_container_width=True)
                        
                        # Download results
                        csv = batch_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv,
                            f"predictions_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}.csv",
                            "text/csv"
                        )
                        
                        # Results visualization
                        if "{task_type}" == "classification":
                            fig = px.histogram(
                                batch_df,
                                x="prediction",
                                title="Prediction Distribution"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            fig = px.histogram(
                                batch_df,
                                x="prediction",
                                title="Prediction Distribution",
                                nbins=50
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
            except Exception as e:
                st.error(f"Error processing batch file: {{str(e)}}")
    
    with tab3:
        st.header("Model Analytics & Insights")
        
        # Model overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Accuracy", f"{test_score:.3f}")
        
        with col2:
            st.metric("CV Score", f"{performance.get('cv_mean', 0):.3f}")
        
        with col3:
            st.metric("CV Std Dev", f"{performance.get('cv_std', 0):.3f}")
        
        with col4:
            overfitting = performance.get('overfitting_score', 0)
            st.metric("Overfitting Score", f"{overfitting:.3f}")
        
        # Performance visualization
        if performance:
            st.subheader("📈 Performance Analysis")
            
            # Create performance comparison chart
            metrics_data = {{
                "Metric": ["Train Score", "Test Score", "CV Mean"],
                "Value": [
                    performance.get("train_score", 0),
                    performance.get("test_score", 0),
                    performance.get("cv_mean", 0)
                ]
            }}
            
            fig = px.bar(
                metrics_data,
                x="Metric",
                y="Value",
                title="Model Performance Comparison",
                color="Value",
                color_continuous_scale="viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Cross-validation scores
            if "cv_scores" in performance:
                cv_scores = performance["cv_scores"]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(cv_scores))),
                    y=cv_scores,
                    mode='lines+markers',
                    name='CV Scores',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title="Cross-Validation Scores",
                    xaxis_title="Fold",
                    yaxis_title="Score"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Model Testing & Validation")
        
        st.subheader("🧪 Model Health Check")
        
        if st.button("Run Health Check"):
            with st.spinner("Running diagnostics..."):
                # Model info test
                info = model.get_model_info()
                
                checks = []
                
                # Check if model is loaded
                checks.append({{
                    "test": "Model Loading",
                    "status": "✅ Pass" if info.get("model_loaded") else "❌ Fail",
                    "details": "Model loaded successfully" if info.get("model_loaded") else "Model not loaded"
                }})
                
                # Check feature importance
                importance = model.get_feature_importance()
                checks.append({{
                    "test": "Feature Importance",
                    "status": "✅ Pass" if importance else "⚠️ N/A",
                    "details": f"{{len(importance)}} features" if importance else "Not available for this model type"
                }})
                
                # Test prediction
                try:
                    sample_data = create_sample_data()
                    result = model.predict(sample_data)
                    pred_status = "✅ Pass" if "error" not in result else "❌ Fail"
                    pred_details = f"Prediction time: {{result.get('prediction_time', 0):.4f}}s" if "error" not in result else result.get("error", "Unknown error")
                except Exception as e:
                    pred_status = "❌ Fail"
                    pred_details = str(e)
                
                checks.append({{
                    "test": "Sample Prediction",
                    "status": pred_status,
                    "details": pred_details
                }})
                
                # Display results
                for check in checks:
                    col1, col2, col3 = st.columns([2, 1, 3])
                    with col1:
                        st.write(check["test"])
                    with col2:
                        st.write(check["status"])
                    with col3:
                        st.write(check["details"])
        
        st.markdown("---")
        
        st.subheader("📋 Model Information")
        
        # Display detailed model information
        model_info = model.get_model_info()
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.json({{
                "Model Type": model_info.get("type"),
                "Task Type": model_info.get("task_type"),
                "Domain": model_info.get("domain"),
                "Expected Features": model_info.get("expected_features")
            }})
        
        with info_col2:
            if "model_parameters" in model_info:
                st.json(model_info["model_parameters"])
            else:
                st.info("Model parameters not available")

if __name__ == "__main__":
    main()
'''
    
    return textwrap.dedent(code)

async def _generate_requirements_file(model_info: Dict[str, Any], requirements: Dict[str, Any]) -> str:
    """Generate comprehensive requirements.txt file"""
    
    base_requirements = [
        "# Core ML and Data Processing",
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.4.0",
        "joblib>=1.3.2",
        "",
        "# Model-specific libraries",
    ]
    
    model_type = model_info.get("model_info", {}).get("type", "unknown")
    
    if model_type == "xgboost":
        base_requirements.append("xgboost>=2.0.3")
    elif model_type == "lightgbm":
        base_requirements.append("lightgbm>=4.2.0")
    elif model_type == "random_forest":
        pass  # Already included in scikit-learn
    
    base_requirements.extend([
        "",
        "# Web Applications",
        "streamlit>=1.31.0",
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "",
        "# Visualization",
        "plotly>=5.17.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.12.0",
        "",
        "# Utilities",
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.0",
        "loguru>=0.7.2",
        "python-multipart>=0.0.6",
        "",
        "# Optional: Advanced features",
        "optuna>=3.5.0  # For hyperparameter optimization",
        "shap>=0.44.0  # For model explainability",
        "lime>=0.2.0.1  # For local interpretability",
        "",
        "# Development and Testing",
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "",
        "# Production deployment",
        "gunicorn>=21.2.0",
        "docker>=6.1.0"
    ])
    
    return "\n".join(base_requirements)

def _calculate_directory_size(directory: Path) -> float:
    """Calculate directory size in MB"""
    total_size = 0
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    return total_size / (1024 * 1024)

def _get_project_structure(project_dir: Path) -> List[str]:
    """Get project directory structure"""
    structure = []
    
    def add_to_structure(path: Path, prefix: str = ""):
        items = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            structure.append(f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir() and not item.name.startswith('.'):
                extension = "    " if is_last else "│   "
                add_to_structure(item, prefix + extension)
    
    add_to_structure(project_dir)
    return structure
# ... (continuing from previous code in code_generation_tools.py)

async def _generate_fastapi_app(model_info: Dict[str, Any], requirements: Dict[str, Any]) -> str:
    """Generate FastAPI application for model serving"""
    
    model_type = model_info.get("model_info", {}).get("type", "unknown")
    task_type = model_info.get("model_info", {}).get("task_type", "classification")
    domain = requirements.get("domain", "general")
    
    code = f'''"""
FastAPI Model Serving Application
Generated automatically by ML Automation Platform

Model: {model_type.title()}
Task: {task_type.title()}
Domain: {domain.title()}
Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
import asyncio
import uuid
from contextlib import asynccontextmanager
import io

# Import our model inference
try:
    from model_inference import MLModelInference
except ImportError:
    raise ImportError("model_inference.py not found. Make sure it's in the same directory.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model_instance: Optional[MLModelInference] = None

# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Single prediction request"""
    data: Dict[str, Union[float, int, str]] = Field(..., description="Feature values for prediction")
    return_probabilities: bool = Field(False, description="Return prediction probabilities (classification only)")
    
    class Config:
        schema_extra = {{
            "example": {{
                "data": {{
                    "feature_0": 1.0,
                    "feature_1": 2.0,
                    "feature_2": 3.0
                }},
                "return_probabilities": True
            }}
        }}

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    data: List[Dict[str, Union[float, int, str]]] = Field(..., description="List of feature dictionaries")
    batch_size: int = Field(100, description="Batch processing size", ge=1, le=1000)
    
    class Config:
        schema_extra = {{
            "example": {{
                "data": [
                    {{"feature_0": 1.0, "feature_1": 2.0}},
                    {{"feature_0": 2.0, "feature_1": 3.0}}
                ],
                "batch_size": 100
            }}
        }}

class PredictionResponse(BaseModel):
    """Prediction response"""
    predictions: List[Union[float, int, str]]
    model_info: Dict[str, Any]
    prediction_time: float
    probabilities: Optional[List[List[float]]] = None
    confidence: Optional[List[float]] = None
    prediction_intervals: Optional[Dict[str, Any]] = None

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    results: List[Dict[str, Any]]
    total_predictions: int
    processing_time: float
    batch_info: Dict[str, Any]

class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_type: str
    task_type: str
    domain: str
    performance_metrics: Dict[str, Any]
    feature_info: Dict[str, Any]
    model_status: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    uptime: float
    version: str
    timestamp: str

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    # Startup
    global model_instance
    
    logger.info("Starting {domain} {task_type} API server...")
    
    try:
        # Load model
        model_path = "models/trained_model.pkl"
        if not Path(model_path).exists():
            logger.error(f"Model file not found: {{model_path}}")
            raise FileNotFoundError(f"Model file not found: {{model_path}}")
        
        model_instance = MLModelInference(model_path)
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {{str(e)}}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server...")
    model_instance = None

# Initialize FastAPI app
app = FastAPI(
    title="{domain.title()} {task_type.title()} API",
    description="Production ML model serving API with comprehensive features",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store for background tasks
background_tasks_status = {{}}

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information"""
    return {{
        "message": "Welcome to {domain.title()} {task_type.title()} ML API",
        "model_type": "{model_type}",
        "task_type": "{task_type}",
        "domain": "{domain}",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model/info"
    }}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    start_time = getattr(app.state, 'start_time', datetime.now())
    uptime = (datetime.now() - start_time).total_seconds()
    
    return HealthResponse(
        status="healthy" if model_instance else "unhealthy",
        model_loaded=model_instance is not None,
        uptime=uptime,
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get comprehensive model information"""
    if not model_instance:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        model_info = model_instance.get_model_info()
        
        return ModelInfoResponse(
            model_type=model_info.get("type", "unknown"),
            task_type=model_info.get("task_type", "unknown"),
            domain=model_info.get("domain", "unknown"),
            performance_metrics={{
                "test_score": {model_info.get("performance", {}).get("test_score", 0)},
                "cv_mean": {model_info.get("performance", {}).get("cv_mean", 0)},
                "cv_std": {model_info.get("performance", {}).get("cv_std", 0)}
            }},
            feature_info={{
                "expected_features": model_info.get("expected_features", 0),
                "has_feature_importance": model_info.get("has_feature_importance", False),
                "n_classes": model_info.get("n_classes")
            }},
            model_status="loaded"
        )
        
    except Exception as e:
        logger.error(f"Failed to get model info: {{str(e)}}")
        raise HTTPException(status_code=500, detail="Failed to get model information")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make single prediction"""
    if not model_instance:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = datetime.now()
        
        # Make prediction
        result = model_instance.predict(
            request.data, 
            return_probabilities=request.return_probabilities
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PredictionResponse(
            predictions=result["predictions"],
            model_info={{
                "type": result.get("model_type"),
                "task_type": result.get("task_type")
            }},
            prediction_time=processing_time,
            probabilities=result.get("probabilities"),
            confidence=result.get("confidence"),
            prediction_intervals=result.get("prediction_intervals")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {{str(e)}}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {{str(e)}}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions"""
    if not model_instance:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.data) > 10000:  # Limit batch size
        raise HTTPException(status_code=413, detail="Batch size too large (max 10000)")
    
    try:
        start_time = datetime.now()
        
        # Process batch
        results = model_instance.batch_predict(request.data, request.batch_size)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchPredictionResponse(
            results=results,
            total_predictions=len(results),
            processing_time=processing_time,
            batch_info={{
                "batch_size": request.batch_size,
                "input_samples": len(request.data),
                "successful_predictions": len([r for r in results if "error" not in r])
            }}
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {{str(e)}}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {{str(e)}}")

@app.post("/predict/file")
async def predict_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload file for batch prediction processing"""
    if not model_instance:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file
    if not file.filename.endswith(('.csv', '.json')):
        raise HTTPException(status_code=400, detail="Only CSV and JSON files supported")
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    try:
        # Read file content
        content = await file.read()
        
        # Start background processing
        background_tasks.add_task(
            process_file_predictions, 
            task_id, 
            content, 
            file.filename,
            file.content_type
        )
        
        # Store task status
        background_tasks_status[task_id] = {{
            "status": "processing",
            "filename": file.filename,
            "started_at": datetime.now().isoformat(),
            "progress": 0
        }}
        
        return {{
            "task_id": task_id,
            "status": "processing",
            "message": f"File {{file.filename}} uploaded successfully. Processing started.",
            "check_status_url": f"/predict/file/status/{{task_id}}"
        }}
        
    except Exception as e:
        logger.error(f"File upload failed: {{str(e)}}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {{str(e)}}")

@app.get("/predict/file/status/{{task_id}}")
async def get_file_prediction_status(task_id: str):
    """Get status of file prediction task"""
    if task_id not in background_tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return background_tasks_status[task_id]

@app.get("/predict/file/download/{{task_id}}")
async def download_predictions(task_id: str):
    """Download prediction results"""
    if task_id not in background_tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_status = background_tasks_status[task_id]
    
    if task_status["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed yet")
    
    result_file = task_status.get("result_file")
    if not result_file or not Path(result_file).exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(
        result_file,
        filename=f"predictions_{{task_id}}.csv",
        media_type="text/csv"
    )

async def process_file_predictions(task_id: str, content: bytes, filename: str, content_type: str):
    """Background task for processing file predictions"""
    try:
        # Update status
        background_tasks_status[task_id]["status"] = "processing"
        background_tasks_status[task_id]["progress"] = 10
        
        # Parse file
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode()))
        elif filename.endswith('.json'):
            df = pd.read_json(io.StringIO(content.decode()))
        else:
            raise ValueError("Unsupported file format")
        
        background_tasks_status[task_id]["progress"] = 30
        
        # Convert to list of dictionaries
        data_list = df.to_dict('records')
        
        background_tasks_status[task_id]["progress"] = 50
        background_tasks_status[task_id]["total_records"] = len(data_list)
        
        # Process predictions
        results = model_instance.batch_predict(data_list)
        
        background_tasks_status[task_id]["progress"] = 80
        
        # Create results DataFrame
        predictions = [r.get("prediction") for r in results]
        df["prediction"] = predictions
        
        if "{task_type}" == "classification":
            confidences = [r.get("confidence") for r in results]
            df["confidence"] = confidences
        
        # Save results
        result_file = f"temp/predictions_{{task_id}}.csv"
        Path("temp").mkdir(exist_ok=True)
        df.to_csv(result_file, index=False)
        
        # Update final status
        background_tasks_status[task_id].update({{
            "status": "completed",
            "progress": 100,
            "completed_at": datetime.now().isoformat(),
            "result_file": result_file,
            "successful_predictions": len([r for r in results if "error" not in r])
        }})
        
    except Exception as e:
        background_tasks_status[task_id].update({{
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        }})
        logger.error(f"File processing failed for task {{task_id}}: {{str(e)}}")

@app.get("/model/feature-importance")
async def get_feature_importance():
    """Get model feature importance"""
    if not model_instance:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        feature_importance = model_instance.get_feature_importance()
        
        if not feature_importance:
            raise HTTPException(status_code=404, detail="Feature importance not available for this model")
        
        return {{
            "feature_importance": feature_importance,
            "top_features": list(feature_importance.keys())[:10],
            "model_type": model_instance.model_info["type"]
        }}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get feature importance: {{str(e)}}")
        raise HTTPException(status_code=500, detail="Failed to get feature importance")

@app.get("/metrics")
async def get_metrics():
    """Get API metrics and statistics"""
    return {{
        "api_version": "1.0.0",
        "model_info": {{
            "type": "{model_type}",
            "task_type": "{task_type}",
            "domain": "{domain}"
        }},
        "active_tasks": len([t for t in background_tasks_status.values() if t["status"] == "processing"]),
        "completed_tasks": len([t for t in background_tasks_status.values() if t["status"] == "completed"]),
        "failed_tasks": len([t for t in background_tasks_status.values() if t["status"] == "failed"]),
        "uptime": (datetime.now() - getattr(app.state, 'start_time', datetime.now())).total_seconds()
    }}

# Set start time
app.state.start_time = datetime.now()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
'''
    
    return textwrap.dedent(code)

async def _generate_docker_files(model_info: Dict[str, Any], requirements: Dict[str, Any]) -> tuple:
    """Generate Docker configuration files"""
    
    # Dockerfile
    dockerfile = f'''# Multi-stage Docker build for ML model deployment
# Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/* \\
    && useradd --create-home --shell /bin/bash app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/app/.local

# Copy application code
COPY . .

# Set ownership and permissions
RUN chown -R app:app /app
USER app

# Set Python path
ENV PATH=/home/app/.local/bin:$PATH
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "api_server.py"]
'''
    
    # docker-compose.yml
    docker_compose = f'''version: '3.8'

services:
  ml-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: {requirements.get("domain", "ml")}-{requirements.get("task_type", "model")}-api
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
      - ENVIRONMENT=production
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
      - ./temp:/app/temp
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    
  streamlit-app:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    container_name: {requirements.get("domain", "ml")}-{requirements.get("task_type", "model")}-app
    ports:
      - "8501:8501"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ./models:/app/models:ro
    restart: unless-stopped
    depends_on:
      - ml-api
    command: ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]

  nginx:
    image: nginx:alpine
    container_name: {requirements.get("domain", "ml")}-{requirements.get("task_type", "model")}-proxy
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    restart: unless-stopped
    depends_on:
      - ml-api
      - streamlit-app

networks:
  default:
    name: ml-network
'''
    
    return dockerfile, docker_compose

async def _generate_config_files(model_info: Dict[str, Any], requirements: Dict[str, Any]) -> str:
    """Generate configuration file"""
    
    code = f'''"""
Configuration settings for the ML application
Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application info
    app_name: str = "{requirements.get('domain', 'ML').title()} {requirements.get('task_type', 'Model').title()} API"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Model configuration
    model_path: str = Field(default="models/trained_model.pkl", env="MODEL_PATH")
    preprocessing_path: Optional[str] = Field(default=None, env="PREPROCESSING_PATH")
    model_type: str = "{model_info.get('model_info', {}).get('type', 'unknown')}"
    task_type: str = "{model_info.get('model_info', {}).get('task_type', 'classification')}"
    
    # API configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    
    # Performance settings
    max_batch_size: int = Field(default=10000, env="MAX_BATCH_SIZE")
    request_timeout: int = Field(default=300, env="REQUEST_TIMEOUT")  # seconds
    max_file_size: int = Field(default=100, env="MAX_FILE_SIZE_MB")  # MB
    
    # Logging configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/app.log", env="LOG_FILE")
    
    # Security settings
    cors_origins: str = Field(default="*", env="CORS_ORIGINS")
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    
    # Database settings (if needed)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # Monitoring settings
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Cache settings
    cache_predictions: bool = Field(default=False, env="CACHE_PREDICTIONS")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # seconds
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create global settings instance
settings = Settings()

# Model performance thresholds
PERFORMANCE_THRESHOLDS = {{
    "classification": {{
        "min_accuracy": 0.7,
        "max_prediction_time": 1.0,  # seconds
        "max_memory_usage": 1024,    # MB
    }},
    "regression": {{
        "min_r2": 0.6,
        "max_prediction_time": 1.0,  # seconds
        "max_memory_usage": 1024,    # MB
    }}
}}

# Feature configuration
FEATURE_CONFIG = {{
    "expected_features": {model_info.get("data_info", {}).get("features", 10)},
    "feature_types": {{
        # Define expected feature types here
        # "feature_name": "numeric" | "categorical" | "datetime"
    }},
    "required_features": [
        # List required features here
    ],
    "optional_features": [
        # List optional features here
    ]
}}

# Preprocessing configuration
PREPROCESSING_CONFIG = {{
    "handle_missing_values": True,
    "scale_features": True,
    "encode_categorical": True,
    "remove_outliers": False,
    "outlier_threshold": 3.0
}}

# Prediction configuration
PREDICTION_CONFIG = {{
    "return_probabilities": True if settings.task_type == "classification" else False,
    "confidence_threshold": 0.5,
    "batch_size": 100,
    "max_concurrent_requests": 10
}}

# Monitoring configuration
MONITORING_CONFIG = {{
    "track_predictions": True,
    "track_performance": True,
    "alert_thresholds": {{
        "error_rate": 0.05,  # 5% error rate
        "response_time": 2.0,  # 2 seconds
        "memory_usage": 2048,  # 2GB
    }}
}}

def get_model_config() -> Dict[str, Any]:
    """Get model-specific configuration"""
    return {{
        "model_path": settings.model_path,
        "model_type": settings.model_type,
        "task_type": settings.task_type,
        "performance_thresholds": PERFORMANCE_THRESHOLDS.get(settings.task_type, {{}}),
        "feature_config": FEATURE_CONFIG,
        "preprocessing_config": PREPROCESSING_CONFIG,
        "prediction_config": PREDICTION_CONFIG
    }}

def get_api_config() -> Dict[str, Any]:
    """Get API configuration"""
    return {{
        "host": settings.api_host,
        "port": settings.api_port,
        "workers": settings.api_workers,
        "cors_origins": settings.cors_origins.split(",") if settings.cors_origins != "*" else ["*"],
        "max_batch_size": settings.max_batch_size,
        "request_timeout": settings.request_timeout,
        "max_file_size_mb": settings.max_file_size
    }}

def ensure_directories():
    """Ensure required directories exist"""
    directories = ["models", "logs", "temp", "data", "static"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

# Initialize directories on import
ensure_directories()
'''
    
    return textwrap.dedent(code)

async def _generate_test_suite(model_info: Dict[str, Any], requirements: Dict[str, Any]) -> str:
    """Generate comprehensive test suite"""
    
    model_type = model_info.get("model_info", {}).get("type", "unknown")
    task_type = model_info.get("model_info", {}).get("task_type", "classification")
    
    code = f'''"""
Comprehensive test suite for ML model
Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import pytest
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from model_inference import MLModelInference
    from api_server import app
    from fastapi.testclient import TestClient
except ImportError as e:
    pytest.skip(f"Could not import required modules: {{e}}", allow_module_level=True)

class TestMLModelInference:
    """Test cases for ML model inference"""
    
    @pytest.fixture
    def sample_model_path(self):
        """Create a temporary mock model file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            # Create a mock model for testing
            mock_model = Mock()
            mock_model.predict.return_value = np.array([1, 0, 1] if "{task_type}" == "classification" else [1.5, 2.3, 0.8])
            
            if "{task_type}" == "classification":
                mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.9, 0.1], [0.3, 0.7]])
                mock_model.classes_ = np.array([0, 1])
            
            mock_model.feature_importances_ = np.array([0.3, 0.5, 0.2])
            
            import joblib
            joblib.dump(mock_model, f.name)
            
            yield f.name
            
            # Cleanup
            os.unlink(f.name)
    
    @pytest.fixture
    def ml_inference(self, sample_model_path):
        """Create MLModelInference instance"""
        return MLModelInference(sample_model_path)
    
    def test_model_initialization(self, sample_model_path):
        """Test model initialization"""
        inference = MLModelInference(sample_model_path)
        
        assert inference.model is not None
        assert inference.model_info["type"] == "{model_type}"
        assert inference.model_info["task_type"] == "{task_type}"
    
    def test_model_initialization_file_not_found(self):
        """Test model initialization with non-existent file"""
        with pytest.raises(FileNotFoundError):
            MLModelInference("non_existent_model.pkl")
    
    def test_single_prediction(self, ml_inference):
        """Test single prediction"""
        sample_data = {{"feature_0": 1.0, "feature_1": 2.0, "feature_2": 3.0}}
        
        result = ml_inference.predict(sample_data)
        
        assert "predictions" in result
        assert "model_type" in result
        assert "prediction_time" in result
        assert len(result["predictions"]) == 1
        
        if "{task_type}" == "classification":
            assert "probabilities" in result or "confidence" in result
    
    def test_batch_prediction(self, ml_inference):
        """Test batch prediction"""
        batch_data = [
            {{"feature_0": 1.0, "feature_1": 2.0, "feature_2": 3.0}},
            {{"feature_0": 2.0, "feature_1": 3.0, "feature_2": 4.0}},
            {{"feature_0": 3.0, "feature_1": 4.0, "feature_2": 5.0}}
        ]
        
        results = ml_inference.batch_predict(batch_data, batch_size=2)
        
        assert len(results) == 3
        assert all("prediction" in r for r in results)
        assert all("batch_index" in r for r in results)
    
    def test_dataframe_input(self, ml_inference):
        """Test DataFrame input"""
        df = pd.DataFrame({{
            "feature_0": [1.0, 2.0],
            "feature_1": [2.0, 3.0],
            "feature_2": [3.0, 4.0]
        }})
        
        result = ml_inference.predict(df)
        
        assert "predictions" in result
        assert len(result["predictions"]) == 2
    
    def test_invalid_input(self, ml_inference):
        """Test invalid input handling"""
        # Empty input
        with pytest.raises(Exception):
            ml_inference.predict({{}}if "{{}}")
        
        # Wrong data type
        result = ml_inference.predict("invalid_input")
        assert "error" in result
    
    def test_get_model_info(self, ml_inference):
        """Test model information retrieval"""
        info = ml_inference.get_model_info()
        
        assert "type" in info
        assert "task_type" in info
        assert "domain" in info
        assert info["model_loaded"] is True
    
    def test_get_feature_importance(self, ml_inference):
        """Test feature importance retrieval"""
        importance = ml_inference.get_feature_importance()
        
        assert importance is not None
        assert isinstance(importance, dict)
        assert len(importance) > 0

class TestAPIEndpoints:
    """Test cases for API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_model(self):
        """Mock model for API testing"""
        mock = Mock()
        mock.predict.return_value = {{
            "predictions": [1] if "{task_type}" == "classification" else [1.5],
            "model_type": "{model_type}",
            "task_type": "{task_type}",
            "prediction_time": 0.001
        }}
        mock.get_model_info.return_value = {{
            "type": "{model_type}",
            "task_type": "{task_type}",
            "domain": "{requirements.get('domain', 'general')}",
            "model_loaded": True
        }}
        mock.get_feature_importance.return_value = {{"feature_0": 0.5, "feature_1": 0.3, "feature_2": 0.2}}
        return mock
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "model_type" in data
        assert "version" in data
    
    @patch('api_server.model_instance')
    def test_health_endpoint(self, mock_model_instance, client):
        """Test health endpoint"""
        mock_model_instance.return_value = Mock()
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime" in data
    
    @patch('api_server.model_instance')
    def test_model_info_endpoint(self, mock_model_instance, client, mock_model):
        """Test model info endpoint"""
        mock_model_instance.return_value = mock_model
        
        response = client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "task_type" in data
        assert "domain" in data
    
    @patch('api_server.model_instance')
    def test_predict_endpoint(self, mock_model_instance, client, mock_model):
        """Test prediction endpoint"""
        mock_model_instance.return_value = mock_model
        
        request_data = {{
            "data": {{"feature_0": 1.0, "feature_1": 2.0, "feature_2": 3.0}},
            "return_probabilities": True
        }}
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "model_info" in data
        assert "prediction_time" in data
    
    @patch('api_server.model_instance')
    def test_batch_predict_endpoint(self, mock_model_instance, client, mock_model):
        """Test batch prediction endpoint"""
        mock_model_instance.return_value = mock_model
        mock_model.batch_predict.return_value = [
            {{"prediction": 1, "batch_index": 0}},
            {{"prediction": 0, "batch_index": 1}}
        ]
        
        request_data = {{
            "data": [
                {{"feature_0": 1.0, "feature_1": 2.0}},
                {{"feature_0": 2.0, "feature_1": 3.0}}
            ],
            "batch_size": 100
        }}
        
        response = client.post("/predict/batch", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total_predictions" in data
        assert "processing_time" in data
    
    @patch('api_server.model_instance')
    def test_feature_importance_endpoint(self, mock_model_instance, client, mock_model):
        """Test feature importance endpoint"""
        mock_model_instance.return_value = mock_model
        
        response = client.get("/model/feature-importance")
        
        assert response.status_code == 200
        data = response.json()
        assert "feature_importance" in data
        assert "top_features" in data
    
    def test_predict_no_model(self, client):
        """Test prediction when model is not loaded"""
        request_data = {{
            "data": {{"feature_0": 1.0}},
            "return_probabilities": False
        }}
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 503
    
    def test_invalid_request_format(self, client):
        """Test invalid request format"""
        response = client.post("/predict", json={{"invalid": "data"}})
        
        assert response.status_code == 422  # Validation error

class TestDataProcessing:
    """Test cases for data processing utilities"""
    
    def test_validate_input_dict(self):
        """Test input validation with dictionary"""
        from model_inference import MLModelInference
        
        # Mock model for testing validation only
        with tempfile.NamedTemporaryFile(suffix='.pkl') as f:
            mock_model = Mock()
            import joblib
            joblib.dump(mock_model, f.name)
            
            inference = MLModelInference(f.name)
            
            # Test dictionary input
            result_df = inference._validate_input({{"feature_0": 1.0, "feature_1": 2.0}})
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == 1
    
    def test_validate_input_list(self):
        """Test input validation with list"""
        from model_inference import MLModelInference
        
        with tempfile.NamedTemporaryFile(suffix='.pkl') as f:
            mock_model = Mock()
            import joblib
            joblib.dump(mock_model, f.name)
            
            inference = MLModelInference(f.name)
            
            # Test list of dictionaries
            data = [{{"feature_0": 1.0}}, {{"feature_0": 2.0}}]
            result_df = inference._validate_input(data)
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == 2

# Performance tests
class TestPerformance:
    """Performance test cases"""
    
    @pytest.fixture
    def large_dataset(self):
        """Create large dataset for performance testing"""
        np.random.seed(42)
        return pd.DataFrame({{
            f"feature_{{i}}": np.random.randn(1000) 
            for i in range(10)
        }})
    
    @patch('api_server.model_instance')
    def test_batch_prediction_performance(self, mock_model_instance, large_dataset):
        """Test batch prediction performance"""
        mock_model = Mock()
        mock_model.batch_predict.return_value = [
            {{"prediction": i % 2, "batch_index": i}} 
            for i in range(len(large_dataset))
        ]
        mock_model_instance.return_value = mock_model
        
        import time
        start_time = time.time()
        
        data_list = large_dataset.to_dict('records')
        results = mock_model.batch_predict(data_list, batch_size=100)
        
        processing_time = time.time() - start_time
        
        assert len(results) == len(large_dataset)
        assert processing_time < 10.0  # Should complete within 10 seconds

# Integration tests
class TestIntegration:
    """Integration test cases"""
    
    @pytest.mark.integration
    def test_full_prediction_pipeline(self):
        """Test complete prediction pipeline"""
        # This test requires actual model files and would be run in a full integration environment
        pytest.skip("Integration test requires actual model files")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
'''
    
    return textwrap.dedent(code)
