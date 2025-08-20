from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from tools.data_tools import analyze_dataset, search_datasets_online
from typing import Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
import json

async def acquire_dataset(requirements: Dict[str, Any]) -> Dict[str, Any]:
    """Acquire dataset based on requirements with performance optimization"""
    try:
        if isinstance(requirements, str):
            requirements = json.loads(requirements)
            
        # Check if dataset path is provided
        dataset_path = requirements.get("dataset_path")
        if dataset_path and Path(dataset_path).exists():
            return await load_user_dataset(dataset_path)
        
        # Search for suitable datasets online
        domain = requirements.get("domain", "general")
        task_type = requirements.get("task_type", "classification")
        
        search_results = await search_datasets_online(domain, task_type)
        
        return {
            "status": "success",
            "source": "online_search",
            "search_results": search_results,
            "recommended_action": "use_synthetic_data",
            "message": "No local dataset provided, recommend generating synthetic data"
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Data acquisition failed: {str(e)}"}

async def load_user_dataset(file_path: str) -> Dict[str, Any]:
    """Load and validate user dataset with optimized processing"""
    try:
        file_path = Path(file_path)
        
        # Optimized file loading based on extension
        if file_path.suffix.lower() == '.csv':
            # Use optimized CSV reading
            df = pd.read_csv(file_path, low_memory=False)
        elif file_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_path.suffix.lower() == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Basic validation
        if df.empty:
            raise ValueError("Dataset is empty")
        
        if df.shape[0] < 10:
            raise ValueError("Dataset too small (< 10 rows)")
            
        return {
            "status": "success",
            "source": "user_provided",
            "path": str(file_path),
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "sample_data": df.head(3).to_dict()  # Reduced sample for performance
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Failed to load dataset: {str(e)}"}

async def profile_dataset(dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    """Profile dataset with comprehensive analysis optimized for speed"""
    try:
        if dataset_info.get("status") != "success":
            return {"status": "error", "message": "Invalid dataset info"}
            
        dataset_path = dataset_info.get("path")
        if not dataset_path:
            return {"status": "error", "message": "No dataset path provided"}
            
        # Use analyze_dataset tool for comprehensive profiling
        analysis = await analyze_dataset(dataset_path)
        
        return analysis
        
    except Exception as e:
        return {"status": "error", "message": f"Dataset profiling failed: {str(e)}"}

# Data Acquisition Agent
data_acquisition_agent = LlmAgent(
    name="data_acquisition",
    model="gemini-2.0-flash",
    instruction="""You are a data acquisition specialist. Your job is to:

1. Load user-provided datasets efficiently
2. Search for suitable datasets online when needed
3. Validate dataset quality and format
4. Provide recommendations for data acquisition

Use the acquire_dataset tool to handle data acquisition based on user requirements.
Always prioritize user-provided data over online searches.""",
    
    tools=[acquire_dataset, load_user_dataset, google_search]
)

# Data Profiling Agent  
data_profiling_agent = LlmAgent(
    name="data_profiling",
    model="gemini-2.0-flash",
    instruction="""You are a data profiling expert. Analyze datasets to understand:

1. Data quality metrics (missing values, duplicates, outliers)
2. Statistical distributions and characteristics
3. Data types and formats
4. Potential issues and recommendations

Use the profile_dataset tool to perform comprehensive data analysis.
Focus on actionable insights for preprocessing and modeling.""",
    
    tools=[profile_dataset, analyze_dataset]
)
