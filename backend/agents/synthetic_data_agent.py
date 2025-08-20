from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from tools.synthetic_tools import synthetic_generator
from tools.search_tools import search_engine
from typing import Dict, Any
import json
import asyncio

async def generate_statistical_synthetic_data(requirements: Dict[str, Any]) -> Dict[str, Any]:
    """Generate synthetic data based on web-researched statistics"""
    try:
        # Parse requirements
        if isinstance(requirements, str):
            requirements = json.loads(requirements)
            
        # Generate research-based synthetic dataset
        dataset_path = await synthetic_generator.generate_research_based_dataset(requirements)
        
        # Get the metadata
        import pandas as pd
        df = pd.read_csv(dataset_path)
        
        return {
            "status": "success",
            "source": "research_based_synthetic",
            "path": dataset_path,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "sample_data": df.head().to_dict(),
            "generation_method": "statistical_web_research",
            "research_enhanced": True
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Synthetic data generation failed: {str(e)}"
        }

async def research_domain_statistics(domain: str, task_type: str) -> Dict[str, Any]:
    """Research domain statistics using web search"""
    try:
        stats = await search_engine.search_statistical_data(domain, task_type)
        return {
            "status": "success",
            "domain": domain,
            "task_type": task_type,
            "statistics": stats,
            "research_source": "web_search"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Domain statistics research failed: {str(e)}"
        }

# Synthetic Data Agent with Google Search Integration
synthetic_data_agent = LlmAgent(
    name="synthetic_data_generator",
    model="gemini-2.0-flash",
    instruction="""You are a synthetic data generation expert that creates realistic datasets based on web-researched statistical parameters.

When generating synthetic data:
1. First research domain-specific statistical characteristics using web search
2. Extract typical ranges, distributions, and correlations from research
3. Generate synthetic data that matches real-world statistical properties
4. Ensure the data is suitable for the requested ML task type

Use research_domain_statistics to gather statistical information, then generate_statistical_synthetic_data to create the dataset.

Always provide detailed information about the statistical basis for the generated data.""",
    
    tools=[generate_statistical_synthetic_data, research_domain_statistics, google_search]
)
