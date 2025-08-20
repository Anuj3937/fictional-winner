from google.adk.agents import SequentialAgent, LoopAgent
from agents.data_agents import data_acquisition_agent, data_profiling_agent
from agents.synthetic_data_agent import synthetic_data_agent
from tools.quality_tools import assess_data_quality
from typing import Dict, Any
import asyncio

class DataQualityGate:
    """Quality gate for data validation with reiteration logic"""
    
    async def __call__(self, data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data quality and determine next action"""
        
        try:
            # Assess quality using quality tools
            quality_assessment = await assess_data_quality(data_profile)
            
            should_reiterate = quality_assessment.get("quality_assessment", {}).get("should_reiterate", False)
            
            if should_reiterate:
                return {
                    "status": "quality_insufficient",
                    "action": "reiterate",
                    "assessment": quality_assessment,
                    "message": "Data quality below threshold, searching for better data or generating synthetic data"
                }
            else:
                return {
                    "status": "quality_approved", 
                    "action": "proceed",
                    "assessment": quality_assessment,
                    "message": "Data quality meets requirements, proceeding to preprocessing"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Quality assessment failed: {str(e)}"
            }

# Create quality gate instance
data_quality_gate = DataQualityGate()

# Data Acquisition Workflow - Sequential execution
data_acquisition_workflow = SequentialAgent(
    name="data_acquisition_workflow",
    sub_agents=[
        data_acquisition_agent,      # Try to acquire dataset
        synthetic_data_agent,        # Generate synthetic if needed  
        data_profiling_agent         # Profile the final dataset
    ]
)

# Data Quality Loop - Reiterate until quality is acceptable
data_quality_loop = LoopAgent(
    name="data_quality_assurance_loop",
    sub_agents=[data_acquisition_workflow],
    max_iterations=3  # Limit iterations to prevent infinite loops
)
