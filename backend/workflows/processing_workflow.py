from google.adk.agents import SequentialAgent
from agents.processing_agents import preprocessing_agent, feature_engineering_agent

# Processing Workflow - Sequential data processing
processing_workflow = SequentialAgent(
    name="data_processing_workflow",
    sub_agents=[
        preprocessing_agent,        # Clean and prepare data
        feature_engineering_agent  # Create valuable features
    ]
)
