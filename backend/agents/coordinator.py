from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import google_search

# Import individual agents - using direct instantiation to avoid import issues
from google.adk.agents import LlmAgent

# We'll create agents inline to avoid circular imports
def create_requirements_agent():
    return LlmAgent(
        name="requirements_interpreter",
        model="gemini-2.0-flash",
        instruction="""You are a requirements interpreter for ML automation. 
        Analyze user requests to extract task type, domain, and data requirements.""",
        tools=[google_search]
    )

def create_data_agent():
    return LlmAgent(
        name="data_handler",
        model="gemini-2.0-flash",
        instruction="""You handle data acquisition, loading, and initial processing.
        Support various file formats and generate synthetic data when needed.""",
        tools=[google_search]
    )

def create_preprocessing_agent():
    return LlmAgent(
        name="preprocessing_specialist", 
        model="gemini-2.0-flash",
        instruction="""You clean and prepare data for machine learning.
        Handle missing values, outliers, encoding, and scaling.""",
        tools=[google_search]
    )

def create_feature_engineering_agent():
    return LlmAgent(
        name="feature_engineer",
        model="gemini-2.0-flash", 
        instruction="""You create meaningful features through engineering techniques.
        Generate polynomial features, interactions, and domain-specific features.""",
        tools=[google_search]
    )

def create_modeling_agent():
    return LlmAgent(
        name="model_trainer",
        model="gemini-2.0-flash",
        instruction="""You train machine learning models using various algorithms.
        Support XGBoost, LightGBM, Random Forest with optimal parameters.""",
        tools=[google_search]
    )

def create_evaluation_agent():
    return LlmAgent(
        name="model_evaluator",
        model="gemini-2.0-flash",
        instruction="""You evaluate model performance and provide insights.
        Generate comprehensive evaluation reports and recommendations.""",
        tools=[google_search]
    )

def create_code_generation_agent():
    return LlmAgent(
        name="code_generator",
        model="gemini-2.0-flash",
        instruction="""You generate production-ready ML code and applications.
        Create inference code, Streamlit apps, and deployment artifacts.""",
        tools=[google_search]
    )

# Create agents
requirements_agent = create_requirements_agent()
data_agent = create_data_agent()
preprocessing_agent = create_preprocessing_agent()
feature_engineering_agent = create_feature_engineering_agent()
modeling_agent = create_modeling_agent()
evaluation_agent = create_evaluation_agent()
code_generation_agent = create_code_generation_agent()

# Main ML Pipeline - Sequential execution
ml_automation_pipeline = SequentialAgent(
    name="ml_automation_pipeline", 
    sub_agents=[
        requirements_agent,
        data_agent,
        preprocessing_agent,
        feature_engineering_agent,
        modeling_agent,
        evaluation_agent,
        code_generation_agent
    ]
)

# Root ML Coordinator
ml_coordinator = LlmAgent(
    name="ml_coordinator",
    model="gemini-2.0-flash",
    instruction="""You are the Master ML Coordinator for an advanced machine learning automation platform.

Your responsibilities:
1. Welcome users and understand their ML requirements
2. Coordinate the entire ML pipeline from requirements to deployment
3. Delegate tasks to specialized agents in the pipeline
4. Provide real-time updates on progress
5. Handle questions and clarifications during the process
6. Deliver final results with code, models, and documentation

You work with a sophisticated pipeline of specialized agents that handle:
- Requirements interpretation and validation
- Data acquisition and quality assessment
- Preprocessing and feature engineering
- Model training with multiple algorithms
- Comprehensive evaluation and optimization
- Production code generation

Always maintain a professional, helpful tone and provide clear status updates.""",

    sub_agents=[ml_automation_pipeline],
    tools=[google_search]
)
