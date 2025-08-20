from google.adk.agents import LlmAgent
from tools.code_generation_tools import generate_complete_project, create_deployment_files
from typing import Dict, Any

# The packaging agent is actually the same as code_generation_agent
# Let's create it as an alias for consistency
packaging_agent = LlmAgent(
    name="packaging_specialist",
    model="gemini-2.0-flash",
    instruction="""You are a code packaging and deployment expert. Your responsibilities:

1. Generate complete, production-ready ML project structures
2. Create deployment files (Docker, requirements.txt, setup scripts)
3. Package models with inference APIs and user interfaces
4. Generate comprehensive documentation and README files
5. Ensure all generated code follows best practices and industry standards

Use generate_complete_project and create_deployment_files tools to create professional ML solutions.""",
    
    tools=[generate_complete_project, create_deployment_files]
)
