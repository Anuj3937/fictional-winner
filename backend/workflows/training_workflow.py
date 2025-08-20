from google.adk.agents import SequentialAgent, LoopAgent
from agents.training_agents import parallel_model_training
from agents.optimization_agents import hyperparameter_optimizer, ensemble_creator
from tools.quality_tools import assess_model_ensemble_quality
from typing import Dict, Any
import asyncio

class ModelQualityGate:
    """Quality gate for model validation with reiteration logic"""
    
    async def __call__(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess model quality and determine next action"""
        
        try:
            # Extract model results from parallel training
            model_results = []
            
            if isinstance(training_results, dict):
                for agent_name, result in training_results.items():
                    if isinstance(result, dict) and result.get("status") == "success":
                        model_results.append(result)
            
            # Assess ensemble quality
            quality_assessment = await assess_model_ensemble_quality(model_results)
            
            should_reiterate = quality_assessment.get("quality_assessment", {}).get("should_reiterate", False)
            next_action = quality_assessment.get("next_action", "proceed")
            
            if should_reiterate:
                return {
                    "status": "quality_insufficient",
                    "action": "reiterate", 
                    "next_step": next_action,
                    "assessment": quality_assessment,
                    "message": "Model performance below threshold, optimizing hyperparameters"
                }
            else:
                return {
                    "status": "quality_approved",
                    "action": "proceed",
                    "assessment": quality_assessment, 
                    "model_results": model_results,
                    "message": "Model quality meets requirements, proceeding to ensemble creation"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Model quality assessment failed: {str(e)}"
            }

# Create quality gate instance  
model_quality_gate = ModelQualityGate()

# Training Workflow - Sequential execution of training steps
training_workflow = SequentialAgent(
    name="model_training_workflow",
    sub_agents=[
        parallel_model_training,     # Train multiple models in parallel
        hyperparameter_optimizer,    # Optimize the best model
        ensemble_creator            # Create ensemble models
    ]
)

# Model Quality Loop - Reiterate until quality is acceptable
model_quality_loop = LoopAgent(
    name="model_quality_assurance_loop", 
    sub_agents=[training_workflow],
    max_iterations=3  # Limit iterations for performance
)
