from google.adk.agents import LlmAgent
from google.adk.tools import google_search
import json
import re
from typing import Dict, Any

async def parse_ml_requirements(user_prompt: str) -> Dict[str, Any]:
    """Parse user requirements from natural language with advanced NLP"""
    
    try:
        # Clean and normalize the prompt
        prompt = user_prompt.lower().strip()
        
        # Determine task type using advanced pattern matching
        task_type = "classification"
        classification_patterns = [
            r"classif", r"predict (class|category|label)", r"categori", 
            r"binary", r"multi-class", r"sentiment", r"fraud detection",
            r"churn", r"spam", r"diagnosis", r"risk assessment"
        ]
        regression_patterns = [
            r"regress", r"predict (value|price|amount|score|rating)", 
            r"forecast", r"estimate", r"continuous", r"numerical",
            r"house price", r"stock price", r"sales forecast"
        ]
        
        if any(re.search(pattern, prompt) for pattern in regression_patterns):
            task_type = "regression"
        elif any(re.search(pattern, prompt) for pattern in classification_patterns):
            task_type = "classification"
        
        # Determine domain with extensive domain mapping
        domain_patterns = {
            "healthcare": [
                r"health", r"medical", r"patient", r"disease", r"diagnosis",
                r"symptom", r"treatment", r"clinical", r"pharmaceutical",
                r"hospital", r"doctor", r"medicine", r"drug"
            ],
            "finance": [
                r"financ", r"bank", r"loan", r"credit", r"investment", 
                r"stock", r"trading", r"portfolio", r"risk", r"insurance",
                r"fraud", r"payment", r"transaction", r"revenue"
            ],
            "retail": [
                r"retail", r"sales", r"product", r"customer", r"purchase",
                r"marketing", r"recommendation", r"inventory", r"e-commerce",
                r"shopping", r"recommendation", r"conversion"
            ],
            "transportation": [
                r"transport", r"vehicle", r"traffic", r"route", r"shipping",
                r"logistics", r"fleet", r"delivery", r"travel", r"mobility"
            ],
            "technology": [
                r"software", r"network", r"security", r"performance",
                r"system", r"application", r"infrastructure", r"cloud"
            ],
            "manufacturing": [
                r"manufactur", r"production", r"quality control", r"supply chain",
                r"equipment", r"maintenance", r"factory", r"industrial"
            ]
        }
        
        domain = "general"
        for domain_name, patterns in domain_patterns.items():
            if any(re.search(pattern, prompt) for pattern in patterns):
                domain = domain_name
                break
        
        # Extract specific requirements
        requirements = {
            "task_type": task_type,
            "domain": domain,
            "user_prompt": user_prompt,
            "has_dataset": bool(re.search(r"dataset|data|file|csv|upload", prompt)),
            "synthetic_requested": bool(re.search(r"synthetic|generate.*data|create.*dataset", prompt)),
            "specific_algorithms": extract_algorithms(prompt),
            "performance_requirements": extract_performance_requirements(prompt),
            "deployment_requirements": extract_deployment_requirements(prompt),
            "data_size_hints": extract_data_size_hints(prompt),
            "feature_preferences": extract_feature_preferences(prompt)
        }
        
        return {
            "status": "success",
            "requirements": requirements,
            "confidence_score": calculate_confidence_score(requirements, prompt),
            "suggestions": generate_suggestions(requirements)
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Requirements parsing failed: {str(e)}"
        }

def extract_algorithms(prompt: str) -> list:
    """Extract specific algorithm mentions"""
    algorithms = []
    algorithm_patterns = {
        "random_forest": [r"random forest", r"rf"],
        "xgboost": [r"xgboost", r"xgb", r"gradient boost"],
        "lightgbm": [r"lightgbm", r"lgbm", r"light gbm"],
        "svm": [r"svm", r"support vector"],
        "neural_network": [r"neural", r"deep learning", r"nn"],
        "logistic_regression": [r"logistic", r"linear regression"],
        "decision_tree": [r"decision tree", r"tree"]
    }
    
    for algo, patterns in algorithm_patterns.items():
        if any(re.search(pattern, prompt.lower()) for pattern in patterns):
            algorithms.append(algo)
    
    return algorithms

def extract_performance_requirements(prompt: str) -> Dict[str, Any]:
    """Extract performance requirements"""
    requirements = {}
    
    # Extract accuracy requirements
    accuracy_match = re.search(r"(\d+)%?\s*accuracy", prompt.lower())
    if accuracy_match:
        requirements["min_accuracy"] = float(accuracy_match.group(1)) / 100
    
    # Extract speed requirements
    if re.search(r"fast|quick|real-time|low latency", prompt.lower()):
        requirements["speed_priority"] = "high"
    elif re.search(r"accurate|precision|best performance", prompt.lower()):
        requirements["accuracy_priority"] = "high"
    
    return requirements

def extract_deployment_requirements(prompt: str) -> Dict[str, Any]:
    """Extract deployment preferences"""
    deployment = {}
    
    if re.search(r"api|rest|web service", prompt.lower()):
        deployment["api_required"] = True
    if re.search(r"streamlit|dashboard|ui|interface", prompt.lower()):
        deployment["ui_required"] = True
    if re.search(r"docker|container", prompt.lower()):
        deployment["containerization"] = True
    if re.search(r"cloud|aws|azure|gcp", prompt.lower()):
        deployment["cloud_deployment"] = True
    
    return deployment

def extract_data_size_hints(prompt: str) -> Dict[str, Any]:
    """Extract data size information"""
    size_hints = {}
    
    # Extract sample count hints
    samples_match = re.search(r"(\d+[kmKM]?)\s*(samples|rows|records|examples)", prompt.lower())
    if samples_match:
        count_str = samples_match.group(1).lower()
        if 'k' in count_str:
            size_hints["estimated_samples"] = int(count_str.replace('k', '')) * 1000
        elif 'm' in count_str:
            size_hints["estimated_samples"] = int(count_str.replace('m', '')) * 1000000
        else:
            size_hints["estimated_samples"] = int(count_str)
    
    # Extract feature count hints
    features_match = re.search(r"(\d+)\s*(features|columns|variables)", prompt.lower())
    if features_match:
        size_hints["estimated_features"] = int(features_match.group(1))
    
    return size_hints

def extract_feature_preferences(prompt: str) -> list:
    """Extract feature engineering preferences"""
    preferences = []
    
    if re.search(r"polynomial|interaction", prompt.lower()):
        preferences.append("polynomial_features")
    if re.search(r"scaling|normali[sz]ation", prompt.lower()):
        preferences.append("feature_scaling")
    if re.search(r"selection|important features", prompt.lower()):
        preferences.append("feature_selection")
    if re.search(r"engineering|new features", prompt.lower()):
        preferences.append("feature_engineering")
    
    return preferences

def calculate_confidence_score(requirements: Dict[str, Any], prompt: str) -> float:
    """Calculate confidence score for requirement parsing"""
    score = 0.5  # Base score
    
    # Boost score for clear indicators
    if requirements.get("specific_algorithms"):
        score += 0.2
    if requirements.get("performance_requirements"):
        score += 0.1
    if requirements.get("deployment_requirements"):
        score += 0.1
    if len(prompt) > 50:  # Detailed prompts are more reliable
        score += 0.1
    
    return min(1.0, score)

def generate_suggestions(requirements: Dict[str, Any]) -> list:
    """Generate helpful suggestions based on requirements"""
    suggestions = []
    
    if requirements["task_type"] == "classification":
        suggestions.append("Consider trying ensemble methods for better accuracy")
    if requirements["task_type"] == "regression":
        suggestions.append("Feature scaling might be important for regression tasks")
    
    if not requirements.get("specific_algorithms"):
        suggestions.append("I'll select optimal algorithms based on your data characteristics")
    
    if requirements["domain"] != "general":
        suggestions.append(f"I'll apply domain-specific techniques for {requirements['domain']} problems")
    
    return suggestions

# Create the Requirements Agent
requirements_agent = LlmAgent(
    name="requirements_interpreter",
    model="gemini-2.0-flash",
    instruction="""You are an expert ML requirements interpreter. Your job is to:

1. Parse natural language ML requests into structured requirements
2. Identify task type (classification/regression), domain, and specific needs
3. Extract algorithm preferences, performance requirements, and deployment needs
4. Provide confidence scores and helpful suggestions
5. Handle ambiguous requests by making reasonable assumptions

Use the parse_ml_requirements tool to analyze user requests thoroughly.
Always provide clear, actionable requirements that downstream agents can use effectively.""",
    
    tools=[parse_ml_requirements, google_search]
)
