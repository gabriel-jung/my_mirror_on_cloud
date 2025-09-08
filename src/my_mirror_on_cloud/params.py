import os

from dotenv import load_dotenv

load_dotenv()

MLFLOW_HOST = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "default")
MLFLOW_PORT = os.environ.get("MLFLOW_PORT", 5000)

MLFLOW_URI = f"{MLFLOW_HOST}:{MLFLOW_PORT}"


WEAVIATE_URL=os.environ.get("WEAVIATE_URL")
WEAVIATE_KEY=os.environ.get("WEAVIATE_KEY")

# Hugging Face and Mistral AI API keys for LLM access
MISTRAL_API_KEY=os.environ.get("MISTRAL_API_KEY" )
HF_TOKEN=os.environ.get("HF_TOKEN")

if __name__=="__main__":
    print(f"MLFLOW_TRACKING_URI = {MLFLOW_URI}")
    print(f"MLFLOW_EXPERIMENT_NAME = {MLFLOW_EXPERIMENT_NAME}")
    print(f"MLFLOW_PORT = {MLFLOW_PORT}")
    print(f"MLFLOW_URI = {MLFLOW_URI}")
    print(f"WEAVIATE_URL = {WEAVIATE_URL}")
    print(f"WEAVIATE_KEY = {WEAVIATE_KEY}")
    print(f"MISTRAL_API_KEY = {MISTRAL_API_KEY}")
    print(f"HF_TOKEN = {HF_TOKEN}")
