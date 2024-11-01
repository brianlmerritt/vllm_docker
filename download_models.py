from huggingface_hub import login, snapshot_download
from dotenv import load_dotenv
import os

load_dotenv()

def download_model(model_id, local_dir):
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        raise ValueError("HUGGINGFACE_TOKEN environment variable is not set")
    
    print(f"Downloading {model_id} to {local_dir}...")
    login(token=token)
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        token=token
    )
    print(f"Finished downloading {model_id}")

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download quantized model (AWQ 4-bit)
    download_model("TheBloke/Mistral-7B-Instruct-v0.1-AWQ", "models/mistral")
    
    # Download embeddings model
    download_model("BAAI/bge-large-en-v1.5", "models/embeddings")
    
    # Download reranker model
    download_model("BAAI/bge-reranker-large", "models/reranker")