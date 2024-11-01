import os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel

def download_model(model_id, local_dir):
    if not os.path.exists(local_dir):
        print(f"Downloading {model_id} to {local_dir}")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            ignore_patterns=["*.msgpack", "*.h5", "*.safetensors"]
        )
        # Initialize tokenizer and model to ensure all files are downloaded
        AutoTokenizer.from_pretrained(local_dir)
        try:
            AutoModel.from_pretrained(local_dir)
        except Exception as e:
            print(f"Note: Model loading failed but files should be downloaded: {e}")
    else:
        print(f"Model directory {local_dir} already exists, skipping download")

def main():
    models = {
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "/models/mixtral",
        "BAAI/bge-large-en-v1.5": "/models/bge",
        "BAAI/bge-reranker-large": "/models/bge-reranker"
    }
    
    for model_id, local_dir in models.items():
        download_model(model_id, local_dir)

if __name__ == "__main__":
    main()
