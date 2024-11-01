# vLLM Docker for RAG Applications

This repository contains a Docker-based setup for running vLLM (very Large Language Model) inference with multiple models for RAG (Retrieval-Augmented Generation) applications. The setup includes support for embeddings, reranking, and function calling through OpenAI-compatible APIs.

## System Requirements

- NVIDIA GPU with CUDA support
- Docker and Docker Compose
- Minimum 24GB GPU RAM recommended (8GB per model)
- Ubuntu 20.04 or later (recommended)

## Quick Start

```bash
# Clone and setup the project
git clone <repository-url>
cd vllm-docker

# Download models
python3 download_models.py

# Build and start the services
docker-compose up -d
```

## Architecture

The setup consists of three main services:

1. **Main LLM (Port 8400)**
   - Model: Mixtral-8x7B-Instruct-v0.1
   - Handles: Chat completions and function calling
   - Endpoint: `http://localhost:8400/v1/chat/completions`

2. **Embeddings Service (Port 8401)**
   - Model: BGE-large-en-v1.5
   - Handles: Text embeddings for vector search
   - Endpoint: `http://localhost:8401/v1/embeddings`

3. **Reranker Service (Port 8402)**
   - Model: BGE-reranker-large
   - Handles: Result reranking
   - Endpoint: `http://localhost:8402/v1/rerank`

## Directory Structure

```
vllm-docker/
├── docker-compose.yaml      # Container orchestration
├── Dockerfile.vllm         # Container build instructions
├── models/                 # Downloaded model files
├── configs/               # Configuration files
│   ├── model_config.json
│   └── function_schemas.json
└── download_models.py     # Model download script
```

## API Usage Examples

### Chat Completion
```python
import requests

response = requests.post(
    "http://localhost:8400/v1/chat/completions",
    json={
        "model": "mixtral",
        "messages": [
            {"role": "user", "content": "What is veterinary medicine?"}
        ]
    }
)
```

### Embeddings
```python
response = requests.post(
    "http://localhost:8401/v1/embeddings",
    json={
        "input": "veterinary anatomy",
        "model": "bge-large-en-v1.5"
    }
)
```

### Reranking
```python
response = requests.post(
    "http://localhost:8402/v1/rerank",
    json={
        "query": "canine diseases",
        "documents": ["text1", "text2", "text3"],
        "model": "bge-reranker-large"
    }
)
```

## Configuration

### Adjusting Model Parameters

Edit `configs/model_config.json` to modify:
- Batch sizes
- Context lengths
- Quantization settings
- Model paths

### Function Calling

Function schemas are defined in `configs/function_schemas.json`. Add or modify functions as needed for your application.

## Resource Management

Each service is configured to use a specific GPU. Modify `CUDA_VISIBLE_DEVICES` in `docker-compose.yaml` to change GPU assignments:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0  # Change to desired GPU index
```

## Development Guidelines

1. **Model Updates**
   - Add new models to `download_models.py`
   - Update corresponding configurations in `model_config.json`

2. **API Extensions**
   - Follow OpenAI API compatibility guidelines
   - Maintain consistent error handling

3. **Performance Optimization**
   - Adjust batch sizes based on your GPU memory
   - Consider quantization for larger models

## Troubleshooting

Common issues and solutions:

1. **Out of Memory**
   - Reduce batch sizes in model configs
   - Enable quantization
   - Use fewer concurrent requests

2. **Model Download Failures**
   - Check network connectivity
   - Ensure sufficient disk space
   - Verify Hugging Face login if needed

3. **API Errors**
   - Check port mappings
   - Verify CUDA availability
   - Monitor container logs

## Monitoring

View container logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f vllm-main
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

[Insert your license information here]

## Citation

If you use this setup in your research, please cite:

```bibtex
[Insert citation information if applicable]
```