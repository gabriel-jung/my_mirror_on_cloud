# My Mirror on Cloud

My Mirror on Cloud

My Mirror on Cloud is a Python package with a Streamlit interface that together form a fashion recommendation engine.

A catalogue of images—here using the H&M dataset from Kaggle—can be analyzed using powerful encoder models or multimodal Large Language Models (LLMs) accessible via Ollama. The main models used are FashionCLIP and Qwen2.5-VL. The encoded representations are stored in a vector database powered by Weaviate, enabling fast and accurate nearest vector searches.

User requests are processed through LLMs for intelligent formatting and interpretation, distinguishing whether a single clothing item is desired or a coordinated outfit for a particular occasion.

## Features
- Embedding of clothing images and descriptions
- Storage of tags and vectors in a SQL-based system
- Efficient nearest vector search via Weaviate

## Requirements
- Python 3.12

## Installation

### 1. Clone the Repository
```
git clone https://github.com/gabriel-jung/My-Mirror-on-Cloud
cd My-Mirror-on-Cloud
```
### 2. Set Up Python Environment
```
pyenv install 3.12.11
pyenv local 3.12.11
```
### Package installation

#### Option A: Using Poetry (Recommended for Development, need version > 2.0)
```
poetry env use 3.12.11
poetry install
eval $(poetry env activate)
```
#### Option B: Using pip
```
pip install -r requirements-dev.txt
pip install -e .
```

#### Test the installation
```
python -c "import my_mirror_on_cloud; print('Installation successful!')"
```
