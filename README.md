# My Mirror on Cloud

An AI-powered fashion recommendation engine to help users discover similar clothing items and complete outfit suggestions.

## 📺 Demo

<p align="center">
  <img src="demo.gif" width="25%" alt="Visual Search" />
  <img src="demo2.gif" width="25%" alt="Virtual Try-On" />
</p>

*Left: Search for full outfit | Right: Virtual try-on with CatVTON*

*Note: This demonstration showcases the recommendation engine built during the Artefact Data Science bootcamp. The application is no longer hosted due to infrastructure costs.*
My Mirror on Cloud is a Python package with a Streamlit interface that together form a fashion recommendation engine.

## 🎯 Project Overview

My Mirror on Cloud is a Python package with a Streamlit interface that forms an intelligent fashion recommendation system. Using the [H&M dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data) from Kaggle, the system analyzes fashion items through FashionCLIP, a specialized encoder model for fashion image understanding.

User queries are processed through Large Language Models to interpret and format requests intelligently, distinguishing whether a single clothing item is desired or a coordinated outfit for a particular occasion. The encoded image representations are stored in a Weaviate vector database for lightning-fast similarity searches.

## ✨ Key Features

- **Visual Search**: Upload a fashion item photo to find visually similar products
- **Text-to-Fashion**: Describe your desired style in natural language
- **Smart Query Understanding**: LLM-powered interpretation distinguishes between single item requests and complete outfit recommendations
- **FashionCLIP Embeddings**: Specialized computer vision model optimized for fashion similarity
- **Fast Retrieval**: Efficient similarity search powered by Weaviate vector database
- **Interactive UI**: User-friendly Streamlit interface requiring no technical knowledge
- **Flexible Architecture**: Modular design tested with multiple models (FashionCLIP, CLIP, Qwen2.5-VL)

## 🏗️ How It Works

1. **Dataset Preparation**: Fashion items from the H&M catalogue are preprocessed and organized
2. **Visual Embedding**: Images are encoded using FashionCLIP to capture visual and fashion-specific features
3. **Vector Storage**: Embeddings are stored in Weaviate vector database alongside metadata in SQL for efficient retrieval
4. **Query Understanding**: User requests are processed by LLMs to interpret intent and format queries appropriately
5. **Intent Detection**: The system distinguishes between single item searches and complete outfit recommendations
6. **Similarity Search**: Nearest neighbor search retrieves the most visually similar items from the vector database
7. **Virtual Try-On** (Optional): CatVTON generates realistic visualizations of selected items on user photos
8. **Results Display**: Recommendations are presented through an intuitive Streamlit interface


## 🛠️ Tech Stack

**Core Technologies**: Python 3.12 • Streamlit 

**ML & AI**: [FashionCLIP](https://github.com/patrickjohncyh/fashion-clip) • [CatVTON](https://github.com/Zheng-Chong/CatVTON) • LLMs (via Ollama and Mistral API)

**Data & Storage**: Weaviate (vector DB) • SQL • H&M Fashion Dataset (Kaggle)

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

## 📝 Project Context

This project was developed as the **final capstone project** for the **Artefact Data Science Bootcamp** as a **collaborative three-person team effort**.
