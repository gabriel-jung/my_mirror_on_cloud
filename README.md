# My Mirror on Cloud

Quick Description...

## Features
- ....

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
