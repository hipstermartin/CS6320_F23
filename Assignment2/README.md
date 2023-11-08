
# CS6320 Assignment 2: Sentiment Analysis with Neural Networks

## Overview

In this assignment, our focus was on building and evaluating sentiment analysis models using two different neural network architectures: Feed-Forward Neural Network (FFNN) and Recurrent Neural Network (RNN). This repository houses the code, datasets, and documentation for these implementations.

## Repository Structure

- **Data_Embedding/**: This directory contains the datasets and pre-trained word embeddings required for the models.
  - `test.json`
  - `train.json`
  - `validate.json`
  - `word_embedding.pkl` - The pre-trained word embeddings.
- `ffnn.py` - The FFNN model implementation.
- `rnn.py` - The RNN model implementation.
- `learningCurve.py` - Script for plotting learning curves.
- `requirements.txt` - Required libraries to run the models.
- `README.md` - Documentation and setup guide.

## Implementation Details

### Neural Network Models

- **FFNN Model (`ffnn.py`)**: Implements a simple feed-forward neural network that uses word embeddings and dense layers for sentiment classification.
- **RNN Model (`rnn.py`)**: Utilizes recurrent neural network architecture to capture sequential information in the text for improved sentiment classification.

## Getting Started

To run the models:

1. Clone the repository and navigate to its root directory.
2. Install the required libraries using:
   ```bash
   pip install -r requirements.txt
   ```


## One example on running the code:`

**FFNN**

```bash 
python ffnn.py --hidden_dim 64 --epochs 5 --train_data training.json --val_data validation.json --test_data test.json
python ffnn.py --hidden_dim 64 --epochs 10 --train_data training.json --val_data validation.json --test_data test.json
python ffnn.py --hidden_dim 128 --epochs 5 --train_data training.json --val_data validation.json --test_data test.json
python ffnn.py --hidden_dim 128 --epochs 10 --train_data training.json --val_data validation.json --test_data test.json
```
**RNN**

```bash
python rnn.py --hidden_dim 64 --epochs 5 --train_data ./Data_Embedding/training.json --val_data ./Data_Embedding/validation.json --test_data ./Data_Embedding/test.json
python rnn.py --hidden_dim 64 --epochs 10 --train_data ./Data_Embedding/training.json --val_data ./Data_Embedding/validation.json --test_data ./Data_Embedding/test.json
python rnn.py --hidden_dim 128 --epochs 5 --train_data ./Data_Embedding/training.json --val_data ./Data_Embedding/validation.json --test_data ./Data_Embedding/test.json
python rnn.py --hidden_dim 128 --epochs 10 --train_data ./Data_Embedding/training.json --val_data ./Data_Embedding/validation.json --test_data ./Data_Embedding/test.json
```