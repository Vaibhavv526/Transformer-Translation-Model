# Transformer Translation Model

A Transformer-based Neural Machine Translation (NMT) system built from scratch using PyTorch. This project implements the complete encoder-decoder Transformer architecture for English-to-Hindi translation, including training, inference, tokenizer generation, and API-based deployment support.

---

# Project Overview

This project focuses on understanding and implementing the core concepts behind modern sequence-to-sequence translation systems.

The model was trained using a custom Transformer implementation inspired by the architecture introduced in the paper:

> Attention Is All You Need

The system performs:

* English → Hindi translation
* Autoregressive decoding
* Tokenization using Hugging Face tokenizers
* Custom Transformer encoder-decoder architecture
* FastAPI inference serving
* Model checkpointing and loading

---

# Features

* Transformer architecture built from scratch
* Multi-head self-attention
* Positional encoding
* Encoder-decoder attention
* Greedy decoding inference
* Tokenizer integration
* FastAPI deployment backend
* Git LFS model hosting
* GPU/CPU training support

---

# Tech Stack

## Core Frameworks

* PyTorch
* FastAPI
* Hugging Face Tokenizers
* NumPy

## Tools

* Git
* Git LFS
* Uvicorn
* TensorBoard

---

# Project Structure

```bash
Transformer_From_Scratch-main/
│
├── app.py
├── server.py
├── train.py
├── translate.py
├── model.py
├── config.py
├── dataSet.py
├── requirements.txt
│
├── tokenizer_en.json
├── tokenizer_hi.json
│
├── docs/
│   └── 1706.03762v7.pdf
│
├── weights/
│   └── best_model.pt
│
└── README.md
```

---

# Model Architecture

The model follows the standard Transformer encoder-decoder design.

## Components

### Encoder

* Multi-head self-attention
* Feed-forward network
* Residual connections
* Layer normalization

### Decoder

* Masked self-attention
* Encoder-decoder attention
* Feed-forward network
* Causal masking

### Additional Components

* Positional encoding
* Token embeddings
* Projection layer

---

# Training Details

## Training Configuration

* Framework: PyTorch
* Architecture: Transformer
* Sequence-to-sequence learning
* Loss Function: Cross Entropy Loss
* Optimizer: Adam
* Autoregressive decoding

## Training Progress

The model was trained up to:

```text
Epoch 64
```

The best-performing checkpoint was selected for inference and deployment usage.

---

# Sample Predictions

## Example 1

```text
SOURCE: install module
PREDICTED: मॉड्यूल स्थापित करें
```

## Example 2

```text
SOURCE: machine learning
PREDICTED: शिक्षा मशीन
```

## Example 3

```text
SOURCE: Variable list
PREDICTED: परिवर्तनीय सूची
```

---

# Running the Project

## 1. Clone Repository

```bash
git clone https://github.com/Vaibhavv526/Transformer-Translation-Model.git
cd Transformer-Translation-Model
```

---

## 2. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 3. Run Translation Script

```bash
python translate.py "hello world"
```

---

## 4. Run FastAPI Server

```bash
uvicorn server:app --reload
```

---

## 5. Open API Docs

```text
http://127.0.0.1:8000/docs
```

---

# API Endpoint

## Translate Text

```http
GET /translate?text=hello
```

### Example Response

```json
{
  "input": "hello",
  "translation": "नमस्ते"
}
```

---

# Model Hosting

The trained model weights are stored using Git LFS due to the large checkpoint size.

## Git LFS Setup

```bash
git lfs install
git lfs track "*.pt"
```

---

# Why the Model Is Not Publicly Deployed

The project includes a fully working FastAPI inference backend and deployment pipeline.

However, the model is currently not hosted on a public cloud deployment platform for the following technical reasons:

## 1. Memory Constraints on Free Hosting Platforms

The trained Transformer checkpoint is relatively large (~266 MB).

When loaded into memory, PyTorch requires significantly more RAM than the raw file size due to:

* model deserialization
* tensor allocation
* runtime overhead
* tokenizer loading
* inference graph initialization

Most free cloud hosting platforms provide only 512 MB RAM, which is insufficient for stable deployment of this model.

---

## 2. Avoiding Model Degradation

The model could be reduced in size using:

* aggressive quantization
* smaller architecture dimensions
* layer reduction
* checkpoint pruning

However, these optimizations would negatively affect translation quality and overall model performance.

The decision was made to preserve the original trained model rather than deploying a heavily compressed variant.

---

## 3. Cost Constraints

Deploying the model reliably would require a higher-memory cloud instance.

Since the primary goal of this project was:

* learning Transformer architecture
* implementing training/inference pipelines
* understanding deployment workflows
* building an end-to-end NLP system

public production hosting was intentionally deferred.

---

# Future Improvements

Potential future upgrades include:

* Beam Search decoding
* BLEU score evaluation
* Mixed precision training
* Better dataset cleaning
* SentencePiece tokenization
* Quantized inference
* Attention visualization
* Streamlit/React frontend
* Production deployment

---

# Learning Outcomes

This project helped build practical understanding of:

* Transformer internals
* Attention mechanisms
* Encoder-decoder systems
* Sequence modeling
* NLP pipelines
* Model serving
* FastAPI deployment
* Git LFS model management

---

# Repository

GitHub Repository:

[https://github.com/Vaibhavv526/Transformer-Translation-Model](https://github.com/Vaibhavv526/Transformer-Translation-Model)

---

# Author

Vaibhav

B.Tech Student | Deep Learning & Machine Learning Enthusiast
