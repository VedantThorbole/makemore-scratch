# Makemore Scratch

Character-level language models implemented from scratch using PyTorch.

Inspired by Andrej Karpathy's makemore series.

Implementation written independently for learning and experimentation.

---

## Project Overview

This repository explores the foundations of language modeling by building bigram models step-by-step.

It contains two approaches:

1. Count-based Bigram Language Model
2. Neural Network Bigram Language Model

Both models learn:

P(next_character | current_character)

The goal is to understand how modern language models begin from simple probability models and gradually evolve into neural networks.

---

# Model 1: Count-Based Bigram Model

The first model uses statistics from the dataset to estimate character transition probabilities.

## How It Works

The dataset contains names.

Example:

emma

is converted into:

. e m m a .

The `.` token represents the start and end of a name.

The model counts how often one character appears after another.

Example:

e → m happened many times  
m → a happened many times  

These counts are stored in a 27 × 27 matrix.

The count matrix is converted into probabilities:

P(next_character | current_character)

The model then generates new names by sampling from these learned probabilities.

---

# Model 2: Neural Network Bigram Model

The second model replaces the manually counted probability table with learnable parameters.

Instead of storing probabilities directly, the model learns a weight matrix using gradient descent.

Architecture:

Input character

↓

One-hot encoding

↓

Linear layer (W)

↓

Logits

↓

Softmax

↓

Next character probabilities

The model learns the relationship between characters by adjusting the weights during training.

---

## Training Process

The neural network is trained using:

### Negative Log Likelihood Loss (NLL)

The objective is to maximize the probability assigned to the correct next character.

The training loop:

1. Forward pass
2. Calculate probabilities
3. Compute loss
4. Backpropagation
5. Update weights using gradient descent

The update rule:

weights = weights - learning_rate × gradient

---

## Concepts Learned

Through this project:

- Character tokenization
- Vocabulary creation
- Character-to-index mapping
- Bigram statistics
- Probability distributions
- One-hot encoding
- Matrix multiplication
- Logits
- Softmax
- Negative Log Likelihood loss
- Gradient descent
- Backpropagation using PyTorch autograd
- Neural network based language modeling

---

## Project Structure

makemore-scratch/

├── bigram_models.py  
├── names.txt  
├── README.md  
└── requirements.txt  

---

## Installation

Clone the repository:

git clone <repository-url>

Move into the project:

cd makemore-scratch

Install dependencies:

pip install -r requirements.txt

---

## Running The Model

Run:

python bigram_models.py

The script will:

- Generate names using the count-based model
- Calculate count model NLL
- Train the neural network bigram model
- Generate names using the neural network

---

## Example Output

Count Model:

Generated names:

maria.  
johan.  
lena.

Neural Network Model:

Generated names:

mariel.  
jovan.  
lina.

(Generated output changes depending on random seed.)

---

## Learning Path

This repository is part of my journey to understand modern AI systems from first principles.

The progression:

Micrograd
(backpropagation engine)

↓

Count-Based Bigram Model

↓

Neural Network Bigram Model

↓

MLP Character-Level Language Model

↓

Embeddings

↓

Transformers

↓

Large Language Models

---

## Future Improvements

- Implement MLP language model
- Add embeddings
- Add hidden layers and non-linear activation functions
- Implement batch normalization
- Build a transformer architecture from scratch

---

## Credits

Inspired by Andrej Karpathy's educational series on neural networks and language models.

This implementation is written independently for learning and experimentation.