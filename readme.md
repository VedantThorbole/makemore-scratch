# Makemore Scratch

A character-level language model implemented from scratch using PyTorch.

Inspired by Andrej Karpathy's makemore series.

Implementation written from scratch for learning purposes.

---

## Project Overview

This project implements a **Bigram Character-Level Language Model**.

A bigram model learns the probability of the next character based on the current character.

The model learns:

P(next_character | current_character)

For example:

Given the character:

a

The model learns the probability of possible next characters:

a → b  
a → n  
a → r  

Using these probabilities, the model can generate new names.

---

## Dataset

The model is trained on a dataset of names.

Each name is treated as a sequence of characters.

Example:

emma

is converted into:

. e m m a .

The `.` character represents the start and end token.

The model learns character transitions:

. → e  
e → m  
m → m  
m → a  
a → .

---

## How It Works

### 1. Character Vocabulary

All unique characters from the dataset are converted into numerical IDs.

Example:

a → 1  
b → 2  
c → 3  
. → 0  

This allows characters to be represented mathematically.

---

### 2. Building Bigram Counts

A 27 × 27 matrix is created.

Each cell stores how many times one character follows another.

Example:

If:

a → b happened 10 times  
a → c happened 5 times  
a → d happened 15 times  

The matrix stores these transition frequencies.

---

### 3. Converting Counts Into Probabilities

The count matrix is normalized row-wise.

Example:

Before normalization:

a → b = 10  
a → c = 5  
a → d = 15  

After normalization:

P(b|a) = 0.33  
P(c|a) = 0.16  
P(d|a) = 0.50  

Each row now represents a probability distribution for the next character.

---

### 4. Generating Names

The model starts from the start token:

.

It samples the next character based on learned probabilities.

Example:

. → m → a → r → i → .

The generated sequence becomes a new name.

---

## Loss Function

The model is evaluated using:

Negative Log Likelihood (NLL)

The goal is to maximize the probability assigned to real character sequences.

A lower NLL means the model assigns better probabilities to real data.

Formula:

Loss = -log(probability of correct next character)

---

## Concepts Learned

- Character tokenization
- Vocabulary creation
- String processing
- Dictionary mappings
- Bigram statistics
- Probability distributions
- Sampling from probability distributions
- PyTorch tensors
- Random generators
- Negative Log Likelihood loss
- Basics of language modeling

---

## Project Structure

makemore-scratch/

makemore-scratch/

├── bigram_model.py
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

python bigram_model.py

Example output:

Generated names:

maria.  
johan.  
lena.

The generated names will change depending on the random seed.

---

## Technologies Used

- Python
- PyTorch

---

## Learning Path

This repository is part of my journey to understand modern AI systems from first principles.

The progression:

Bigram Language Model

↓

MLP Character-Level Language Model

↓

Backpropagation From Scratch

↓

Neural Networks

↓

Transformers

↓

Large Language Models

---

## Future Improvements

- Add neural network based language model
- Implement embeddings
- Implement backpropagation manually
- Build an MLP language model
- Implement a transformer architecture from scratch

---

## Credits

Inspired by Andrej Karpathy's educational series on neural networks and language models.

This implementation is written independently for learning and experimentation.