# Neural Network from Scratch

A feedforward neural network implemented from scratch using NumPy. The project is intended as a learning resource for the fundamentals of neural networks, without relying on deep learning libraries such as TensorFlow or PyTorch.

---

## Features

- Forward propagation
- Activation functions
- Softmax and cross-entropy loss
- Prediction logic
- Accuracy calculation
- Educational code comments

---

## Repository Contents

- `neural_network.ipynb`: A step-by-step Jupyter Notebook containing the full implementation and walkthrough.

---

## Requirements

- Python 3.x
- NumPy
- Jupyter Notebook or Google Colab

---

## Getting Started

### Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `neural_network.ipynb`.
3. Run the notebook cells sequentially.

### Local Setup

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
jupyter notebook neural_network.ipynb
```

---

## Sample Functions

```python
def get_predictions(a2):
    return np.argmax(a2, 0)

def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size
```

These functions extract predicted labels from the output layer and compute classification accuracy.

---

## Learning Objectives

By working through the notebook, you will learn how to:

- Structure a basic neural network
- Perform forward propagation using matrix operations
- Apply softmax and compute categorical cross-entropy loss
- Derive predictions and evaluate accuracy
- Understand why the gradient of the loss with respect to Z2 simplifies to `A2 - Y` when using softmax with cross-entropy

---

## Project Structure

```
neural_network/
├── neural_network.ipynb   # Main notebook
└── README.md              # Project documentation
```

---

## References

- *Derivative of the Softmax Function and the Categorical Cross-Entropy Loss*: This article explains how the derivative of softmax combined with cross-entropy loss simplifies to `A2 - Y`, a key result in backpropagation.

---

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute it.
