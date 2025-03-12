# PRKL-ANN  
**An educational C++ framework for Artificial Neural Networks**  

## What it is  
✔ **An educational framework** with an intuitive C++ API designed for learning and experimentation.  
✔ **Supports Multi-Layer Perceptrons (MLPs)** with dense, fully connected layers.  
✔ **Implements Stochastic Gradient Descent (SGD)** for backpropagation-based learning.  
✔ **Features Adaptive Learning Rate (ALR)**, dynamically adjusting learning rates to improve convergence stability.  
✔ **Uses the Swish activation function** (and its derivative) for neuron activations.  
✔ **Highly configurable training process**, with stable defaults that converge quickly on the MNIST dataset.  
✔ **Pre-trained MNIST models included**, achieving **97.59% accuracy** at a **loss factor of 0.0466**.  
✔ **Custom binary formats** for datasets (`.prklset`) and trained models (`.prklmodel`).  
✔ **CLI tools for training and evaluation**, supporting arbitrary datasets and supervised learning setups.  

## What it is not (yet)  
❌ Optimized for performance (memory management is lazy, and cache efficiency is not a priority).  
❌ A framework for Convolutional Neural Networks (CNNs).  
❌ A framework for Large Language Models (LLMs) or Transformers.  

## Installation  

Build via CMake, or simply open the folder in VSCode and hit build.

```sh
# Clone the repository
git clone https://github.com/perkele1989/prkl-ann.git
cd prkl-ann

# Build using CMake
mkdir build && cd build
cmake ..
make
```

## Example Usage  
```sh
# Train a model with a dataset
prkl-train -t dataset.prklset -o model.prklmodel -p 50 -c 784,128,64,64,10

# Evaluate the trained model (preferrably with a different set than what you trained it on)
prkl-evaluate -e evaluation.prklset -m model.prklmodel
```

## Roadmap  
- [ ] Implement dropout layers for regularization.  
- [ ] Add support for the Adam optimizer.  
- [ ] Introduce Convolutional Neural Networks (CNNs).  
- [ ] Explore recurrent networks for sequence-based tasks.  

