# PRKL-ANN  
**An educational C++ framework for Artificial Neural Networks**  

![build status](https://github.com/perkele1989/prkl-ann/actions/workflows/cmake-multi-platform.yml/badge.svg)

## What it is  
✔ **An educational framework** with an intuitive C++ API designed for learning and experimentation.  
✔ **Supports Multi-Layer Perceptrons (MLPs)** with dense, fully connected layers.  
✔ **Implements Stochastic Gradient Descent (SGD)** for backpropagation-based learning.  
✔ **Features Adaptive Learning Rate (ALR)**, dynamically adjusting learning rates to improve convergence stability.  
✔ **Uses the Swish activation function** (and its derivative) for neuron activations.  
✔ **Highly configurable training process**, with stable defaults that converge quickly on the MNIST dataset.  
✔ **Pre-trained MNIST models included**, achieving **98.11% accuracy** at a **loss factor of 0.0432963**.  
✔ **Custom binary formats** for datasets (`.prklset`) and trained models (`.prklmodel`).  
✔ **CLI tools for training and evaluation**, supporting arbitrary datasets and supervised learning setups.  

## What it is not (yet)  
❌ Optimized for performance (memory management is lazy, and cache efficiency is not a priority).  
❌ A framework for Convolutional Neural Networks (CNNs).  
❌ A framework for Large Language Models (LLMs) or Transformers.  

## Pretrained models
The repository contains 3 models of different sizes, that are pretrained on the MNIST digits dataset. These models are simple MLP's with dense layers, and do not use convolution layers and such.

| Model | Size | Layers | Accuracy | Loss |
| ---| --- | --- | --- | --- |
|`prkl-mnist-digits-small`|210 KB|784, 64, 32, 10|96.24%|  0.0850094 |
|`prkl-mnist-digits-medium`|448 KB|784, 128, 64, 64, 10|97.36%|  0.0471145 |
|`prkl-mnist-digits-big`|1362 KB|784, 392, 98, 10|98.11%| 0.0432963 |

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

To run inference on your models, create the model in C++ using `ann_model`, then run `ann_model::forward_propagate()` and simply read the activations from its output layer.

See `mnist-digits.cpp` for a small example that does this.

## Importing datasets  

Importing datasets into prkl-ann is not well-documented, but straight-forward given the simplicity of the format.

As an example, here is a Python script that imports the original MNIST dataset for handwritten digits:

```python
from mlxtend.data import loadlocal_mnist
import numpy as np
import struct 

D, L = loadlocal_mnist(
        images_path='train-images.idx3-ubyte', 
        labels_path='train-labels.idx1-ubyte')

with open("mnnist-digits-training.prklset", "wb") as f:
    # input neurons, output neurons, num pairs 
    f.write(struct.pack("!QQQ", 28*28, 10, D.shape[0] ))

    for i in range(D.shape[0]):
        # input data
        for d in D[i]:
            f.write(struct.pack("!1f", float(d) / 255.0))
        # output data
        l = np.zeros(10, dtype=np.float32)
        l[L[i]] = 1.0
        f.write(struct.pack("!10f", *l))
```

## Roadmap  
- [ ] Implement dropout layers for regularization.  
- [ ] Add support for the Adam optimizer.  
- [ ] Introduce Convolutional Neural Networks (CNNs).  
- [ ] Explore recurrent networks for sequence-based tasks.  

