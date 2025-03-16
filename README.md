# PRKL-ANN  
**An educational C++ framework for Artificial Neural Networks**  

![build status](https://github.com/perkele1989/prkl-ann/actions/workflows/cmake-multi-platform.yml/badge.svg)

## What it is  
✔ **An educational framework** with an intuitive C++ API designed for learning and experimentation.  
✔ **Multi-Layer Perceptrons (MLPs)** with dense, fully connected layers. *(Convolutional/Pooling layers are in progress!)*  
✔ **Stochastic Gradient Descent (SGD)** for backpropagation-based learning.  
✔ **Activation Functions**: Linear, ReLU, Leaky ReLU, Swish, Tanh, Sigmoid.  
✔ **Evaluation Types**: Regression, Multiclass (with Softmax), Binary, Multilabel.  
✔ **Adaptive Learning Rate (ALR)**, dynamically adjusting learning rates to improve convergence stability.  
✔ **Automatic Divergence Detection**, detects loss-based divergence and early exits to speed up training.  
✔ **Automatic Overfitting Prevention**, detects evaluated success divergence, and early exits to prevent overfitting.  
✔ **Highly configurable training process** with stable defaults that converge quickly on the MNIST dataset.  
✔ **Pre-trained MNIST models included**, achieving a **98.55% success rate** against the evaluation set.  
✔ **Custom .json model configs** to make building and training models more convenient.  
✔ **Custom binary formats** for datasets (`.prklset`) and trained models (`.prklmodel`).  
✔ **CLI tools for training and evaluation**, supporting arbitrary datasets and supervised learning setups.  

## TODO
❌ Convolutional Layer Support (check out `convolutional-layers` branch)  
❌ Transformers/LLM's (longterm plans)

## Pretrained models
The repository contains 3 models of different sizes, that are pretrained on the MNIST digits dataset. These models are simple MLP's with dense layers, and do not use convolution layers. As such, they do not generalize well, and these models are mostly here as a performance metric for the framework.

| Model | Size | Layers | Accuracy |
| ---| --- | --- | --- |
|`prkl-mnist-digits-small`|210 KB|784, 64, 32, 10|97.84%|
|`prkl-mnist-digits-medium`|448 KB|784, 128, 64, 64, 10|98.27%|
|`prkl-mnist-digits-big`|1362 KB|784, 392, 98, 10|98.55%|

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

First create a model configuration (`.json` file) like so:

```json
{
    "evaluation_type": "multiclass_classification",
    "layers": [
        {
            "type": "dense",
            "num_neurons": 784,
            "num_inputs": 0
        },
        {
            "type": "dense",
            "num_neurons": 64,
            "num_inputs": 784,
            "activation_func": "leaky_relu"
        },
        {
            "type": "dense",
            "num_neurons": 32,
            "num_inputs": 64,
            "activation_func": "leaky_relu"
        },
        {
            "type": "dense",
            "num_neurons": 10,
            "num_inputs": 32,
            "activation_func": "linear"
        }
    ]
}
```

Then train it:

```sh
# Train a model with a dataset, and a model configuration passed as json
prkl-train -t dataset.prklset -o model.prklmodel -p 50 -c model.json

# Train a model like above, but with an evaluation set, this does automatic overfitting prevention!
prkl-train -t dataset.prklset -e evaluation.prklset -o model.prklmodel -p 50 -c model.json

# Evaluate a pre-trained model
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

