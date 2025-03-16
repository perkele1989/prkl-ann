#include "common.hpp"
#include "layer.hpp"

#include <vector>
std::mt19937& prkl::random_device()
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return gen;
}


prkl::ann_settings &prkl::settings()
{
    static ann_settings defaults;
    return defaults;
} 


prkl::real prkl::activation(prkl::ann_layer_base const* layer, prkl::real x) 
{
    switch(layer->activation_func)
    {
        default:
        case ann_activation::swish:
            return swish(x);
        case ann_activation::tanh:
            return tanh(x);
        case ann_activation::relu:
            return relu(x);
        case ann_activation::leaky_relu:
            return leaky_relu(x, layer->leaky_alpha);
        case ann_activation::sigmoid:
            return sigmoid(x);
        case ann_activation::linear:
            return linear(x);
    }
}

prkl::real prkl::activation_derivative(prkl::ann_layer_base const* layer, prkl::real x) 
{
    switch(layer->activation_func)
    {
        default:
        case ann_activation::swish:
            return swish_derivative(x);
        case ann_activation::tanh:
            return tanh_derivative(x);
        case ann_activation::relu:
            return relu_derivative(x);
        case ann_activation::leaky_relu:
            return leaky_relu_derivative(x, layer->leaky_alpha);
        case ann_activation::sigmoid:
            return sigmoid_derivative(x);
        case ann_activation::linear:
            return linear_derivative(x);
    }
}

std::vector<std::vector<prkl::real>> softmax_derivative(std::vector<prkl::real> const& softmax_output) 
{
    size_t size = softmax_output.size();
    std::vector<std::vector<prkl::real>> jacobian(size, std::vector<prkl::real>(size));

    for (size_t i = 0; i < size; i++) 
    {
        for (size_t j = 0; j < size; j++) 
        {
            if (i == j) 
                jacobian[i][j] = softmax_output[i] * (1.0 - softmax_output[i]);
            else 
                jacobian[i][j] = -softmax_output[i] * softmax_output[j];
        }
    }

    return jacobian;
}
std::vector<prkl::real> softmax(std::vector<prkl::real> const& inputs) 
{
    std::vector<prkl::real> exp_values(inputs.size());
    prkl::real max_input = *std::max_element(inputs.begin(), inputs.end()); // Numerical stability

    prkl::real sum_exp = 0.0;
    for (size_t i = 0; i < inputs.size(); i++) 
    {
        exp_values[i] = std::exp(inputs[i] - max_input);
        sum_exp += exp_values[i];
    }

    for (size_t i = 0; i < inputs.size(); i++) 
    {
        exp_values[i] /= sum_exp;
    }

    return exp_values;
}