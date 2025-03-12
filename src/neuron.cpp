
#include "neuron.hpp"
#include "layer.hpp"

#include <cassert>
#include <random>

prkl::ann_neuron::ann_neuron(prkl::integer num_inputs)
{
    if(num_inputs > 0)
        input_weights.resize(num_inputs, (prkl::real)0.0);
}

prkl::real prkl::ann_neuron::activate(prkl::ann_layer const &input_layer)
{
    if(input_layer.neurons.size() != input_weights.size())
    {
        throw std::exception("incompatible input layer");
    }

    if(input_weights.empty())
        return activation;

    activation = bias;
    for(integer weight_index = 0; weight_index < input_weights.size(); weight_index++)
    {
        ann_neuron const& input_neuron = input_layer.neurons.at(weight_index);
        real const& input_weight = input_weights.at(weight_index);
        activation += input_neuron.activation * input_weight;
    }
    activation = swish(activation);
    return activation;
}


void prkl::ann_neuron::randomize()
{
    auto& rnd = prkl::random_device();

    real weight_range = std::sqrt(2.0f / input_weights.size());

    // @todo use gaussian distribution instead
    std::uniform_real_distribution<real> w_dist(-weight_range, weight_range);

    for(prkl::real &w : input_weights)
    {
        w = w_dist(rnd);
    }

    // @todo experiment with initializing biases to 0.0
    std::uniform_real_distribution<real> bias_dist(-0.1f, 0.1f);
    bias = bias_dist(rnd);
}