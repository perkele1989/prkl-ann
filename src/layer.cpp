
#include "layer.hpp"
#include "neuron.hpp"

prkl::ann_layer::ann_layer(prkl::integer num_neurons, prkl::integer num_inputs)
{
    neurons.resize(num_neurons, ann_neuron(num_inputs));
}

void prkl::ann_layer::randomize()
{
    for(prkl::ann_neuron &n : neurons)
        n.randomize();
}

prkl::integer prkl::ann_layer::min_activation_index()
{
    real c_min = std::numeric_limits<prkl::real>::infinity();
    integer c_index = 0;
    for(integer i = 0; i < neurons.size(); i++)
    {
        if(neurons[i].activation < c_min)
        {
            c_min = neurons[i].activation;
            c_index = i;
        }
    }

    return c_index;
}

prkl::integer prkl::ann_layer::max_activation_index()
{
    real c_max = 0.0;
    integer c_index = 0;
    for(integer i = 0; i < neurons.size(); i++)
    {
        if(neurons[i].activation > c_max)
        {
            c_max = neurons[i].activation;
            c_index = i;
        }
    }

    return c_index;
}