
#pragma once

#include "common.hpp"

#include <vector>

namespace prkl 
{

    struct ann_neuron;

    struct ann_layer
    {
        ann_layer()=default;
        ann_layer(integer num_neurons, integer num_inputs);
        
        void randomize();

        integer min_activation_index();
        integer max_activation_index();

        std::vector<ann_neuron> neurons;
    };
}