
#pragma once 

#include "neuron.hpp"
#include "layer.hpp"
#include "set.hpp"

namespace prkl 
{
    /** A network with dense (fully connected) layers */
    struct ann_model 
    {
        ann_model(char const* path);
        ann_model(integer num_input_neurons);
        void add_layer(integer num_neurons);

        bool write_file(char const* path);

        bool forward_propagate();

        ann_layer &hidden(integer index);
        ann_layer &input();
        ann_layer &output();

        bool train(ann_set &training_set, integer epochs);

        std::vector<ann_layer> layers;
    };

}