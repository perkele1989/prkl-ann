
#pragma once 

#include "layer.hpp"
#include "set.hpp"

namespace prkl 
{

    struct ann_model;

    struct ann_snapshot 
    {
        ann_snapshot(ann_model & model);
        ~ann_snapshot();

        void update(ann_model &model);

        std::vector<ann_layer_base*> layers;
    };

    /** A network with dense (fully connected) layers */
    struct ann_model 
    {
        ann_model(char const* path);
        ann_model(nlohmann::json &cfg);
        ann_model()=default;
        ~ann_model();

        ann_model clone();

        ann_dense_layer* add_dense_layer(integer num_neurons);

        bool write_file(char const* path);

        bool forward_propagate();

        ann_layer_base *hidden(integer index);
        ann_layer_base *input();
        ann_layer_base *output();

        bool train(ann_set &training_set, integer epochs, ann_set *underfit_set = nullptr);
        real evaluate(ann_set &evaluation_set);

        void apply_snapshot(ann_snapshot const& snapshot);

        ann_evaluation_type evaluation_type{ann_evaluation_type::regression};
        ann_loss_function regression_loss_function{ann_loss_function::mean_squared_error};

        std::vector<ann_layer_base*> layers;
    };

}