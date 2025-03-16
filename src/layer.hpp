
#pragma once

#include "common.hpp"

#include <vector>

namespace prkl 
{

    struct ann_neuron;

    using ann_gradients = std::vector<real>;

    struct ann_layer_base 
    {
        ann_layer_base()=default;
        ann_layer_base(nlohmann::json &cfg);
        virtual ~ann_layer_base()=default;

        virtual void write(std::ofstream &file)=0;

        virtual ann_layer_base *clone() const = 0;
        virtual integer num_activations() const = 0;
        
        virtual integer min_activation_index() const = 0;
        virtual integer max_activation_index() const = 0;
        virtual void apply_softmax() =0;
        virtual real get_activation(integer activation_index) const = 0;
        virtual void set_activation(integer activation_index, real new_activation) = 0;
        virtual void forward(ann_layer_base const*prev_layer) = 0; 
        virtual void gradients_from_expected_output(ann_evaluation_type evaluation_type,  ann_loss_function loss_function,std::vector<real> const& expected_output, ann_gradients &out_gradients, real &out_loss) const  = 0;
        virtual void gradients_backpropagate(ann_gradients const& next_gradients, ann_layer_base *next_layer,ann_gradients &out_gradients) const  = 0;
        virtual void update_weights(ann_gradients const &layer_gradients, ann_layer_base const* prev_layer, real learning_rate) = 0;
        
        virtual real* get_weights_array(integer neuron_index) const =0;

        ann_activation activation_func {ann_activation::linear};
        real leaky_alpha{(real)0.01};
    };

    struct ann_dense_layer : public ann_layer_base
    {
        ann_dense_layer(nlohmann::json &cfg);
        ann_dense_layer(integer num_neurons, integer num_inputs);
        ann_dense_layer(std::ifstream &file, ann_model_version version);
        virtual ~ann_dense_layer();
        virtual void write(std::ofstream &file) override;
        void randomize_weights();
        
        virtual integer min_activation_index() const override;
        virtual integer max_activation_index() const override;

        virtual ann_layer_base *clone() const override;
        virtual integer num_activations() const override;
        virtual real get_activation(integer activation_index) const override;
        virtual void set_activation(integer activation_index, real new_activation) override;

        virtual void forward(ann_layer_base const*prev_layer) override;
        virtual void apply_softmax() override;
        virtual void gradients_from_expected_output(ann_evaluation_type evaluation_type, ann_loss_function loss_function, std::vector<real> const& expected_output, ann_gradients &out_gradients, real &out_loss) const override;
        virtual void gradients_backpropagate(ann_gradients const& next_gradients, ann_layer_base *next_layer,  ann_gradients &out_gradients) const override;
        virtual void update_weights(ann_gradients const &layer_gradients, ann_layer_base const* prev_layer, real learning_rate) override;

        virtual real* get_weights_array(integer neuron_index) const override;

        integer num_neurons; // how many neurons this layer has
        integer num_inputs; // how many input neurons this layer has been configured for 

        real *activations; // num_neurons
        real *biases; // num_neurons
        real* weights; // num_neurons * num_inputs, stored in row-major order, x = neuron index, y = input index 
    };
}