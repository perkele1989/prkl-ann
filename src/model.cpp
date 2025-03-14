
#include "model.hpp"

#include <iostream>
#include <cinttypes>

prkl::ann_model::ann_model(char const* path)
{
    // std::ifstream file(path, std::ios::binary);
    // if(!file)
    // {
    //     std::cerr << "failed to open file for reading: " << path << std::endl;
    //     return;
    // }

    // integer input_neurons = read_uint64_be(file); // input neurons
    // integer trained_layers = read_uint64_be(file); // trained layer count (hidden+output)

    // layers.resize(trained_layers + 1);
    // layers[0].neurons.resize(input_neurons, ann_neuron(0));

    // for(integer i = 1; i < layers.size(); i++)
    // {
    //     ann_layer &layer = layers[i];
    //     integer num_neurons = read_uint64_be(file);
    //     layer.neurons.resize(num_neurons, ann_neuron(0));

    //     for(ann_neuron &n : layer.neurons)
    //     {
    //         n.bias = read_float_be(file);
    //         integer num_weights = read_uint64_be(file);
    //         n.input_weights.resize(num_weights);
    //         for(real &w : n.input_weights)
    //         {
    //             w = read_float_be(file);
    //         }
    //     }
    // }
    
}

prkl::ann_model::~ann_model()
{
    for(ann_layer_base *layer : layers)
    {
        delete layer;
    }
}

bool prkl::ann_model::write_file(char const* path)
{
    return false;
    // if(layers.size() < 2)
    // {
    //     std::cerr << "failed to write model with less than 2 layers" << std::endl;
    //     return false;
    // }

    // std::ofstream file(path, std::ios::binary);
    // if(!file)
    // {
    //     std::cerr << "failed to open file for writing: " << path << std::endl;
    //     return false;
    // }

    // write_uint64_be(file, layers[0].neurons.size()); // input neurons
    // write_uint64_be(file, layers.size() - 1); // trained layer count (hidden+output)
    // for(integer i = 1; i < layers.size(); i++)
    // {
    //     ann_layer &layer = layers[i];
    //     write_uint64_be(file, layer.neurons.size());
    //     for(ann_neuron &n : layer.neurons)
    //     {
    //         write_float_be(file, n.bias);
    //         write_uint64_be(file, n.input_weights.size());
    //         for(real w : n.input_weights)
    //         {
    //             write_float_be(file, w);
    //         }
    //     }
    // }

    // return true;
}

bool prkl::ann_model::forward_propagate()
{
    if(layers.size() < 2)
    {
        std::cerr << "can't propagate model that has less than 2 layers" << std::endl;
        return false;
    }

    for(integer layer_index = 1; layer_index < layers.size(); layer_index++)
    {
        ann_layer_base *layer = layers[layer_index];
        ann_layer_base *prev_layer = layers[layer_index - 1];
        layer->forward(prev_layer);
    }

    return true;
}

void prkl::ann_model::add_dense_layer(integer num_neurons)
{
    integer prev_activations = 0;
    if(!layers.empty())
    {
        ann_layer_base *last_layer = layers.back();
        prev_activations = last_layer->num_activations();
    }
    ann_dense_layer *new_layer = new ann_dense_layer(num_neurons, prev_activations);
    new_layer->randomize_weights();
    layers.push_back(new_layer);
}

prkl::ann_layer_base *prkl::ann_model::hidden(integer index)
{
    assert(layers.size() > 2 && index + 1 < layers.size() && "hidden layer out of bounds");
    return layers[1 + index];
}

prkl::ann_layer_base *prkl::ann_model::input()
{
    assert(!layers.empty() && "can't get input layer in empty network");
    return layers.front();
}

prkl::ann_layer_base *prkl::ann_model::output()
{
    assert(layers.size() > 1 && "can't get output layer in network with less than 2 layers");
    return layers.back();
}


prkl::ann_model prkl::ann_model::clone()
{
    ann_model returner;
    returner.layers.reserve(layers.size());

    for(ann_layer_base *l : layers)
    {
        returner.layers.push_back(l->clone());
    }

    return returner;
}

bool prkl::ann_model::train(ann_set &training_set, integer epochs)
{
    ann_layer_base *input_layer = input();
    ann_layer_base *output_layer = output();

    real learning_rate = settings.base_rate;

    real min_loss = 1.0f;
    ann_snapshot best_model = ann_snapshot(*this);

    if(training_set.num_inputs != input_layer->num_activations())
    {
        std::cerr << "input size mismatch: training set has " << training_set.num_inputs << " but model has " << input_layer->num_activations() << std::endl;
        return false;
    }

    if(training_set.num_outputs != output_layer->num_activations())
    {
        std::cerr << "output size mismatch: training set has " << training_set.num_outputs << " but model has " << output_layer->num_activations() << std::endl;
        return false;
    }

    for (integer epoch = 0; epoch < epochs; ++epoch)
    {
        real total_loss = 0.0f;
        for(ann_setpair &training_pair : training_set.pairs)
        {
            // set input activations
            for (size_t i = 0; i < input_layer->num_activations(); ++i)
            {
                input_layer->set_activation(i, training_pair.input[i]);
            }

            // forward propagate
            if(!forward_propagate())
            {
                std::cerr << "layer propagation failed" << std::endl;
                return false;
            }

            // calculate gradients from expected output
            std::vector<ann_gradients> layer_gradients(layers.size()-1);
            output_layer->gradients_from_expected_output(training_pair.output, layer_gradients.back(), total_loss);

            for (integer layer_index = (integer)layers.size() - 2; layer_index > 0; --layer_index)
            {
                ann_gradients &curr_gradients = layer_gradients[layer_index - 1];
                ann_gradients &next_gradients = layer_gradients[layer_index];

                ann_layer_base *current_layer = layers[layer_index];
                ann_layer_base *next_layer = layers[layer_index + 1];
                current_layer->gradients_backpropagate(next_gradients, next_layer, curr_gradients);
            }

            for (integer layer_index = 1; layer_index < layers.size(); ++layer_index)
            {
                ann_layer_base *current_layer = layers[layer_index];
                ann_layer_base *previous_layer = layers[layer_index - 1];
                ann_gradients &curr_gradients = layer_gradients[layer_index - 1];
                current_layer->update_weights(curr_gradients, previous_layer, learning_rate);
            }
        }

        real avg_loss = total_loss / training_set.pairs.size();

        if(avg_loss < min_loss)
        {
            min_loss = avg_loss;
            best_model.update(*this);
            std::cout << "Epoch " << epoch << " Rate: " << learning_rate <<  " Loss: " << avg_loss << "(Snapshot)" << std::endl;
        }
        else 
        {
            std::cout << "Epoch " << epoch << " Rate: " << learning_rate <<  " Loss: " << avg_loss << std::endl;
        }
        

        
        if(avg_loss > min_loss * 1.2f)
        {
            std::cout << "Diverging, exiting early.." << std::endl;
            break;
        }

        learning_rate = adaptive_learning_rate(avg_loss);
    }

    std::cout << "Trained model to loss rate of " << min_loss << std::endl;
    apply_snapshot(best_model);

    return true;
}

prkl::ann_snapshot::ann_snapshot(ann_model &model)
{
    for(ann_layer_base *l : model.layers)
    {
        layers.push_back(l->clone());
    }
}


void prkl::ann_snapshot::update(ann_model &model)
{
    for(ann_layer_base *l : layers)
    {
        delete l;
    }
    layers.clear();
    for(ann_layer_base *l : model.layers)
    {
        layers.push_back(l->clone());
    }
}

prkl::ann_snapshot::~ann_snapshot()
{
    for(ann_layer_base* l : layers)
    {
        delete l;
    }
}

void prkl::ann_model::apply_snapshot(ann_snapshot const& snapshot)
{
    for(ann_layer_base *l : layers)
        delete l;
    layers.clear();

    for(ann_layer_base *l : snapshot.layers)
    {
        layers.push_back(l->clone());
    }
}