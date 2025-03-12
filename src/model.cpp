
#include "model.hpp"

#include <iostream>
#include <cinttypes>

prkl::ann_model::ann_model(char const* path)
{
    std::ifstream file(path, std::ios::binary);
    if(!file)
    {
        std::cerr << "failed to open file for reading: " << path << std::endl;
        return;
    }

    integer input_neurons = read_uint64_be(file); // input neurons
    integer trained_layers = read_uint64_be(file); // trained layer count (hidden+output)

    layers.resize(trained_layers + 1);
    layers[0].neurons.resize(input_neurons, ann_neuron(0));

    for(integer i = 1; i < layers.size(); i++)
    {
        ann_layer &layer = layers[i];
        integer num_neurons = read_uint64_be(file);
        layer.neurons.resize(num_neurons, ann_neuron(0));

        for(ann_neuron &n : layer.neurons)
        {
            n.bias = read_float_be(file);
            integer num_weights = read_uint64_be(file);
            n.input_weights.resize(num_weights);
            for(real &w : n.input_weights)
            {
                w = read_float_be(file);
            }
        }
    }
    
}

prkl::ann_model::ann_model(integer num_input_neurons)
{
    layers.emplace_back(num_input_neurons, 0);
}


bool prkl::ann_model::write_file(char const* path)
{
    if(layers.size() < 2)
    {
        std::cerr << "failed to write model with less than 2 layers" << std::endl;
        return false;
    }

    std::ofstream file(path, std::ios::binary);
    if(!file)
    {
        std::cerr << "failed to open file for writing: " << path << std::endl;
        return false;
    }

    write_uint64_be(file, layers[0].neurons.size()); // input neurons
    write_uint64_be(file, layers.size() - 1); // trained layer count (hidden+output)
    for(integer i = 1; i < layers.size(); i++)
    {
        ann_layer &layer = layers[i];
        write_uint64_be(file, layer.neurons.size());
        for(ann_neuron &n : layer.neurons)
        {
            write_float_be(file, n.bias);
            write_uint64_be(file, n.input_weights.size());
            for(real w : n.input_weights)
            {
                write_float_be(file, w);
            }
        }
    }

    return true;
}

bool prkl::ann_model::forward_propagate()
{
    if(layers.size() < 2)
    {
        std::cerr << "can't propagate model that has less than 2 layers" << std::endl;
        return false;
    }

    try 
    {
        for(integer layer_index = 1; layer_index < layers.size(); layer_index++)
        {
            for(prkl::ann_neuron &neuron : layers[layer_index].neurons)
            {
                neuron.activate(layers[layer_index - 1]);
            }
        }
    }
    catch(std::exception &e)
    {
        std::cerr << "neuron activation failed: " << e.what() << std::endl;
        return false;
    }

    return true;
}

void prkl::ann_model::add_layer(integer num_neurons)
{
    layers.emplace_back(num_neurons, (integer)layers[layers.size() - 1].neurons.size()).randomize();
}

prkl::ann_layer &prkl::ann_model::hidden(integer index)
{
    assert(layers.size() > 2 && index + 1 < layers.size() && "hidden layer out of bounds");
    return layers[1 + index];
}

prkl::ann_layer &prkl::ann_model::input()
{
    assert(!layers.empty() && "can't get input layer in empty network");
    return layers[0];
}

prkl::ann_layer &prkl::ann_model::output()
{
    assert(layers.size() > 1 && "can't get output layer in network with less than 2 layers");
    return layers[layers.size() - 1];
}

bool prkl::ann_model::train(ann_set &training_set, integer epochs)
{
    ann_layer &input_layer = input();
    ann_layer &output_layer = output();

    real learning_rate = settings.base_rate;
    real prev_loss = 1.0f;

    real min_loss = 1.0f;
    std::vector<ann_layer> best_layers;

    if(training_set.num_inputs != input_layer.neurons.size())
    {
        std::cerr << "input size mismatch: training set has " << training_set.num_inputs << " but model has " << input_layer.neurons.size() << std::endl;
        return false;
    }

    if(training_set.num_outputs != output_layer.neurons.size())
    {
        std::cerr << "output size mismatch: training set has " << training_set.num_outputs << " but model has " << output_layer.neurons.size() << std::endl;
        return false;
    }

    for (integer epoch = 0; epoch < epochs; ++epoch)
    {
        real total_loss = 0.0f;
        for(ann_setpair &training_pair : training_set.pairs)
        {
            // set input activations
            for (size_t i = 0; i < input_layer.neurons.size(); ++i)
            {
                input_layer.neurons[i].activation = training_pair.input[i];
            }

            // forward propagate
            if(!forward_propagate())
            {
                std::cerr << "layer propagation failed" << std::endl;
                return false;
            }

            // compute output layer error
            std::vector<real> output_errors(output_layer.neurons.size());
            for (size_t i = 0; i < output_layer.neurons.size(); ++i)
            {
                output_errors[i] = training_pair.output[i] - output_layer.neurons[i].activation;
                total_loss += output_errors[i] * output_errors[i];
            }

            // compute output layer gradients
            std::vector<real> output_gradients(output_layer.neurons.size());
            for (size_t i = 0; i < output_layer.neurons.size(); ++i)
            {
                output_gradients[i] = output_errors[i] * swish_derivative(output_layer.neurons[i].activation);
            }

            // backpropagate errors to hidden layers
            std::vector<std::vector<real>> layer_gradients(layers.size());
            layer_gradients.back() = output_gradients;

            for (integer layer_index = (integer)layers.size() - 2; layer_index > 0; --layer_index)
            {
                ann_layer &current_layer = layers[layer_index];
                ann_layer &next_layer = layers[layer_index + 1];

                layer_gradients[layer_index] = std::vector<real>(current_layer.neurons.size(), 0.0f);

                for (size_t i = 0; i < current_layer.neurons.size(); ++i)
                {
                    real sum = 0.0f;
                    for (size_t j = 0; j < next_layer.neurons.size(); ++j)
                    {
                        sum += layer_gradients[layer_index + 1][j] * next_layer.neurons[j].input_weights[i];
                    }
                    layer_gradients[layer_index][i] = sum * swish_derivative(current_layer.neurons[i].activation);
                }
            }

            // update weights and biases using gradients
            for (integer layer_index = 1; layer_index < layers.size(); ++layer_index)
            {
                ann_layer &current_layer = layers[layer_index];
                ann_layer &previous_layer = layers[layer_index - 1];

                for (size_t i = 0; i < current_layer.neurons.size(); ++i)
                {
                    for (size_t j = 0; j < previous_layer.neurons.size(); ++j)
                    {
                        current_layer.neurons[i].input_weights[j] += learning_rate * layer_gradients[layer_index][i] * previous_layer.neurons[j].activation;
                    }
                    current_layer.neurons[i].bias += learning_rate * layer_gradients[layer_index][i];
                }
            }
        }

        real avg_loss = total_loss / training_set.pairs.size();
        prev_loss = avg_loss;

        std::cout << "Epoch " << epoch << " Rate: " << learning_rate <<  " Loss: " << avg_loss << std::endl;

        if(avg_loss < min_loss)
        {
            min_loss = avg_loss;
            best_layers = layers;
        }
        
        if(avg_loss > min_loss * 1.2f)
        {
            std::cout << "Diverging, exiting early.." << std::endl;
            break;
        }

        learning_rate = adaptive_learning_rate(avg_loss);
    }

    std::cout << "Trained model to loss rate of " << min_loss << std::endl;
    layers = best_layers;

    return true;
}