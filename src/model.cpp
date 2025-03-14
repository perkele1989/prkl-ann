
#include "model.hpp"

#include <iostream>
#include <cinttypes>

#define ann_model_magic 248912394734577843

prkl::ann_model::ann_model(char const* path)
{
    std::ifstream file(path, std::ios::binary);
    if(!file)
    {
        std::cerr << "failed to open file for reading: " << path << std::endl;
        return;
    }

    integer magic = read_uint64_be(file);

    if(magic != ann_model_magic)
    {
        std::cerr << "invalid model, magic mismatch" << std::endl;
        return;
    }

    integer version = read_uint64_be(file);
    if(version > (integer)ann_model_version::latest)
    {
        std::cerr << "unsupported model version, please update this software to the latest version in order to load this model" << std::endl;
        return;
    }

    integer num_layers = read_uint64_be(file);
    layers.resize(num_layers, nullptr);
   
    for(integer i = 0; i < num_layers; i++)
    {
        ann_layer_type layer_type = (ann_layer_type)read_uint64_be(file);
        ann_layer_base *new_layer = nullptr;
        switch(layer_type)
        {
            case ann_layer_type::dense:
                layers[i] = new ann_dense_layer(file);
            break;
            case ann_layer_type::convolutional:
                std::cerr << "convolutional layers not yet supported" << std::endl;
                return;
            break;
            case ann_layer_type::pooling:
                std::cerr << "pooling layers not yet supported" << std::endl;
                return;
            break;
            default:
                std::cerr << "unsupported layer type" << std::endl;
                return;
            break;
        }

    }
    
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
    std::ofstream file(path, std::ios::binary);
    if(!file)
    {
        std::cerr << "failed to open file for writing: " << path << std::endl;
        return false;
    }
    
    write_uint64_be(file, ann_model_magic);
    write_uint64_be(file, (uint64_t)ann_model_version::latest);

    write_uint64_be(file, layers.size());
    for(ann_layer_base *layer : layers)
        layer->write(file);

    return true;
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

prkl::real prkl::ann_model::evaluate(ann_set &evaluation_set)
{
    prkl::ann_layer_base *input_layer = input();
    prkl::ann_layer_base *output_layer = output();
    prkl::integer num_miss = 0;
    prkl::integer num_pairs = evaluation_set.pairs.size();

    for(prkl::integer e = 0; e <  num_pairs; e++)
    {
        prkl::ann_setpair &eval_pair = evaluation_set.pairs[e];

        for(prkl::integer i = 0; i < input_layer->num_activations(); i++)
        {
            input_layer->set_activation(i, eval_pair.input[i]);
        }

        if(!forward_propagate())
        {
            std::cerr << "forward propagation failed: evaluation failed!" << std::endl;
            return 0.0;
        }

        prkl::integer max_index_output = output_layer->max_activation_index();
        prkl::integer max_index_expected = eval_pair.max_output_index();

        if(max_index_output != max_index_expected)
        {
            num_miss++;
        }
    }

    real success_rate = (1.0 - (prkl::real(num_miss) / prkl::real(num_pairs)));
    std::cout << "Evaluated " << num_pairs << " pairs, with " << num_miss << " misses. Success rate: " << (100.0 * success_rate) << "%" << std::endl;
    return success_rate;
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