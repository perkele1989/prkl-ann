
#include "layer.hpp"

#include <omp.h>

prkl::ann_dense_layer::ann_dense_layer(integer in_neurons, integer in_inputs)
{
    num_neurons = in_neurons;
    num_inputs = in_inputs;

    activations = new real[num_neurons]();

    if(num_inputs > 0)
    {
        weights = new real[num_inputs * num_neurons]();
        biases = new real[num_neurons]();
    }
    else
    { 
        weights = nullptr;
        biases = nullptr;
    }
}

prkl::ann_dense_layer::ann_dense_layer(std::ifstream &file)
{
    num_neurons = read_uint64_be(file);
    num_inputs = read_uint64_be(file);

    activations = new real[num_neurons]();
    for(integer n = 0; n < num_neurons; n++)
    {
        activations[n] = read_float_be(file);
    }

    if(num_inputs > 0)
    {
        weights = new real[num_inputs * num_neurons]();
        biases = new real[num_neurons]();

        for(integer w = 0; w < num_neurons * num_inputs; w++)
        {
            weights[w] = read_float_be(file);
        }

        for(integer n = 0; n < num_neurons; n++)
        {
            biases[n] = read_float_be(file);
        }
    }
    else
    { 
        weights = nullptr;
        biases = nullptr;
    }
}

void prkl::ann_dense_layer::write(std::ofstream &file)
{
    write_uint64_be(file, (uint64_t)ann_layer_type::dense);
    write_uint64_be(file, num_neurons);
    write_uint64_be(file, num_inputs);

    for(integer n = 0; n < num_neurons; n++)
    {
        write_float_be(file, activations[n]);
    }

    if(num_inputs > 0)
    {
        for(integer w = 0; w < num_neurons * num_inputs; w++)
        {
            write_float_be(file, weights[w]);
        }

        for(integer n = 0; n < num_neurons; n++)
        {
            write_float_be(file, biases[n]);
        }
    }
}

prkl::ann_dense_layer::~ann_dense_layer()
{
    delete[] activations;

    if(num_inputs > 0)
    {
        delete[] biases;
        delete[] weights;
    }
}

void prkl::ann_dense_layer::randomize_weights()
{
    if(num_inputs == 0)
        return;

    auto& rnd = prkl::random_device();

    for(integer n = 0; n < num_neurons; n++)
    {
        real weight_range = std::sqrt(2.0f / num_inputs);

        // @todo use gaussian distribution instead
        std::uniform_real_distribution<real> w_dist(-weight_range, weight_range);

        real *weight_array = get_weights_array(n);
        for(integer w = 0; w < num_inputs; w++)
        {
            weight_array[w] = w_dist(rnd);
        }

        // @todo experiment with initializing biases to 0.0
        std::uniform_real_distribution<real> bias_dist(-0.1f, 0.1f);
        biases[n] = bias_dist(rnd);
    }
}

prkl::integer prkl::ann_dense_layer::min_activation_index() const
{
    real min_value = std::numeric_limits<real>::infinity();
    integer min_index = 0;
    for(integer a = 0; a < num_neurons; a++)
    {
        real curr_value = activations[a];
        if(curr_value < min_value)
        {
            min_value = curr_value;
            min_index = a;
        }
    }

    return min_index;
}

prkl::integer prkl::ann_dense_layer::max_activation_index() const
{
    real max_value = -std::numeric_limits<real>::infinity();
    integer max_index = 0;
    for(integer a = 0; a < num_neurons; a++)
    {
        real curr_value = activations[a];
        if(curr_value > max_value)
        {
            max_value = curr_value;
            max_index = a;
        }
    }

    return max_index;
}

prkl::ann_layer_base *prkl::ann_dense_layer::clone() const
{
    prkl::ann_dense_layer* new_layer = new ann_dense_layer(num_neurons, num_inputs);

    std::memcpy(new_layer->activations, activations, num_neurons * sizeof(prkl::real));

    if(num_inputs > 0)
    {
        std::memcpy(new_layer->weights, weights, num_neurons * num_inputs * sizeof(prkl::real));
        std::memcpy(new_layer->biases, biases, num_neurons * sizeof(prkl::real));
    }

    return new_layer;
}

prkl::integer prkl::ann_dense_layer::num_activations() const
{
    return num_neurons;
}

prkl::real prkl::ann_dense_layer::get_activation(prkl::integer activation_index) const
{
    assert(activation_index < num_neurons && "activation index out of range");
    return activations[activation_index];
}

void prkl::ann_dense_layer::set_activation(integer activation_index, real new_activation) 
{
    assert(activation_index < num_neurons && "activation index out of range");
    activations[activation_index] = new_activation;
}


void prkl::ann_dense_layer::forward(prkl::ann_layer_base const*prev_layer)
{
    if(num_inputs == 0)
        return;

    #pragma omp parallel for if(num_neurons >= 128)
    for(natural n = 0; n < num_neurons; n++)
    {
        activations[n] = biases[n];

        real* neuron_weights = get_weights_array(n);
        for(integer i = 0; i < num_inputs; i++)
        {
            activations[n] +=  prev_layer->get_activation(i) * neuron_weights[i];
        }
        activations[n] = swish(activations[n]);
    }
}

void prkl::ann_dense_layer::gradients_from_expected_output(std::vector<real> const& expected_output, ann_gradients &out_gradients, real &out_loss) const
{
    if(num_inputs == 0)
        return;

    out_gradients.resize(num_neurons);

    real tmp_loss = out_loss;

    #pragma omp parallel for if(num_neurons >= 128) reduction(+:tmp_loss)
    for (natural i = 0; i < num_neurons; ++i)
    {
        real output_error = expected_output[i] - activations[i];
        tmp_loss += output_error * output_error;

        out_gradients[i] = output_error * swish_derivative(activations[i]);
    }

    out_loss = tmp_loss;
}

void prkl::ann_dense_layer::gradients_backpropagate(ann_gradients const& next_gradients, ann_layer_base *next_layer, ann_gradients &out_gradients) const
{
    if(num_inputs == 0)
        return;


    out_gradients.resize(num_neurons);

    #pragma omp parallel for if(num_neurons >= 128)
    for (natural i = 0; i < num_neurons; ++i)
    {
        real sum = 0.0f;

        for (size_t j = 0; j < next_gradients.size(); ++j)
        {
            real* next_weights = next_layer->get_weights_array(j);

            sum += next_gradients[j] * next_weights[i];
        }

        out_gradients[i] = sum * swish_derivative(activations[i]);
    }
}

void prkl::ann_dense_layer::update_weights(ann_gradients const &layer_gradients, ann_layer_base const* prev_layer, real learning_rate)
{
    if(num_inputs == 0)
        return;

    for (integer i = 0; i < num_neurons; ++i)
    {
        real *neuron_weights = get_weights_array(i);
        for (integer j = 0; j < num_inputs; ++j)
        {
            neuron_weights[j] += learning_rate * layer_gradients[i] * prev_layer->get_activation(j);
        }
        biases[i] += learning_rate * layer_gradients[i];
    }
}


prkl::real* prkl::ann_dense_layer::get_weights_array(integer neuron_index) const
{
    if(num_inputs > 0)
        return weights + (num_inputs * neuron_index);

    return nullptr;
}