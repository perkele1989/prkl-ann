
#include "layer.hpp"

#include <omp.h>

prkl::ann_layer_base::ann_layer_base(nlohmann::json &cfg)
{
    if(cfg.contains("activation_func"))
    {
        std::string activation_func_str = cfg.at("activation_func").template get<std::string>();
        if(activation_func_str == "linear")
        {
            activation_func = prkl::ann_activation::linear;
            std::cout << "model config: layer activation: linear" << std::endl;
        }
        else if(activation_func_str == "sigmoid")
        {
            activation_func = prkl::ann_activation::sigmoid;
            std::cout << "model config: layer activation: sigmoid" << std::endl;
        } 
        else if(activation_func_str == "leaky_relu")
        {
            activation_func = prkl::ann_activation::leaky_relu;
            std::cout << "model config: layer activation: leaky_relu" << std::endl;
        } 
        else if(activation_func_str == "relu")
        {
            activation_func = prkl::ann_activation::relu;
            std::cout << "model config: layer activation: relu" << std::endl;
        } 
        else if(activation_func_str == "tanh")
        {
            activation_func = prkl::ann_activation::tanh;
            std::cout << "model config: layer activation: tanh" << std::endl;
        } 
        else if(activation_func_str == "swish")
        {
            activation_func = prkl::ann_activation::swish;
            std::cout << "model config: layer activation: swish" << std::endl;
        } 
        else
        {
            std::cerr << "unrecognized activation_func: " << activation_func_str << std::endl;
        }
    }

    if(cfg.contains("leaky_alpha"))
    {
        leaky_alpha = cfg.at("leaky_alpha").template get<prkl::real>();
        std::cout << "model config: layer activation: leaky alpha:" << leaky_alpha << std::endl;
    }
}

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

prkl::ann_dense_layer::ann_dense_layer(nlohmann::json &cfg) 
    : prkl::ann_layer_base(cfg)
{
    if(!cfg.contains("num_neurons") || !cfg.contains("num_inputs"))
    {
        std::cerr << "num_neurons and num_inputs required for dense layers" << std::endl;
        return;
    }

    num_neurons = cfg.at("num_neurons").template get<prkl::integer>();
    num_inputs = cfg.at("num_inputs").template get<prkl::integer>();

    std::cout << "model config: num_neurons: " << num_neurons << std::endl;
    std::cout << "model config: num_inputs: " << num_inputs << std::endl;

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

prkl::ann_dense_layer::ann_dense_layer(std::ifstream &file, ann_model_version version)
{
    if(version >= ann_model_version::layer_parameters)
    {
        activation_func = (ann_activation)read_uint64_be(file);
        leaky_alpha = read_float_be(file);
    }
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

    write_uint64_be(file, (uint64_t)activation_func);
    write_float_be(file, leaky_alpha);

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
    new_layer->activation_func = activation_func;
    new_layer->leaky_alpha = leaky_alpha;

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

void prkl::ann_dense_layer::forward(prkl::ann_layer_base const* prev_layer)
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
            activations[n] += prev_layer->get_activation(i) * neuron_weights[i];
        }
        activations[n] = activation(this, activations[n]);
    }

}

void prkl::ann_dense_layer::apply_softmax()
{
    real max_activation = activations[0];
    for (natural i = 1; i < num_neurons; i++) {
        max_activation = std::max(max_activation, activations[i]);
    }

    real sum_exp = 0.0;
    for (natural i = 0; i < num_neurons; i++) {
        real shifted = std::max(activations[i] - max_activation, -80.0f); // Prevent extreme underflow
        activations[i] = std::exp(shifted);
        sum_exp += activations[i];
    }

    sum_exp += 1e-08f;  // Avoid division by zero
    for (natural i = 0; i < num_neurons; i++) {
        activations[i] /= sum_exp;
    }
}

// void prkl::ann_dense_layer::forward(prkl::ann_layer_base const*prev_layer)
// {
//     if(num_inputs == 0)
//         return;

//     #pragma omp parallel for if(num_neurons >= 128)
//     for(natural n = 0; n < num_neurons; n++)
//     {
//         activations[n] = biases[n];

//         real* neuron_weights = get_weights_array(n);
//         for(integer i = 0; i < num_inputs; i++)
//         {
//             activations[n] +=  prev_layer->get_activation(i) * neuron_weights[i];
//         }
//         activations[n] = activation(this, activations[n]);
//     }
// }

// void prkl::ann_dense_layer::gradients_from_expected_output(std::vector<real> const& expected_output, ann_gradients &out_gradients, real &out_loss) const
// {
//     if(num_inputs == 0)
//         return;

//     out_gradients.resize(num_neurons);

//     real tmp_loss = out_loss;

//     #pragma omp parallel for if(num_neurons >= 128) reduction(+:tmp_loss)
//     for (natural i = 0; i < num_neurons; ++i)
//     {
//         real output_error = expected_output[i] - activations[i];
//         tmp_loss += output_error * output_error;

//         out_gradients[i] = output_error * activation_derivative(this, activations[i]);
//     }

//     out_loss = tmp_loss;
// }


void prkl::ann_dense_layer::gradients_from_expected_output(ann_evaluation_type evaluation_type,  ann_loss_function loss_function,std::vector<real> const& expected_output, ann_gradients &out_gradients, real &out_loss) const
{
    if(num_inputs == 0)
        return;

    out_gradients.resize(num_neurons);
    real tmp_loss = out_loss;

    #pragma omp parallel for if(num_neurons >= 128) reduction(+:tmp_loss)
    for (natural i = 0; i < num_neurons; ++i)
    {
        real output_error = expected_output[i] - activations[i];

        switch (evaluation_type)
        {
            case ann_evaluation_type::regression:
                if (loss_function == ann_loss_function::mean_squared_error)
                {
                    tmp_loss += output_error * output_error; // MSE
                    out_gradients[i] = output_error * activation_derivative(this, activations[i]);
                }
                else if (loss_function == ann_loss_function::mean_absolute_error)
                {
                    tmp_loss += std::abs(output_error);  // MAE
                    out_gradients[i] = (output_error >= 0 ? 1.0 : -1.0) * activation_derivative(this, activations[i]);
                }
                break;
            case ann_evaluation_type::multiclass_classification:
                // Cross-entropy loss, assumes softmax was applied
                tmp_loss -= expected_output[i] * std::log( std::max(activations[i],  1e-08f));  // Avoid log(0)
                out_gradients[i] = output_error * activation_derivative(this, activations[i]);
                break;
        
            case ann_evaluation_type::binary_classification:
            case ann_evaluation_type::multilabel_classification:
                // Binary cross-entropy loss (BCE)
                tmp_loss -= expected_output[i] * std::log(activations[i] + 1e-5f) + (1 - expected_output[i]) * std::log(1 - activations[i] + 1e-5f);
                out_gradients[i] = output_error * activation_derivative(this, activations[i]);
                break;
        }
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

        out_gradients[i] = sum * activation_derivative(this, activations[i]);
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