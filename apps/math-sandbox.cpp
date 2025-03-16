
#include "model.hpp"
#include "cmdparser.hpp"
#include <iostream>
#include <iomanip>

prkl::ann_set generate_set_dot(prkl::integer num_pairs)
{
    auto &rnd = prkl::random_device();
    std::uniform_real_distribution<prkl::real> dist(-1.0, 1.0);

    prkl::ann_set set(6, 1);

    set.pairs.reserve(num_pairs);
    for(prkl::integer i = 0; i < num_pairs; i++)
    {
        prkl::ann_setpair new_pair;
        new_pair.input.reserve(6);
        new_pair.output.reserve(1);

        prkl::real a1, a2, a3, b1, b2, b3, r;
        a1 = dist(rnd);
        a2 = dist(rnd);
        a3 = dist(rnd);
        b1 = dist(rnd);
        b2 = dist(rnd);
        b3 = dist(rnd);

        prkl::real alen_inv = 1.0f / sqrtf(a1 * a1 + a2 * a2 + a3 * a3);
        a1 *= alen_inv;
        a2 *= alen_inv;
        a3 *= alen_inv;

        prkl::real blen_inv = 1.0f / sqrtf(b1 * b1 + b2 * b2 + b3 * b3);
        b1 *= blen_inv;
        b2 *= blen_inv;
        b3 *= blen_inv;

        r = a1 * b1 + a2 * b2 + a3 * b3;

        new_pair.input.push_back(a1);
        new_pair.input.push_back(a2);
        new_pair.input.push_back(a3);
        new_pair.input.push_back(b1);
        new_pair.input.push_back(b2);
        new_pair.input.push_back(b3);

        new_pair.output.push_back(r);

        set.pairs.push_back(new_pair);
    }

    return set;
}

int32_t main(int32_t argc, char **argv)
{
    // cli::Parser parser(argc, argv);
    // parser.set_required<std::string>("m", "model", "Path to model (.prklmodel file)");
    // parser.set_required<std::string>("e", "evaluation-set", "", "Path to evaluation set (.prklset file)");
    // parser.run_and_exit_if_error();

    prkl::settings().early_exit = true;
    prkl::settings().loss_edge = 0.001f;
    prkl::settings().alr = false;
    prkl::settings().grad_limit = 1.0f;
    prkl::settings().base_rate = 0.01f;
    
    prkl::ann_model model_dot;
    model_dot.regression_loss_function = prkl::ann_loss_function::mean_squared_error;
    model_dot.evaluation_type = prkl::ann_evaluation_type::regression;
    
    model_dot.add_dense_layer(6);

    
    prkl::ann_dense_layer *hidden1 = model_dot.add_dense_layer(64);
    hidden1->activation_func = prkl::ann_activation::relu;

    prkl::ann_dense_layer *hidden2 = model_dot.add_dense_layer(32);
    hidden2->activation_func = prkl::ann_activation::relu;


    // prkl::ann_dense_layer *hidden1 = model_dot.add_dense_layer(64);
    // hidden1->activation_func = prkl::ann_activation::tanh;

    // prkl::ann_dense_layer *hidden2 = model_dot.add_dense_layer(32);
    // hidden2->activation_func = prkl::ann_activation::tanh;
    
    prkl::ann_dense_layer *output = model_dot.add_dense_layer(1);
    output->activation_func = prkl::ann_activation::linear;

    prkl::ann_set set_dot = generate_set_dot(25000);
    model_dot.train(set_dot, 50);

    
    prkl::ann_set eval_dot = generate_set_dot(10000);

    prkl::ann_layer_base *input_layer = model_dot.input();
    prkl::ann_layer_base *output_layer = model_dot.output();
    prkl::integer num_miss = 0;
    prkl::integer num_pairs = eval_dot.pairs.size();

    prkl::real total_loss =0.0f;
    prkl::real max_loss = 0.0f;
    prkl::real min_loss = std::numeric_limits<prkl::real>::infinity();
    for(prkl::integer e = 0; e <  num_pairs; e++)
    {
        prkl::ann_setpair &eval_pair = eval_dot.pairs[e];

        for(prkl::integer i = 0; i < input_layer->num_activations(); i++)
        {
            input_layer->set_activation(i, eval_pair.input[i]);
        }

        if(!model_dot.forward_propagate())
        {
            std::cerr << "forward propagation failed: evaluation failed!" << std::endl;
            return 0.0;
        }

        prkl::real expected = eval_pair.output[0];
        prkl::real actual = output_layer->get_activation(0);
        prkl::real loss = std::fabs(actual - expected); 
        total_loss += loss;
        if(loss < min_loss)
        {
            min_loss = loss;
        }

        if(loss > max_loss)
        {
            max_loss = loss;
        }

        
    }

    
    std::cout << "Difference: avg: "<<std::setprecision(6) << std::fixed << (total_loss / (prkl::real)num_pairs) << ", min: " << min_loss << ", max: " << max_loss << std::endl;


    return 0;
}