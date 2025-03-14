
#include "model.hpp"
#include "cmdparser.hpp"
#include <iostream>

int32_t main(int32_t argc, char **argv)
{
    cli::Parser parser(argc, argv);
    parser.set_required<std::string>("t", "training-set", "Path to training set (.prklset file)");
    parser.set_optional<std::string>("e", "evaluation-set", "", "Path to evaluation set (.prklset file)");
    parser.set_optional<prkl::real>("a", "alr-ease-alpha", prkl::settings.ease_alpha, "Adaptive Learning Rate: Ease alpha");
    parser.set_optional<bool>("s", "alr-ease", prkl::settings.ease, "Adaptive Learning Rate: Ease");
    parser.set_optional<prkl::real>("b", "alr-base-rate", prkl::settings.base_rate, "Adaptive Learning Rate: Base rate");
    parser.set_optional<prkl::real>("m", "alr-min-rate", prkl::settings.min_rate, "Adaptive Learning Rate: Minimum rate");
    parser.set_optional<prkl::real>("l", "alr-loss-edge", prkl::settings.loss_edge, "Adaptive Learning Rate: Loss edge");
    parser.set_optional<prkl::real>("g", "grad-limit", prkl::settings.grad_limit, "Maximum gradient amplitude");
    parser.set_optional<std::string>("o", "output", "", "Path to output file (.prklmodel file)");
    parser.set_optional<prkl::integer>("p", "epochs", 10, "Number of epochs");
    parser.set_required<std::string>("c", "config", "Layer configuration, e.g. 768,64,32,10");
    parser.run_and_exit_if_error();

    std::string config = parser.get<std::string>("c");
    std::vector<prkl::integer> layer_sizes;
    std::string nbuff;
    for(char s : config)
    {
        if(std::isdigit(s))
        {
            nbuff.push_back(s);
        }
        else 
        {
            if(!nbuff.empty())
            {
                prkl::integer new_size = 0;
                try
                {
                    new_size = std::stoll(nbuff);
                }
                catch(...)
                {
                    std::cerr << "invalid config parameter" << std::endl;
                    return 1;
                }
                layer_sizes.push_back(new_size);
                nbuff.clear();
            }
        }
    }
    if(!nbuff.empty())
    {
        prkl::integer new_size = 0;
        try
        {
            new_size = std::stoll(nbuff);
        }
        catch(...)
        {
            std::cerr << "invalid config parameter" << std::endl;
            return 1;
        }
        layer_sizes.push_back(new_size);
        nbuff.clear();
    }

    if(layer_sizes.size() < 2)
    {
        std::cerr << "At least 2 layer sizes are required" << std::endl;
        return 1;
    }

    std::string training_set_path = parser.get<std::string>("t");
    std::string evaluation_set_path = parser.get<std::string>("e");
    bool do_evaluation = !evaluation_set_path.empty();
    
    prkl::settings.ease_alpha = parser.get<prkl::real>("a");
    prkl::settings.ease = parser.get<bool>("s");
    prkl::settings.base_rate = parser.get<prkl::real>("b");
    prkl::settings.min_rate = parser.get<prkl::real>("m");
    prkl::settings.loss_edge = parser.get<prkl::real>("l");
    prkl::settings.grad_limit = parser.get<prkl::real>("g");

    std::string output_path = parser.get<std::string>("o");
    bool do_output = !output_path.empty();

    prkl::ann_set training_set(training_set_path.c_str());
    prkl::ann_set evaluation_set;
    if(do_evaluation)
    {
        evaluation_set = prkl::ann_set(evaluation_set_path.c_str());
        std::cout << "Evaluation enabled" << std::endl;
    }
    
    prkl::integer num_epochs = parser.get<prkl::integer>("p");
    if(num_epochs < 1)
    {
        std::cerr << "At least 1 epoch is required (but many more are recommended)" << std::endl;
        return 1;
    }

    std::cout << " --- Configuration ---" << std::endl;
    std::cout << "Training set: " << training_set_path << std::endl;
    std::cout << "Evaluation set: " << evaluation_set_path << std::endl;
    std::cout << "Output model: " << output_path << std::endl;
    std::cout << "Layer configuration: " << config << std::endl;
    std::cout << "Num. epochs: " << num_epochs << std::endl;
    std::cout << "Gradient limit: " << prkl::settings.grad_limit << std::endl;
    std::cout << "ALR loss edge: " <<  prkl::settings.loss_edge << std::endl;
    std::cout << "ALR base rate: " <<  prkl::settings.base_rate << std::endl;
    std::cout << "ALR minimum rate: " <<  prkl::settings.min_rate << std::endl;
    std::cout << "ALR ease: " <<  prkl::settings.ease << std::endl;
    std::cout << "ALR ease alpha: " <<  prkl::settings.ease_alpha << std::endl;
    std::cout << " ---------------------" << std::endl;


    prkl::ann_model model;

    for(prkl::integer i = 0; i < layer_sizes.size(); i++)
    {
        model.add_dense_layer(layer_sizes[i]);
        if(i == layer_sizes.size() - 1)
        {
            std::cout << "output layer size: " << layer_sizes[i] << std::endl;
        }
        else if (i == 0)
        {
            std::cout << "input layer size: " << layer_sizes[i] << std::endl;
        }
        else 
        {
            std::cout << "hidden layer size " << i << ": " << layer_sizes[i] << std::endl;
        }
    }

    std::cout << " --- Training model --- " << std::endl;
    if(!model.train(training_set, num_epochs))
    {
        std::cerr << "training failed" << std::endl;
        return 1;
    }

    if(do_evaluation)
    {
        std::cout << " --- Evaluating model --- " << std::endl;
        prkl::ann_layer_base *input_layer = model.input();
        prkl::ann_layer_base *output_layer = model.output();
        prkl::integer num_miss = 0;
        prkl::integer num_pairs = evaluation_set.pairs.size();

        for(prkl::integer e = 0; e <  num_pairs; e++)
        {
            prkl::ann_setpair &eval_pair = evaluation_set.pairs[e];

            for(prkl::integer i = 0; i < input_layer->num_activations(); i++)
            {
                input_layer->set_activation(i, eval_pair.input[i]);
            }

            if(!model.forward_propagate())
            {
                std::cerr << "forward propagation failed: evaluation failed!" << std::endl;
                return 1;
            }

            prkl::integer max_index_output = output_layer->max_activation_index();
            prkl::integer max_index_expected = eval_pair.max_output_index();

            if(max_index_output != max_index_expected)
            {
                num_miss++;
            }
        }

        std::cout << "Evaluated " << num_pairs << " pairs, with " << num_miss << " misses. Success rate: " << (100.0 * (1.0 - (prkl::real(num_miss) / prkl::real(num_pairs)))) << "%" << std::endl;
    }

    if(do_output)
    {
        std::cout << "Writing model: " << output_path << std::endl;
        model.write_file(output_path.c_str());
    }

    return 0;
}