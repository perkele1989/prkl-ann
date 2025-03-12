
#include "model.hpp"
#include "cmdparser.hpp"
#include <iostream>

int32_t main(int32_t argc, char **argv)
{
    cli::Parser parser(argc, argv);
    parser.set_required<std::string>("m", "model", "Path to model (.prklmodel file)");
    parser.set_required<std::string>("e", "evaluation-set", "", "Path to evaluation set (.prklset file)");
    parser.run_and_exit_if_error();

    std::string evaluation_set_path = parser.get<std::string>("e");

    std::cout << " --- Loading evaluation set --- " << std::endl;
    prkl::ann_set evaluation_set = prkl::ann_set(evaluation_set_path.c_str());
    
    std::string model_path = parser.get<std::string>("m");

    std::cout << " --- Loading model --- " << std::endl;
    prkl::ann_model model(model_path.c_str());

    if(model.input().neurons.size() != evaluation_set.num_inputs || model.output().neurons.size() != evaluation_set.num_outputs)
    {
        std::cerr << "Incompatible evaluation set" << std::endl;
        return 1;
    }

    std::cout << " --- Evaluating model --- " << std::endl;
    prkl::ann_layer &input_layer = model.input();
    prkl::ann_layer &output_layer = model.output();
    prkl::integer num_miss = 0;
    prkl::integer num_pairs = evaluation_set.pairs.size();

    for(prkl::integer e = 0; e <  num_pairs; e++)
    {
        prkl::ann_setpair &eval_pair = evaluation_set.pairs[e];
        for(prkl::integer i = 0; i < input_layer.neurons.size(); i++)
        {
            input_layer.neurons[i].activation = eval_pair.input[i];
        }

        if(!model.forward_propagate())
        {
            std::cerr << "forward propagation failed: evaluation failed!" << std::endl;
            return 1;
        }

        prkl::integer max_index_output = output_layer.max_activation_index();
        prkl::integer max_index_expected = eval_pair.max_output_index();

        if(max_index_output != max_index_expected)
        {
            num_miss++;
        }
    }

    std::cout << "Evaluated " << num_pairs << " pairs, with " << num_miss << " misses. Success rate: " << (100.0 * (1.0 - (prkl::real(num_miss) / prkl::real(num_pairs)))) << "%" << std::endl;

    return 0;
}