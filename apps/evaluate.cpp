
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

    if(model.input()->num_activations() != evaluation_set.num_inputs || model.output()->num_activations() != evaluation_set.num_outputs)
    {
        std::cerr << "Incompatible evaluation set" << std::endl;
        return 1;
    }

    std::cout << " --- Evaluating model --- " << std::endl;
    model.evaluate(evaluation_set);
    return 0;
}