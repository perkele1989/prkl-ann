
#include "model.hpp"
#include "cmdparser.hpp"
#include <iostream>

int32_t main(int32_t argc, char **argv)
{
    cli::Parser parser(argc, argv);
    parser.set_required<std::string>("t", "training-set", "Path to training set (.prklset file)");
    parser.set_optional<std::string>("e", "evaluation-set", "", "Path to evaluation set (.prklset file)");
    parser.set_optional<prkl::real>("a", "alr-ease-alpha", prkl::settings().ease_alpha, "Adaptive Learning Rate: Ease alpha");
    parser.set_optional<bool>("s", "alr-ease", prkl::settings().ease, "Adaptive Learning Rate: Ease");
    parser.set_optional<bool>("z", "alr", prkl::settings().alr, "Adaptive Learning Rate Enabled");
    parser.set_optional<prkl::real>("b", "learning-rate", prkl::settings().base_rate, "Learning rate (for ALR, this specifies base rate)");
    parser.set_optional<prkl::real>("m", "alr-min-rate", prkl::settings().min_rate, "Adaptive Learning Rate: Minimum rate");
    parser.set_optional<prkl::real>("l", "alr-loss-edge", prkl::settings().loss_edge, "Adaptive Learning Rate: Loss edge");
    parser.set_optional<prkl::real>("g", "grad-limit", prkl::settings().grad_limit, "Maximum gradient amplitude");
    parser.set_optional<std::string>("o", "output", "", "Path to output file (.prklmodel file)");
    parser.set_optional<prkl::integer>("p", "epochs", 10, "Number of epochs");
    parser.set_required<std::string>("c", "config", "Path to model config (.json file)");
    parser.run_and_exit_if_error();

    std::string config_path = parser.get<std::string>("c");
    std::ifstream config_file(config_path);
    if(!config_file)
    {
        std::cerr << "Failed to read model configuration: " << config_path << std::endl;
        return 1;
    }
    nlohmann::json config = nlohmann::json::parse(config_file);

    std::string training_set_path = parser.get<std::string>("t");
    std::string evaluation_set_path = parser.get<std::string>("e");
    bool do_evaluation = !evaluation_set_path.empty();
    
    prkl::settings().ease_alpha = parser.get<prkl::real>("a");
    prkl::settings().ease = parser.get<bool>("s");
    prkl::settings().alr = parser.get<bool>("z");
    prkl::settings().base_rate = parser.get<prkl::real>("b");
    prkl::settings().min_rate = parser.get<prkl::real>("m");
    prkl::settings().loss_edge = parser.get<prkl::real>("l");
    prkl::settings().grad_limit = parser.get<prkl::real>("g");

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
    std::cout << "Layer configuration: " << config_path << std::endl;
    std::cout << "Training set: " << training_set_path << std::endl;
    std::cout << "Evaluation set: " << evaluation_set_path << std::endl;
    std::cout << "Output model: " << output_path << std::endl;
    std::cout << "Num. epochs: " << num_epochs << std::endl;
    std::cout << "Gradient limit: " << prkl::settings().grad_limit << std::endl;
    std::cout << "ALR enabled:" << prkl::settings().alr << std::endl;
    std::cout << "ALR loss edge: " <<  prkl::settings().loss_edge << std::endl;
    std::cout << "ALR base rate: " <<  prkl::settings().base_rate << std::endl;
    std::cout << "ALR minimum rate: " <<  prkl::settings().min_rate << std::endl;
    std::cout << "ALR ease: " <<  prkl::settings().ease << std::endl;
    std::cout << "ALR ease alpha: " <<  prkl::settings().ease_alpha << std::endl;
    std::cout << " ---------------------" << std::endl;


    prkl::ann_model model(config);


    std::cout << " --- Training model --- " << std::endl;
    if(!model.train(training_set, num_epochs, do_evaluation ? &evaluation_set : nullptr))
    {
        std::cerr << "training failed" << std::endl;
        return 1;
    }

    if(do_evaluation)
    {
        std::cout << " --- Evaluating model --- " << std::endl;
        model.evaluate(evaluation_set);
    }

    if(do_output)
    {
        std::cout << "Writing model: " << output_path << std::endl;
        model.write_file(output_path.c_str());
    }

    return 0;
}