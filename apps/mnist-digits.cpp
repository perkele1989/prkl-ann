
#include "model.hpp"
#include "cmdparser.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
#include <iostream>

int32_t main(int32_t argc, char **argv)
{
    prkl::ann_settings &settings = prkl::settings();
    settings.loss_edge = 0.1;
    settings.base_rate = 0.01;

    std::cout << " --- Building model ---" << std::endl;
    prkl::ann_model model;
    model.evaluation_type = prkl::ann_evaluation_type::multiclass_classification;

    prkl::ann_layer_base *input_layer = model.add_dense_layer(784);

    prkl::ann_layer_base *hidden1 = model.add_dense_layer(64);
    hidden1->activation_func = prkl::ann_activation::leaky_relu;

    prkl::ann_layer_base *hidden2 = model.add_dense_layer(32);
    hidden2->activation_func = prkl::ann_activation::leaky_relu;

    prkl::ann_layer_base *output_layer = model.add_dense_layer(10);
    output_layer->activation_func = prkl::ann_activation::linear;

    std::cout << " --- Loading training set ---" << std::endl;
    prkl::ann_set training_set("C:/dev/prkl-ann/data/mnnist-digits-training.prklset");

    std::cout << " --- Loading evaluation set ---" << std::endl;
    prkl::ann_set evaluation_set("C:/dev/prkl-ann/data/mnnist-digits-evaluation.prklset");

    std::cout << " --- Training model ---" << std::endl;
    model.train(training_set, 20);

    std::cout << " --- Running inference ---" << std::endl;
    if(!model.forward_propagate())
    {
        std::cerr << "forward propagation failed" << std::endl;
        return 1;
    }

    model.evaluate(evaluation_set);

    return 0;
}