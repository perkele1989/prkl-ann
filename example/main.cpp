
#include "model.hpp"

#include <iostream>

int32_t main(int32_t argc, char **argv)
{
    prkl::ann_set training_set("C:/dev/prkl-ann/data/mnnist-digits-training.prklset");
    prkl::ann_set evaluation_set("C:/dev/prkl-ann/data/mnnist-digits-evaluation.prklset");

    prkl::ann_model model;
    model.add_dense_layer(784);
    model.add_dense_layer(64);
    model.add_dense_layer(32);
    model.add_dense_layer(10);

    model.train(training_set, 50);
    model.evaluate(evaluation_set);
    return 0;
}