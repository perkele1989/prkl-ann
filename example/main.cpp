
#include "model.hpp"

#include <iostream>

int32_t main(int32_t argc, char **argv)
{
    prkl::ann_set training_set("C:/dev/prkl-ann/data/mnnist-digits-training.prklset");
    prkl::ann_set evaluation_set("C:/dev/prkl-ann/data/mnnist-digits-evaluation.prklset");

    prkl::ann_model model(784);
    model.add_layer(64);
    model.add_layer(32);
    model.add_layer(10);

    model.train(training_set, 50);

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

        model.forward_propagate();

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