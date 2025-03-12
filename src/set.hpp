
#pragma once 

#include "common.hpp"

namespace prkl 
{

    struct ann_setpair 
    {
        integer min_input_index();
        integer max_input_index();

        integer min_output_index();
        integer max_output_index();

        std::vector<real> input;
        std::vector<real> output;
    };

    struct ann_set
    {
        ann_set()=default;
        ann_set(integer input_size, integer output_size);
        ann_set(char const* path);

        integer num_inputs{};
        integer num_outputs{};
        std::vector<ann_setpair> pairs;
    };

}