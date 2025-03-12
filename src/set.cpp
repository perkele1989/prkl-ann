
#include "set.hpp"

#include <iostream>


prkl::ann_set::ann_set(integer input_size, integer output_size)
    : num_inputs(input_size)
    , num_outputs(output_size)
{

}

prkl::ann_set::ann_set(char const* path)
{
    std::ifstream file(path, std::ios::binary);
    if(!file)
    {
        std::cerr << "failed to open file for reading: " << path << std::endl;
        return;
    }

    num_inputs = read_uint64_be(file);
    num_outputs = read_uint64_be(file);

    uint64_t num_pairs = read_uint64_be(file);
    pairs.resize(num_pairs);

    for(uint64_t i = 0; i < num_pairs; i++)
    {
        ann_setpair &new_pair = pairs[i];
        new_pair.input.resize(num_inputs);
        for(uint64_t j = 0; j < num_inputs; j++)
        {
            new_pair.input[j] = read_float_be(file);
        }

        new_pair.output.resize(num_outputs);
        for(uint64_t j = 0; j < num_outputs; j++)
        {
            new_pair.output[j] = read_float_be(file);
        }
    }

    file.close();
}


prkl::integer prkl::ann_setpair::min_input_index()
{
    real c_min = std::numeric_limits<prkl::real>::infinity();
    integer c_index = 0;
    for(integer i = 0; i < input.size(); i++)
    {
        if(input[i] < c_min)
        {
            c_min = input[i];
            c_index = i;
        }
    }

    return c_index;
}

prkl::integer prkl::ann_setpair::max_input_index()
{
    real c_max = 0.0;
    integer c_index = 0;
    for(integer i = 0; i < input.size(); i++)
    {
        if(input[i] > c_max)
        {
            c_max = input[i];
            c_index = i;
        }
    }

    return c_index;
}

prkl::integer prkl::ann_setpair::min_output_index()
{
    real c_min = std::numeric_limits<prkl::real>::infinity();
    integer c_index = 0;
    for(integer i = 0; i < output.size(); i++)
    {
        if(output[i] < c_min)
        {
            c_min = output[i];
            c_index = i;
        }
    }

    return c_index;
}
prkl::integer prkl::ann_setpair::max_output_index()
{
    real c_max = 0.0;
    integer c_index = 0;
    for(integer i = 0; i < output.size(); i++)
    {
        if(output[i] > c_max)
        {
            c_max = output[i];
            c_index = i;
        }
    }

    return c_index;
}