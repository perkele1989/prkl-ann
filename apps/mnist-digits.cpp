
#include "model.hpp"
#include "cmdparser.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
#include <iostream>

int32_t main(int32_t argc, char **argv)
{
    cli::Parser parser(argc, argv);
    parser.set_required<std::string>("m", "model", "Path to MNIST digits model, needs to have 784 inputs and 10 outputs (.prklmodel file)");
    parser.set_required<std::string>("i", "input", "Path to input image file, will be scaled to 28x28 for inference");
    parser.run_and_exit_if_error();


    std::string model_path = parser.get<std::string>("m");
    std::string input_path = parser.get<std::string>("i");
    
    std::cout << " --- Loading model ---" << std::endl;
    prkl::ann_model model(model_path.c_str());
    prkl::ann_layer_base *input_layer = model.input();
    prkl::ann_layer_base *output_layer = model.output();
    if(input_layer->num_activations() != 784)
    {
        std::cerr << "Model input size is not 784, incompatible with MNIST digits set" << std::endl;
        return 1;
    }

    if(output_layer->num_activations() != 10)
    {
        std::cerr << "Model output size is not 10, incompatible with MNIST digits set" << std::endl;
        return 1;
    }

    std::cout << " --- Loading image ---" << std::endl;
    int x, y, cmp;
    float *data = stbi_loadf(input_path.c_str(), &x, &y, &cmp, 1);
    if(!data)
    {
        std::cerr << "failed to load input image: " << input_path << std::endl;
        return 1;
    }

    float *scaled_data = stbir_resize_float_linear(data, x, y, 0, NULL, 28, 28, 0, (stbir_pixel_layout)1);
    if(!scaled_data)
    {
        std::cerr << "failed to resize input image" << std::endl;
        return 1;
    }
    free(data);

    for(prkl::integer i = 0; i < 784; i++)
    {
        input_layer->set_activation(i, scaled_data[i]);
    }

    free(scaled_data);


    std::cout << " --- Running inference ---" << std::endl;
    if(!model.forward_propagate())
    {
        std::cerr << "forward propagation failed" << std::endl;
        return 1;
    }

    prkl::integer max_index = output_layer->max_activation_index();

    prkl::real certainty = std::clamp(output_layer->get_activation(max_index) * 100.0f, 0.0f, 100.0f);
    std::cout << "The image is identified as the digit " << max_index << " with a certainty of " << certainty << "%" << std::endl; 


    return 0;
}