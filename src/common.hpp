
#pragma once 

#include <cmath>
#include <cstdint>
#include <random>
#include <cassert>
#include <numbers>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <vector>
#include <iostream>
#include "json.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "Ws2_32.lib")
#else
    #include <arpa/inet.h>
#endif

#define ntohll(x) ((uint64_t(ntohl(x & 0xFFFFFFFF)) << 32) | ntohl(x >> 32))
#define htonll(x) ((uint64_t(htonl(x & 0xFFFFFFFF)) << 32) | htonl(x >> 32))



namespace prkl 
{
    using real = float;
    using natural = int64_t;
    using integer = uint64_t;

    // grad_limit = 0.75
    //constexpr real grad_limit = 0.75;
    //constexpr real min_rate = 0.00001;// 0.0979244 96.34
    //constexpr real min_rate = 0.0000025; // 0.0841101 96.04 
    //constexpr real min_rate = 0.000001; // 0.0846624 96.4
    //constexpr real min_rate = 0.00000025; // 0.0836112 96.36
    //constexpr real min_rate = 0.00000005; //  0.0790566  96.45
    
    // grad_limit = 0.7
    //constexpr real grad_limit = 0.7;
    //constexpr real min_rate = 0.00000005; // 0.0785027 96.36%

    // grad_limit = 0.6
    //constexpr real min_rate = 0.00000005; // 0.0968233 95.58
    //constexpr real min_rate = 0.00000025; // 0.0890659   95.72 SLOW DIVERGENCE! 

    enum class ann_activation : integer 
    {
        swish = 0,
        tanh,
        relu,
        leaky_relu,
        sigmoid,
        linear
    };

    struct ann_settings 
    {
        real base_rate {(real)0.01};
        real loss_edge {(real)0.25};
        real min_rate {(real)0.00000005};
        real grad_limit {(real)0.75};
        bool alr {true};
    
        bool ease{false};
        real ease_alpha {(real)1.0};
    
        bool early_exit{true};
        real early_exit_treshold{(real)0.2};
    };

    ann_settings &settings(); 

    struct ann_layer_base;

    inline real ease_in_sine(real x)
    {
        return  (real)1.0 - (real)std::cos((x * std::numbers::pi) / 2.0);
      
    }

    inline real adaptive_learning_rate(real loss) 
    {    
        real rate = settings().base_rate;
        if(!settings().alr)
            return rate; 

        if(loss < settings().loss_edge)
        {
            real rate_alpha = std::clamp(loss/settings().loss_edge, (real)0.0, (real)1.0);
            if(settings().ease)
            {
                real rate_beta = ease_in_sine(rate_alpha);
                rate_alpha = std::lerp(rate_alpha, rate_beta, settings().ease_alpha);
            }
            rate = std::lerp(settings().min_rate, settings().base_rate, rate_alpha);
        }
        
        return std::max(settings().min_rate, rate);
    }

    inline real swish(real x) 
    {
        real safe_x = std::max(-10.0f, std::min(10.0f, x));
        real denominator = (real)1.0 + std::exp(-safe_x);
        real returner = safe_x / denominator;

        assert(!std::isnan(returner) && "NaN detected in swish()");
        return returner;
    }

    inline real swish_derivative(real x) 
    {
        real safe_x = std::max(-10.0f, std::min(10.0f, x));
        real sigmoid_x = (real)1.0 / ((real)1.0 + std::exp(-safe_x));
        real returner = sigmoid_x + safe_x * sigmoid_x * ((real)1.0 - sigmoid_x);

        // clip the derivative to prevent exploding gradients
        returner = std::max(-settings().grad_limit, std::min(settings().grad_limit, returner));

        assert(!std::isnan(returner) && "NaN detected in swish_derivative()");
        return returner;
    }

    inline prkl::real tanh(prkl::real x) 
    {
        prkl::real ex = std::exp(x);
        prkl::real enx = std::exp(-x);
        return (ex - enx) / (ex + enx);
    }
    
    inline prkl::real tanh_derivative(prkl::real x) 
    {
        prkl::real tanh_x = tanh(x);
        return 1.0f - (tanh_x * tanh_x);
    }

    inline prkl::real leaky_relu(prkl::real x, prkl::real alpha) 
    {
        return (x > 0) ? x : alpha * x;
    }

    inline prkl::real leaky_relu_derivative(prkl::real x, prkl::real alpha) 
    {
        return (x > 0) ? (prkl::real)1 : alpha;
    }

    inline prkl::real relu(prkl::real x) 
    {
        return std::max((prkl::real)0, x);
    }

    inline prkl::real relu_derivative(prkl::real x) 
    {
        return (x > 0) ? (prkl::real)1 : (prkl::real)0;
    }

    inline real sigmoid(real x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }
    
    inline real sigmoid_derivative(real x)
    {
        real s = sigmoid(x);
        return s * (1.0f - s);
    }

    inline real linear(real x)
    {
        return x;
    }

    inline real linear_derivative(real x)
    {
        return 1;
    }



    std::mt19937& random_device();


    inline uint64_t read_uint64_be(std::ifstream &file)
    {
        uint64_t val;
        file.read(reinterpret_cast<char*>(&val), sizeof(val));
    
        val = ntohll(val);
        return val;
    }
    
    inline float read_float_be(std::ifstream &file)
    {
        uint32_t val;
        file.read(reinterpret_cast<char*>(&val), sizeof(val));
    
        val = ntohl(val);
    
        float result;
        std::memcpy(&result, &val, sizeof(result));
        return result;
    }

    inline void write_uint64_be(std::ofstream &file, uint64_t val)
    {
        val = htonll(val);
        file.write(reinterpret_cast<const char*>(&val), sizeof(val));
    }
    
    inline void write_float_be(std::ofstream &file, float val)
    {
        uint32_t int_val;
        std::memcpy(&int_val, &val, sizeof(val));
        int_val = htonl(int_val);
        file.write(reinterpret_cast<const char*>(&int_val), sizeof(int_val));
    }

    
    prkl::real activation(prkl::ann_layer_base const* layer, prkl::real x);

    prkl::real activation_derivative(prkl::ann_layer_base const* layer, prkl::real x);

    enum class ann_model_version : integer 
    {
        initial = 0,
        layer_parameters,
        count,
        latest = count - 1 
    };

    enum class ann_layer_type : integer
    {
        undefined = 0,
        dense = 1,
        convolutional = 2,
        pooling = 3,
    };

    /** Loss function for regression-based models */
    enum class ann_loss_function : integer 
    {
        mean_squared_error,   // MSE: Penalizes larger errors more
        mean_absolute_error   // MAE: Penalizes all errors equally
    };

    enum class ann_evaluation_type : integer 
    {
        /** Loss function: MSE or MAE. Recommended: Linear Output Activation  */
        regression = 0,
        /** Loss function: CE. Softmax will be applied. Recommended: Linear Output Activation  */
        multiclass_classification,
        /** Loss function: BCE. Recommended: Sigmoid Output Activation  */
        binary_classification,
        /** Loss function: BCE per neuron. Recommended: Sigmoid Output Activation  */
        multilabel_classification
    };
}