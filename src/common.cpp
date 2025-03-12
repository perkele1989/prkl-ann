#include "common.hpp"

std::mt19937& prkl::random_device()
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return gen;
}