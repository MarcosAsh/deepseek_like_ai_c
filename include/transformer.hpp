#pragma once
#include "tensor.hpp"
#include "layers/linear.hpp"
#include "layers/norm.hpp"
#include "layers/mla.hpp"
#include "layers/moe.hpp"

class Transformer {
public:
    Transformer();
    void forward();
};
