#include "transformer.hpp"
#include <iostream>

Transformer::Transformer() {}

void Transformer::forward() {
    MLA mla(8, 8, 2, 4);
    Tensor input(8, 1);
    input.fill(1.0f);
    Tensor out = mla.forward(input);
}
