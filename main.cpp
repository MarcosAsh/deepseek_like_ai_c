#include "transformer.hpp"
#include <iostream>

int main() {
    const int input_dim = 64;
    const int hidden_dim = 64;
    const int n_heads = 4;
    const int compress_dim = 16;
    const int num_layers = 3;
    
    // Create transformer
    Transformer transformer(num_layers, input_dim, hidden_dim, n_heads, compress_dim);
    
    // Create test input
    Tensor input(input_dim, 1);
    input.fill(1.0f);
    
    // Forward pass
    Tensor output = transformer.forward(input);
    
    std::cout << "Transformer ran successfully!" << std::endl;
    return 0;
}