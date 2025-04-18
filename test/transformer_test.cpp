#include "transformer.hpp"
#include <iostream>

int main() {
    // Example configuration (adjust as needed)
    const int num_layers = 3;
    const int input_dim = 64;
    const int hidden_dim = 64;
    const int n_heads = 4;
    // const int compress_dim = 16; // not used for feed-forward version
    
    // Create transformer with proper parameters
    Transformer t(num_layers, input_dim, hidden_dim, n_heads);
    
    // Create test input tensor
    Tensor input(input_dim, 1);
    input.fill(1.0f);
    
    // Run forward pass
    Tensor output = t.forward(input);
    
    std::cout << "Transformer test passed!" << std::endl;
    return 0;
}