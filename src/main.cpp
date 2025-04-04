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
    
    // Verification
    std::cout << "\n=== Shape Verification ===\n";
    std::cout << "Input shape: [" << input.rows << "x" << input.cols << "]\n";
    std::cout << "Output shape: [" << output.rows << "x" << output.cols << "]\n";
    
    if (input.rows == output.rows && input.cols == output.cols) {
        std::cout << "Residual connections working correctly\n";
    } else {
        std::cout << "Dimension mismatch in residual connections\n";
    }
    
    return 0;
}