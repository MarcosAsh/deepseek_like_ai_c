#include "transformer.cpp"
#include <cassert>
#include <iostream>

int main() {
    Transformer t;
    t.forward();
    std::cout << "Transformer test passed!" << std::endl;
    return 0;
}
