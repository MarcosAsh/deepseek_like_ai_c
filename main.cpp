#include "transformer.hpp"
#include <iostream>

int main() {
    Transformer transformer;
    transformer.forward();
    std::cout << "Transformer ran successfully!" << std::endl;
    return 0;
}
