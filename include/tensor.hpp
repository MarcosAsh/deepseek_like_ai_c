#pragma once
#include <vector>

class Tensor {
public:
    std::vector<float> data;
    int size;

    Tensor(int size);
    void fill(float value);
};
