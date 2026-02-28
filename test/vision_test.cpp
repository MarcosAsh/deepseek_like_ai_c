#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "tensor.hpp"
#include "autodiff.hpp"
#include "layers/ad_conv2d.hpp"
#include "layers/ad_pool2d.hpp"
#include "layers/ad_batchnorm2d.hpp"
#include "layers/ad_flatten.hpp"
#include "layers/ad_linear.hpp"

static void test_ndim_tensor() {
    std::cout << "Testing N-dim tensor operations...\n";

    // 4D tensor [1, 3, 4, 4]
    Tensor t({1, 3, 4, 4});
    assert(t.ndim() == 4);
    assert(t.numel() == 48);
    assert(t.rows == -1);
    assert(t.shape[0] == 1);
    assert(t.shape[1] == 3);
    assert(t.shape[2] == 4);
    assert(t.shape[3] == 4);

    // 2D backward compat
    Tensor t2(3, 5);
    assert(t2.ndim() == 2);
    assert(t2.rows == 3);
    assert(t2.cols == 5);
    assert(t2.shape[0] == 3);
    assert(t2.shape[1] == 5);

    // Reshape
    Tensor r = t.reshape({1, 3, 16});
    assert(r.ndim() == 3);
    assert(r.shape[0] == 1);
    assert(r.shape[1] == 3);
    assert(r.shape[2] == 16);

    // Reshape with -1
    Tensor r2 = t.reshape({1, -1});
    assert(r2.ndim() == 2);
    assert(r2.shape[1] == 48);

    // Flatten
    Tensor f = t.flatten(1);
    assert(f.ndim() == 2);
    assert(f.shape[0] == 1);
    assert(f.shape[1] == 48);

    // Squeeze/unsqueeze
    Tensor sq = Tensor({1, 3, 1, 4}).squeeze();
    assert(sq.ndim() == 2);
    assert(sq.shape[0] == 3);
    assert(sq.shape[1] == 4);

    Tensor usq = Tensor({3, 4}).unsqueeze(0);
    assert(usq.ndim() == 3);
    assert(usq.shape[0] == 1);

    // Permute
    Tensor p = Tensor({2, 3, 4}).permute({2, 0, 1});
    assert(p.shape[0] == 4);
    assert(p.shape[1] == 2);
    assert(p.shape[2] == 3);

    std::cout << "  N-dim tensor ops: PASSED\n";
}

static void test_conv2d_forward() {
    std::cout << "Testing Conv2D forward...\n";
    clear_parameters();

    ADConv2D conv(3, 8, 3, 1, 1); // 3->8 channels, 3x3 kernel, stride 1, padding 1

    // Input: [1, 3, 8, 8]
    auto input = std::make_shared<ADTensor>(std::vector<int>{1, 3, 8, 8});
    for (size_t i = 0; i < input->val.data.size(); ++i) {
        input->val.data[i] = static_cast<float>(i % 7) * 0.1f;
    }

    auto output = conv.forward(input);

    // Should be [1, 8, 8, 8] (same padding)
    assert(output->val.shape.size() == 4);
    assert(output->val.shape[0] == 1);
    assert(output->val.shape[1] == 8);
    assert(output->val.shape[2] == 8);
    assert(output->val.shape[3] == 8);

    std::cout << "  Conv2D forward shape: PASSED\n";
}

static void test_conv2d_backward() {
    std::cout << "Testing Conv2D backward...\n";
    clear_parameters();

    ADConv2D conv(1, 2, 3, 1, 0);

    auto input = std::make_shared<ADTensor>(std::vector<int>{1, 1, 4, 4});
    for (size_t i = 0; i < input->val.data.size(); ++i) {
        input->val.data[i] = static_cast<float>(i) * 0.1f;
    }

    auto output = conv.forward(input);
    // Output: [1, 2, 2, 2]
    assert(output->val.shape[2] == 2);
    assert(output->val.shape[3] == 2);

    // Sum and backward
    auto loss = sum(reshape_ad(output, {-1, 1}));
    loss->backward();

    // Check gradients exist and are non-zero
    bool has_input_grad = false;
    for (float g : input->grad.data) {
        if (std::abs(g) > 1e-8f) { has_input_grad = true; break; }
    }
    assert(has_input_grad);

    bool has_weight_grad = false;
    for (float g : conv.weight->grad.data) {
        if (std::abs(g) > 1e-8f) { has_weight_grad = true; break; }
    }
    assert(has_weight_grad);

    std::cout << "  Conv2D backward: PASSED\n";
}

static void test_maxpool2d() {
    std::cout << "Testing MaxPool2D...\n";

    ADMaxPool2D pool(2, 2, 0);

    auto input = std::make_shared<ADTensor>(std::vector<int>{1, 1, 4, 4});
    for (int i = 0; i < 16; ++i) input->val.data[i] = static_cast<float>(i);

    auto output = pool.forward(input);
    assert(output->val.shape[0] == 1);
    assert(output->val.shape[1] == 1);
    assert(output->val.shape[2] == 2);
    assert(output->val.shape[3] == 2);

    // Max values should be: [5, 7, 13, 15] (2x2 patches)
    assert(std::abs(output->val.data[0] - 5.0f) < 1e-6f);
    assert(std::abs(output->val.data[1] - 7.0f) < 1e-6f);
    assert(std::abs(output->val.data[2] - 13.0f) < 1e-6f);
    assert(std::abs(output->val.data[3] - 15.0f) < 1e-6f);

    std::cout << "  MaxPool2D: PASSED\n";
}

static void test_avgpool2d() {
    std::cout << "Testing AvgPool2D...\n";

    ADAvgPool2D pool(2, 2, 0);

    auto input = std::make_shared<ADTensor>(std::vector<int>{1, 1, 4, 4});
    for (int i = 0; i < 16; ++i) input->val.data[i] = static_cast<float>(i);

    auto output = pool.forward(input);
    assert(output->val.shape[2] == 2);
    assert(output->val.shape[3] == 2);

    // Avg of [0,1,4,5] = 2.5
    assert(std::abs(output->val.data[0] - 2.5f) < 1e-6f);

    std::cout << "  AvgPool2D: PASSED\n";
}

static void test_batchnorm2d() {
    std::cout << "Testing BatchNorm2D...\n";
    clear_parameters();

    ADBatchNorm2D bn(3);

    auto input = std::make_shared<ADTensor>(std::vector<int>{2, 3, 4, 4});
    for (size_t i = 0; i < input->val.data.size(); ++i) {
        input->val.data[i] = static_cast<float>(i % 13) * 0.1f;
    }

    auto output = bn.forward(input);
    assert(output->val.shape == input->val.shape);

    // After batchnorm, each channel should be approximately zero-mean
    for (int c = 0; c < 3; ++c) {
        float sum = 0.0f;
        int count = 0;
        for (int b = 0; b < 2; ++b) {
            for (int h = 0; h < 4; ++h) {
                for (int w = 0; w < 4; ++w) {
                    sum += output->val.data[((b * 3 + c) * 4 + h) * 4 + w];
                    count++;
                }
            }
        }
        float mean = sum / count;
        assert(std::abs(mean) < 0.01f); // approximately zero mean
    }

    std::cout << "  BatchNorm2D: PASSED\n";
}

static void test_flatten() {
    std::cout << "Testing Flatten...\n";

    ADFlatten flat(1, -1);

    auto input = std::make_shared<ADTensor>(std::vector<int>{2, 3, 4, 4});
    auto output = flat.forward(input);

    assert(output->val.shape.size() == 2);
    assert(output->val.shape[0] == 2);
    assert(output->val.shape[1] == 48); // 3*4*4

    std::cout << "  Flatten: PASSED\n";
}

static void test_cnn_pipeline() {
    std::cout << "Testing full CNN pipeline: Conv2D -> ReLU -> MaxPool2D -> Flatten -> Linear...\n";
    clear_parameters();

    // Build a simple CNN
    ADConv2D conv(3, 8, 3, 1, 1);       // [1,3,8,8] -> [1,8,8,8]
    ADMaxPool2D pool(2, 2);               // [1,8,8,8] -> [1,8,4,4]
    ADFlatten flat(1, -1);                // [1,8,4,4] -> [1,128]
    ADLinear linear(128, 10);             // [1,128] -> [1,10]

    // Input: [1, 3, 8, 8]
    auto input = std::make_shared<ADTensor>(std::vector<int>{1, 3, 8, 8});
    for (size_t i = 0; i < input->val.data.size(); ++i) {
        input->val.data[i] = static_cast<float>(i % 11) * 0.05f;
    }

    // Forward pass
    auto x = conv.forward(input);
    x = relu_ad(x);
    x = pool.forward(x);
    x = flat.forward(x);
    x = linear.forward(x);

    assert(x->val.shape.size() == 2);
    assert(x->val.shape[0] == 1);
    assert(x->val.shape[1] == 10);

    // Backward pass
    auto loss = sum(x);
    loss->backward();

    // Verify gradients flow back to input
    bool has_grad = false;
    for (float g : input->grad.data) {
        if (std::abs(g) > 1e-10f) { has_grad = true; break; }
    }
    assert(has_grad);

    std::cout << "  CNN pipeline: PASSED\n";
}

static void test_relu_sigmoid_ad() {
    std::cout << "Testing ReLU and Sigmoid AD...\n";

    auto input = std::make_shared<ADTensor>(std::vector<int>{2, 3});
    input->val.data = {-1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 0.5f};

    auto relu_out = relu_ad(input);
    // Check: negative values become 0
    assert(relu_out->val.data[0] == 0.0f);
    assert(relu_out->val.data[2] == 1.0f);
    assert(relu_out->val.data[3] == 2.0f);

    auto input2 = std::make_shared<ADTensor>(std::vector<int>{1, 1});
    input2->val.data[0] = 0.0f;
    auto sig_out = sigmoid_ad(input2);
    assert(std::abs(sig_out->val.data[0] - 0.5f) < 1e-6f);

    std::cout << "  ReLU/Sigmoid AD: PASSED\n";
}

int main() {
    std::cout << "=== Vision Test Suite ===\n\n";

    test_ndim_tensor();
    test_relu_sigmoid_ad();
    test_conv2d_forward();
    test_conv2d_backward();
    test_maxpool2d();
    test_avgpool2d();
    test_batchnorm2d();
    test_flatten();
    test_cnn_pipeline();

    std::cout << "\nAll vision tests PASSED!\n";
    return 0;
}
