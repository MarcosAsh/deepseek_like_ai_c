#include "layers/moe.hpp"
#include "layers/ad_moe.hpp"
#include "autodiff.hpp"
#include "tensor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

static bool almost_eq(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    // ===================== MoE (Non-AD) Tests =====================

    // Test 1: MoE output shape
    {
        int input_dim = 8, expert_dim = 16, num_experts = 4, top_k = 2;
        MoE moe(input_dim, expert_dim, num_experts, top_k);
        Tensor input(input_dim, 5);  // seq_len=5
        for (auto& v : input.data) v = 0.3f;
        float aux_loss = 0.0f;
        Tensor out = moe.forward(input, aux_loss);
        assert(out.rows == input_dim);
        assert(out.cols == 5);
        std::cout << "  [PASS] MoE output shape\n";
    }

    // Test 2: MoE aux_loss >= 0
    {
        int input_dim = 8, expert_dim = 16, num_experts = 4, top_k = 2;
        MoE moe(input_dim, expert_dim, num_experts, top_k);
        Tensor input(input_dim, 4);
        for (auto& v : input.data) v = 0.5f;
        float aux_loss = 0.0f;
        moe.forward(input, aux_loss);
        assert(aux_loss >= 0.0f);
        std::cout << "  [PASS] MoE aux_loss >= 0\n";
    }

    // Test 3: MoE top_k routing - different top_k values
    {
        int input_dim = 4, expert_dim = 8, num_experts = 4;
        for (int top_k : {1, 2, 4}) {
            MoE moe(input_dim, expert_dim, num_experts, top_k);
            Tensor input(input_dim, 3);
            for (auto& v : input.data) v = 0.2f;
            float aux_loss = 0.0f;
            Tensor out = moe.forward(input, aux_loss);
            assert(out.rows == input_dim);
            assert(out.cols == 3);
            // All output values should be finite
            for (auto& v : out.data) {
                assert(std::isfinite(v));
            }
        }
        std::cout << "  [PASS] MoE different top_k configs\n";
    }

    // Test 4: MoE different expert counts
    {
        int input_dim = 4, expert_dim = 8;
        for (int num_experts : {2, 4, 8}) {
            MoE moe(input_dim, expert_dim, num_experts, 2);
            Tensor input(input_dim, 2);
            for (auto& v : input.data) v = 0.1f;
            float aux_loss = 0.0f;
            Tensor out = moe.forward(input, aux_loss);
            assert(out.rows == input_dim);
            assert(out.cols == 2);
        }
        std::cout << "  [PASS] MoE different expert counts\n";
    }

    // ===================== ADMoE Tests =====================

    // Test 5: ADMoE output shape
    {
        clear_parameters();
        int embed = 8, hidden = 16, num_experts = 4, top_k = 2;
        ADMoE admoe(embed, hidden, num_experts, top_k);
        Tensor input_t(embed, 4);
        for (auto& v : input_t.data) v = 0.3f;
        auto input = make_ad(input_t);
        auto result = admoe.forward(input);
        assert(result.output->val.rows == embed);
        assert(result.output->val.cols == 4);
        std::cout << "  [PASS] ADMoE output shape\n";
    }

    // Test 6: ADMoE aux_loss is valid ADTensor
    {
        clear_parameters();
        int embed = 8, hidden = 16, num_experts = 4, top_k = 2;
        ADMoE admoe(embed, hidden, num_experts, top_k);
        Tensor input_t(embed, 3);
        for (auto& v : input_t.data) v = 0.2f;
        auto input = make_ad(input_t);
        auto result = admoe.forward(input);
        assert(result.aux_loss != nullptr);
        assert(result.aux_loss->val.rows == 1);
        assert(result.aux_loss->val.cols == 1);
        assert(result.aux_loss->val.data[0] >= 0.0f);
        std::cout << "  [PASS] ADMoE aux_loss valid ADTensor\n";
    }

    // Test 7: ADMoE backward - finite gradients
    {
        clear_parameters();
        int embed = 4, hidden = 8, num_experts = 2, top_k = 1;
        ADMoE admoe(embed, hidden, num_experts, top_k);
        Tensor input_t(embed, 2);
        for (auto& v : input_t.data) v = 0.5f;
        auto input = make_ad(input_t);
        register_parameter(input);
        auto result = admoe.forward(input);
        auto total = add(sum(result.output), result.aux_loss);
        auto s = sum(total);
        s->backward();
        for (auto& v : input->grad.data) {
            assert(std::isfinite(v));
        }
        std::cout << "  [PASS] ADMoE backward finite grads\n";
    }

    std::cout << "All MoE tests passed." << std::endl;
    return 0;
}
