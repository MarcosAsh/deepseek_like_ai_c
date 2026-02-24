#include "layers/rope.hpp"
#include "layers/ad_swiglu.hpp"
#include "layers/ad_rmsnorm.hpp"
#include "lr_scheduler.hpp"
#include "autodiff.hpp"
#include "tensor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

// Helper: check all values finite
static void assert_finite(const Tensor& t, const char* name) {
    for (size_t i = 0; i < t.data.size(); ++i) {
        assert(std::isfinite(t.data[i]) && name);
    }
}

// ========================== RoPE Tests ==========================

void test_rope_basic() {
    RoPE rope(8, 64);
    Tensor x(8, 4);
    for (auto& v : x.data) v = 1.0f;
    Tensor out = rope.apply(x);
    assert(out.rows == 8 && out.cols == 4);
    assert_finite(out, "rope_basic");
    std::cout << "  [PASS] RoPE basic forward\n";
}

void test_rope_identity_at_zero() {
    // At position 0, cos=1 and sin=0, so RoPE should be close to identity
    RoPE rope(4, 64);
    Tensor x(4, 1);
    x.data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor out = rope.apply(x);
    // At pos=0: cos(0)=1, sin(0)=0
    // out[0] = x[0]*1 - x[2]*0 = x[0]
    // out[2] = x[0]*0 + x[2]*1 = x[2]
    assert(std::abs(out.data[0] - 1.0f) < 1e-5f);
    assert(std::abs(out.data[2] - 3.0f) < 1e-5f);
    std::cout << "  [PASS] RoPE identity at position 0\n";
}

void test_rope_ad_gradient() {
    clear_parameters();
    RoPE rope(4, 64);
    Tensor x_t(4, 2);
    for (auto& v : x_t.data) v = 0.5f;
    auto x = make_ad(x_t);
    auto out = rope.apply_ad(x);
    auto s = sum(out);
    s->backward();
    // Gradient should be non-zero and finite
    for (auto& g : x->grad.data) {
        assert(std::isfinite(g));
    }
    std::cout << "  [PASS] RoPE autodiff gradient\n";
}

void test_rope_pos_offset() {
    RoPE rope(4, 64);
    Tensor x(4, 1);
    x.data = {1.0f, 0.0f, 0.0f, 0.0f};
    Tensor out0 = rope.apply(x, 0);
    Tensor out5 = rope.apply(x, 5);
    // Different positions should give different results
    bool different = false;
    for (size_t i = 0; i < out0.data.size(); ++i) {
        if (std::abs(out0.data[i] - out5.data[i]) > 1e-6f) {
            different = true;
            break;
        }
    }
    assert(different);
    std::cout << "  [PASS] RoPE position offset\n";
}

// ========================== SwiGLU Tests ==========================

void test_swiglu_basic() {
    clear_parameters();
    ADSwiGLU swiglu(8, 16);
    Tensor x_t(8, 4);
    for (auto& v : x_t.data) v = 0.1f;
    auto x = make_ad(x_t);
    auto out = swiglu.forward(x);
    assert(out->val.rows == 8);
    assert(out->val.cols == 4);
    assert_finite(out->val, "swiglu_basic");
    std::cout << "  [PASS] SwiGLU basic forward\n";
}

void test_swiglu_gradient() {
    clear_parameters();
    ADSwiGLU swiglu(4, 8);
    Tensor x_t(4, 2);
    for (auto& v : x_t.data) v = 0.5f;
    auto x = make_ad(x_t);
    auto out = swiglu.forward(x);
    auto s = sum(out);
    s->backward();
    for (auto& g : x->grad.data) {
        assert(std::isfinite(g));
    }
    // Check that gradient is non-zero (active function)
    float grad_norm = 0.0f;
    for (auto& g : x->grad.data) grad_norm += g * g;
    assert(grad_norm > 1e-10f);
    std::cout << "  [PASS] SwiGLU gradient flow\n";
}

void test_swiglu_different_from_zero() {
    clear_parameters();
    ADSwiGLU swiglu(4, 8);
    Tensor x_t(4, 2);
    for (auto& v : x_t.data) v = 1.0f;
    auto x = make_ad(x_t);
    auto out = swiglu.forward(x);
    // Output shouldn't be all zeros
    float sum_abs = 0.0f;
    for (auto& v : out->val.data) sum_abs += std::abs(v);
    assert(sum_abs > 0.0f);
    std::cout << "  [PASS] SwiGLU non-zero output\n";
}

// ========================== RMSNorm Tests ==========================

void test_rmsnorm_basic() {
    clear_parameters();
    ADRMSNorm rn(8);
    Tensor x_t(8, 4);
    for (size_t i = 0; i < x_t.data.size(); ++i) x_t.data[i] = (float)(i % 5) * 0.3f;
    auto x = make_ad(x_t);
    auto out = rn.forward(x);
    assert(out->val.rows == 8);
    assert(out->val.cols == 4);
    assert_finite(out->val, "rmsnorm_basic");
    std::cout << "  [PASS] RMSNorm basic forward\n";
}

void test_rmsnorm_unit_rms() {
    clear_parameters();
    ADRMSNorm rn(4);
    Tensor x_t(4, 1);
    x_t.data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x = make_ad(x_t);
    auto out = rn.forward(x);
    // After RMSNorm, each column should have RMS close to 1 (when gamma=1)
    float rms = 0.0f;
    for (int i = 0; i < 4; ++i) rms += out->val.data[i] * out->val.data[i];
    rms = std::sqrt(rms / 4.0f);
    assert(std::abs(rms - 1.0f) < 0.1f);
    std::cout << "  [PASS] RMSNorm normalizes to unit RMS\n";
}

void test_rmsnorm_gradient() {
    clear_parameters();
    ADRMSNorm rn(4);
    Tensor x_t(4, 2);
    for (auto& v : x_t.data) v = 0.5f;
    auto x = make_ad(x_t);
    auto out = rn.forward(x);
    auto s = sum(out);
    s->backward();
    for (auto& g : x->grad.data) {
        assert(std::isfinite(g));
    }
    std::cout << "  [PASS] RMSNorm gradient flow\n";
}

// ========================== LR Scheduler Tests ==========================

void test_lr_scheduler_warmup() {
    LRScheduler sched(0.001f, 10, 100);
    // During warmup, LR should increase linearly
    float lr0 = sched.get_lr();
    sched.step();
    float lr1 = sched.get_lr();
    assert(lr1 > lr0);
    // At the end of warmup
    for (int i = 1; i < 10; ++i) sched.step();
    float lr_peak = sched.get_lr();
    assert(std::abs(lr_peak - 0.001f) < 1e-6f);
    std::cout << "  [PASS] LR scheduler warmup\n";
}

void test_lr_scheduler_cosine_decay() {
    LRScheduler sched(0.001f, 0, 100, 0.0001f);
    float lr_start = sched.get_lr();
    assert(std::abs(lr_start - 0.001f) < 1e-6f);
    // After some steps, LR should decrease
    for (int i = 0; i < 50; ++i) sched.step();
    float lr_mid = sched.get_lr();
    assert(lr_mid < lr_start);
    assert(lr_mid > 0.0001f);
    // At the end, LR should be close to min_lr
    for (int i = 50; i < 100; ++i) sched.step();
    float lr_end = sched.get_lr();
    assert(std::abs(lr_end - 0.0001f) < 1e-5f);
    std::cout << "  [PASS] LR scheduler cosine decay\n";
}

void test_lr_scheduler_warmup_then_decay() {
    LRScheduler sched(0.01f, 5, 20, 0.001f);
    // Warmup phase: should increase
    std::vector<float> lrs;
    for (int i = 0; i < 20; ++i) {
        lrs.push_back(sched.get_lr());
        sched.step();
    }
    // LR should increase during warmup
    for (int i = 1; i < 5; ++i) {
        assert(lrs[i] > lrs[i-1]);
    }
    // Peak should be at warmup end
    assert(std::abs(lrs[5] - 0.01f) < 1e-5f);
    // Should decrease after warmup
    assert(lrs[15] < lrs[5]);
    std::cout << "  [PASS] LR scheduler warmup + decay\n";
}

// ========================== Main ==========================

int main() {
    std::cout << "=== RoPE Tests ===\n";
    test_rope_basic();
    test_rope_identity_at_zero();
    test_rope_ad_gradient();
    test_rope_pos_offset();

    std::cout << "\n=== SwiGLU Tests ===\n";
    test_swiglu_basic();
    test_swiglu_gradient();
    test_swiglu_different_from_zero();

    std::cout << "\n=== RMSNorm Tests ===\n";
    test_rmsnorm_basic();
    test_rmsnorm_unit_rms();
    test_rmsnorm_gradient();

    std::cout << "\n=== LR Scheduler Tests ===\n";
    test_lr_scheduler_warmup();
    test_lr_scheduler_cosine_decay();
    test_lr_scheduler_warmup_then_decay();

    std::cout << "\nAll new module tests passed.\n";
    return 0;
}
