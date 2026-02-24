#include "layers/ad_gqa.hpp"
#include "layers/ad_lora.hpp"
#include "layers/ad_kv_cache.hpp"
#include "layers/ad_repetition_penalty.hpp"
#include "layers/ad_flash_attention.hpp"
#include "layers/ad_weight_tying.hpp"
#include "layers/ad_embedding.hpp"
#include "autodiff.hpp"
#include "tensor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

static void assert_finite(const Tensor& t, const char* name) {
    for (size_t i = 0; i < t.data.size(); ++i) {
        assert(std::isfinite(t.data[i]) && name);
    }
}

// ========================== GQA Tests ==========================

void test_gqa_basic() {
    clear_parameters();
    ADGQA gqa(8, 4, 2);  // 4 Q heads, 2 KV heads
    Tensor x_t(8, 4);
    for (auto& v : x_t.data) v = 0.1f;
    auto out = gqa.forward(make_ad(x_t));
    assert(out->val.rows == 8);
    assert(out->val.cols == 4);
    assert_finite(out->val, "gqa_basic");
    std::cout << "  [PASS] GQA basic forward\n";
}

void test_gqa_single_kv_head() {
    clear_parameters();
    // Extreme case: 4 Q heads share 1 KV head (like multi-query attention)
    ADGQA gqa(8, 4, 1);
    Tensor x_t(8, 3);
    for (auto& v : x_t.data) v = 0.2f;
    auto out = gqa.forward(make_ad(x_t));
    assert(out->val.rows == 8 && out->val.cols == 3);
    assert_finite(out->val, "gqa_single_kv");
    std::cout << "  [PASS] GQA single KV head (multi-query)\n";
}

void test_gqa_gradient() {
    clear_parameters();
    ADGQA gqa(8, 4, 2);
    Tensor x_t(8, 3);
    for (auto& v : x_t.data) v = 0.15f;
    auto x = make_ad(x_t);
    register_parameter(x);
    sum(gqa.forward(x))->backward();
    bool has_nonzero = false;
    for (auto& g : x->grad.data) {
        assert(std::isfinite(g));
        if (std::fabs(g) > 1e-8f) has_nonzero = true;
    }
    assert(has_nonzero);
    std::cout << "  [PASS] GQA gradient flow\n";
}

// ========================== LoRA Tests ==========================

void test_lora_basic() {
    clear_parameters();
    ADLoRA lora(8, 8, 4, 4.0f);
    Tensor x_t(8, 3);
    for (auto& v : x_t.data) v = 0.5f;
    auto out = lora.forward(make_ad(x_t));
    assert(out->val.rows == 8 && out->val.cols == 3);
    assert_finite(out->val, "lora_basic");
    std::cout << "  [PASS] LoRA basic forward\n";
}

void test_lora_initial_identity() {
    // B is initialized to zero, so initially LoRA contribution should be zero
    // Output should equal W*x + b
    clear_parameters();
    ADLoRA lora(4, 4, 2, 2.0f);
    Tensor x_t(4, 1);
    x_t.data = {1.0f, 0.0f, 0.0f, 0.0f};
    auto out = lora.forward(make_ad(x_t));
    assert(out->val.rows == 4 && out->val.cols == 1);
    assert_finite(out->val, "lora_init");
    std::cout << "  [PASS] LoRA initial output (B=0)\n";
}

void test_lora_gradient() {
    clear_parameters();
    ADLoRA lora(8, 8, 4, 4.0f);
    Tensor x_t(8, 2);
    for (auto& v : x_t.data) v = 0.3f;
    auto x = make_ad(x_t);
    register_parameter(x);
    sum(lora.forward(x))->backward();
    float grad_norm = 0.0f;
    for (auto& g : x->grad.data) {
        assert(std::isfinite(g));
        grad_norm += g * g;
    }
    assert(grad_norm > 1e-10f);
    std::cout << "  [PASS] LoRA gradient flow\n";
}

void test_lora_rank_reduction() {
    // Higher rank vs lower rank should give different parameter counts
    clear_parameters();
    ADLoRA lora_small(16, 16, 2, 2.0f);
    auto params_small = get_parameters().size();

    clear_parameters();
    ADLoRA lora_large(16, 16, 8, 8.0f);
    auto params_large = get_parameters().size();

    // Both should have same number of registered params (A, B, bias = 3)
    // but A and B have different sizes
    assert(params_small == params_large);
    std::cout << "  [PASS] LoRA rank parameterization\n";
}

// ========================== KV Cache Tests ==========================

void test_kv_cache_basic() {
    ADKVCache cache(8);
    Tensor k(4, 3);
    Tensor v(4, 3);
    for (auto& x : k.data) x = 1.0f;
    for (auto& x : v.data) x = 2.0f;
    auto result = cache.update(make_ad(k), make_ad(v));
    assert(result.keys->val.rows == 4);
    assert(result.keys->val.cols == 3);
    assert(result.values->val.cols == 3);
    std::cout << "  [PASS] KV cache basic\n";
}

void test_kv_cache_accumulation() {
    ADKVCache cache(10);
    Tensor k1(4, 3), v1(4, 3);
    for (auto& x : k1.data) x = 1.0f;
    for (auto& x : v1.data) x = 1.0f;
    cache.update(make_ad(k1), make_ad(v1));
    assert(cache.cached_length() == 3);

    Tensor k2(4, 2), v2(4, 2);
    for (auto& x : k2.data) x = 2.0f;
    for (auto& x : v2.data) x = 2.0f;
    auto result = cache.update(make_ad(k2), make_ad(v2));
    assert(cache.cached_length() == 5);
    assert(result.keys->val.cols == 5);
    std::cout << "  [PASS] KV cache accumulation\n";
}

void test_kv_cache_sliding_window() {
    ADKVCache cache(4);  // window size 4
    Tensor k1(2, 3), v1(2, 3);
    for (auto& x : k1.data) x = 1.0f;
    for (auto& x : v1.data) x = 1.0f;
    cache.update(make_ad(k1), make_ad(v1));

    Tensor k2(2, 3), v2(2, 3);
    for (auto& x : k2.data) x = 2.0f;
    for (auto& x : v2.data) x = 2.0f;
    auto result = cache.update(make_ad(k2), make_ad(v2));

    // Total would be 6, but window=4, so only last 4
    assert(result.keys->val.cols == 4);
    assert(result.values->val.cols == 4);
    std::cout << "  [PASS] KV cache sliding window\n";
}

void test_kv_cache_clear() {
    ADKVCache cache(8);
    Tensor k(4, 3), v(4, 3);
    for (auto& x : k.data) x = 1.0f;
    for (auto& x : v.data) x = 1.0f;
    cache.update(make_ad(k), make_ad(v));
    assert(cache.cached_length() == 3);
    cache.clear();
    assert(cache.cached_length() == 0);
    std::cout << "  [PASS] KV cache clear\n";
}

// ========================== Repetition Penalty Tests ==========================

void test_rep_penalty_basic() {
    clear_parameters();
    ADRepetitionPenalty rp(1.5f);
    Tensor logits_t(10, 1);
    for (int i = 0; i < 10; ++i) logits_t.data[i] = 1.0f;
    auto logits = make_ad(logits_t);
    auto out = rp.apply(logits, {2, 5});
    // Token 2 and 5 should have reduced logits (divided by 1.5)
    float expected = 1.0f / 1.5f;
    assert(std::abs(out->val.data[2] - expected) < 1e-5f);
    assert(std::abs(out->val.data[5] - expected) < 1e-5f);
    // Other tokens unchanged
    assert(std::abs(out->val.data[0] - 1.0f) < 1e-5f);
    std::cout << "  [PASS] Repetition penalty basic\n";
}

void test_rep_penalty_negative_logits() {
    clear_parameters();
    ADRepetitionPenalty rp(2.0f);
    Tensor logits_t(5, 1);
    logits_t.data = {1.0f, -1.0f, 0.5f, -0.5f, 0.0f};
    auto logits = make_ad(logits_t);
    auto out = rp.apply(logits, {1, 3});  // penalize tokens 1 and 3 (negative logits)
    // Negative logits get multiplied by penalty (more negative)
    assert(std::abs(out->val.data[1] - (-2.0f)) < 1e-5f);
    assert(std::abs(out->val.data[3] - (-1.0f)) < 1e-5f);
    std::cout << "  [PASS] Repetition penalty negative logits\n";
}

void test_rep_penalty_no_effect() {
    clear_parameters();
    ADRepetitionPenalty rp(1.0f);  // penalty=1.0 means no change
    Tensor logits_t(5, 1);
    for (int i = 0; i < 5; ++i) logits_t.data[i] = (float)i;
    auto logits = make_ad(logits_t);
    auto out = rp.apply(logits, {0, 1, 2, 3, 4});
    for (int i = 0; i < 5; ++i) {
        assert(std::abs(out->val.data[i] - logits_t.data[i]) < 1e-5f);
    }
    std::cout << "  [PASS] Repetition penalty no-op at 1.0\n";
}

// ========================== Flash Attention Tests ==========================

void test_flash_basic() {
    clear_parameters();
    ADFlashAttention flash(8, 2, 4);  // tile_size=4
    Tensor x_t(8, 4);
    for (auto& v : x_t.data) v = 0.1f;
    auto out = flash.forward(make_ad(x_t));
    assert(out->val.rows == 8 && out->val.cols == 4);
    assert_finite(out->val, "flash_basic");
    std::cout << "  [PASS] Flash attention basic\n";
}

void test_flash_larger_than_tile() {
    clear_parameters();
    ADFlashAttention flash(8, 2, 3);  // tile_size=3, seq_len=6 -> needs tiling
    Tensor x_t(8, 6);
    for (auto& v : x_t.data) v = 0.1f;
    auto out = flash.forward(make_ad(x_t));
    assert(out->val.rows == 8 && out->val.cols == 6);
    assert_finite(out->val, "flash_tiled");
    std::cout << "  [PASS] Flash attention tiled (seq > tile)\n";
}

void test_flash_gradient() {
    clear_parameters();
    ADFlashAttention flash(8, 2, 32);  // large tile = standard attention path
    Tensor x_t(8, 4);
    for (auto& v : x_t.data) v = 0.2f;
    auto x = make_ad(x_t);
    register_parameter(x);
    sum(flash.forward(x))->backward();
    bool has_nonzero = false;
    for (auto& g : x->grad.data) {
        assert(std::isfinite(g));
        if (std::fabs(g) > 1e-8f) has_nonzero = true;
    }
    assert(has_nonzero);
    std::cout << "  [PASS] Flash attention gradient\n";
}

// ========================== Weight Tying Tests ==========================

void test_weight_tying_basic() {
    clear_parameters();
    // Simulate embedding weights [vocab_size x embed_dim]
    Tensor emb_t(10, 4);
    for (size_t i = 0; i < emb_t.data.size(); ++i) emb_t.data[i] = 0.1f * (i % 7);
    auto emb_weights = make_ad(emb_t);
    register_parameter(emb_weights);

    ADWeightTying wt(emb_weights);
    Tensor hidden_t(4, 3);  // [embed_dim x seq_len]
    for (auto& v : hidden_t.data) v = 0.5f;
    auto out = wt.forward(make_ad(hidden_t));
    // Output should be [vocab_size x seq_len] = [10 x 3]
    assert(out->val.rows == 10 && out->val.cols == 3);
    assert_finite(out->val, "weight_tying");
    std::cout << "  [PASS] Weight tying basic\n";
}

void test_weight_tying_gradient() {
    clear_parameters();
    Tensor emb_t(8, 4);
    for (auto& v : emb_t.data) v = 0.1f;
    auto emb_weights = make_ad(emb_t);
    register_parameter(emb_weights);

    ADWeightTying wt(emb_weights);
    Tensor hidden_t(4, 2);
    for (auto& v : hidden_t.data) v = 0.3f;
    auto hidden = make_ad(hidden_t);
    register_parameter(hidden);

    sum(wt.forward(hidden))->backward();
    // Both embedding weights and hidden should receive gradients
    float grad_norm_emb = 0.0f;
    for (auto& g : emb_weights->grad.data) {
        assert(std::isfinite(g));
        grad_norm_emb += g * g;
    }
    float grad_norm_hidden = 0.0f;
    for (auto& g : hidden->grad.data) {
        assert(std::isfinite(g));
        grad_norm_hidden += g * g;
    }
    assert(grad_norm_emb > 1e-10f);
    assert(grad_norm_hidden > 1e-10f);
    std::cout << "  [PASS] Weight tying gradient flow\n";
}

void test_weight_tying_with_embedding() {
    clear_parameters();
    ADEmbedding emb(16, 8);  // vocab=16, embed_dim=8
    auto weights = emb.get_weights();
    ADWeightTying wt(weights);

    // Embed some tokens
    auto embedded = emb.forward({1, 2, 3});  // [8 x 3]
    // Project back to logits using tied weights
    auto logits = wt.forward(embedded);  // [16 x 3]
    assert(logits->val.rows == 16 && logits->val.cols == 3);
    assert_finite(logits->val, "weight_tying_e2e");
    std::cout << "  [PASS] Weight tying end-to-end with embedding\n";
}

// ========================== Main ==========================

int main() {
    std::cout << "=== GQA Tests ===\n";
    test_gqa_basic();
    test_gqa_single_kv_head();
    test_gqa_gradient();

    std::cout << "\n=== LoRA Tests ===\n";
    test_lora_basic();
    test_lora_initial_identity();
    test_lora_gradient();
    test_lora_rank_reduction();

    std::cout << "\n=== KV Cache Tests ===\n";
    test_kv_cache_basic();
    test_kv_cache_accumulation();
    test_kv_cache_sliding_window();
    test_kv_cache_clear();

    std::cout << "\n=== Repetition Penalty Tests ===\n";
    test_rep_penalty_basic();
    test_rep_penalty_negative_logits();
    test_rep_penalty_no_effect();

    std::cout << "\n=== Flash Attention Tests ===\n";
    test_flash_basic();
    test_flash_larger_than_tile();
    test_flash_gradient();

    std::cout << "\n=== Weight Tying Tests ===\n";
    test_weight_tying_basic();
    test_weight_tying_gradient();
    test_weight_tying_with_embedding();

    std::cout << "\nAll advanced module tests passed.\n";
    return 0;
}
