cmake_minimum_required(VERSION 3.10)
project(DeepSeek_AI)
enable_testing()

set(CMAKE_CXX_STANDARD 17)

# Default to Release build with high-performance flags
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# Architecture tuning: optimize for the host CPU
set(ARCH_FLAGS "-march=native")
# Release flags: optimize, fast math, LTO
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math ${ARCH_FLAGS} -flto")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto")

# Find all source files
# Collect all source files
file(GLOB_RECURSE ALL_SRC "src/*.cpp")
file(GLOB_RECURSE TEST_SOURCES "test/*.cpp")

# Separate library sources (exclude entry points and server-specific code)
list(FILTER ALL_SRC EXCLUDE REGEX "src/main\\.cpp$")
list(FILTER ALL_SRC EXCLUDE REGEX "src/train_bpe\\.cpp$")
list(FILTER ALL_SRC EXCLUDE REGEX "src/node_server_main\\.cpp$")
list(FILTER ALL_SRC EXCLUDE REGEX "src/server/")
set(LIB_SOURCES ${ALL_SRC})

# Main executable (includes main.cpp)
add_executable(deepseek_ai src/main.cpp ${LIB_SOURCES})

# Test executable (uses test mains)
add_executable(transformer_test test/transformer_test.cpp ${LIB_SOURCES})
add_test(NAME transformer_test COMMAND transformer_test)

# Include directories
# Include headers
target_include_directories(deepseek_ai PRIVATE include)
target_include_directories(transformer_test PRIVATE include)

# Unit test for Tensor operations
add_executable(tensor_test test/tensor_test.cpp ${LIB_SOURCES})
target_include_directories(tensor_test PRIVATE include)
if(APPLE)
  target_compile_definitions(tensor_test PRIVATE USE_ACCELERATE)
  target_link_libraries(tensor_test PRIVATE "-framework Accelerate")
endif()
add_test(NAME tensor_test COMMAND tensor_test)

# Unit test for Tokenizer
add_executable(tokenizer_test test/tokenizer_test.cpp ${LIB_SOURCES})
target_include_directories(tokenizer_test PRIVATE include)
if(APPLE)
  target_compile_definitions(tokenizer_test PRIVATE USE_ACCELERATE)
  target_link_libraries(tokenizer_test PRIVATE "-framework Accelerate")
endif()
add_test(NAME tokenizer_test COMMAND tokenizer_test)

# Unit test for Autodiff operations
add_executable(autodiff_test test/autodiff_test.cpp ${LIB_SOURCES})
target_include_directories(autodiff_test PRIVATE include)
if(APPLE)
  target_compile_definitions(autodiff_test PRIVATE USE_ACCELERATE)
  target_link_libraries(autodiff_test PRIVATE "-framework Accelerate")
endif()
add_test(NAME autodiff_test COMMAND autodiff_test)

# Unit tests for individual layers (Embedding, PositionalEncoding, Linear)
add_executable(layers_test test/layers_test.cpp ${LIB_SOURCES})
target_include_directories(layers_test PRIVATE include)
if(APPLE)
  target_compile_definitions(layers_test PRIVATE USE_ACCELERATE)
  target_link_libraries(layers_test PRIVATE "-framework Accelerate")
endif()
add_test(NAME layers_test COMMAND layers_test)

# Unit test for softmax_cross_entropy loss
add_executable(loss_test test/loss_test.cpp ${LIB_SOURCES})
target_include_directories(loss_test PRIVATE include)
if(APPLE)
  target_compile_definitions(loss_test PRIVATE USE_ACCELERATE)
  target_link_libraries(loss_test PRIVATE "-framework Accelerate")
endif()
add_test(NAME loss_test COMMAND loss_test)
  
# AD attention test
add_executable(ad_attention_test test/ad_attention_test.cpp ${LIB_SOURCES})
target_include_directories(ad_attention_test PRIVATE include)
if(APPLE)
  target_compile_definitions(ad_attention_test PRIVATE USE_ACCELERATE)
  target_link_libraries(ad_attention_test PRIVATE "-framework Accelerate")
endif()
add_test(NAME ad_attention_test COMMAND ad_attention_test)

# Feed-forward test
add_executable(feed_forward_test test/feed_forward_test.cpp ${LIB_SOURCES})
target_include_directories(feed_forward_test PRIVATE include)
if(APPLE)
  target_compile_definitions(feed_forward_test PRIVATE USE_ACCELERATE)
  target_link_libraries(feed_forward_test PRIVATE "-framework Accelerate")
endif()
add_test(NAME feed_forward_test COMMAND feed_forward_test)

# Quantization test
add_executable(quantization_test test/quantization_test.cpp ${LIB_SOURCES})
target_include_directories(quantization_test PRIVATE include)
if(APPLE)
  target_compile_definitions(quantization_test PRIVATE USE_ACCELERATE)
  target_link_libraries(quantization_test PRIVATE "-framework Accelerate")
endif()
add_test(NAME quantization_test COMMAND quantization_test)

# LayerNorm + Dropout test
add_executable(layer_norm_test test/layer_norm_test.cpp ${LIB_SOURCES})
target_include_directories(layer_norm_test PRIVATE include)
if(APPLE)
  target_compile_definitions(layer_norm_test PRIVATE USE_ACCELERATE)
  target_link_libraries(layer_norm_test PRIVATE "-framework Accelerate")
endif()
add_test(NAME layer_norm_test COMMAND layer_norm_test)

# AD layers test (ADLinear, ADEmbedding, ADLayerNorm, ADPositionalEncoding)
add_executable(ad_layers_test test/ad_layers_test.cpp ${LIB_SOURCES})
target_include_directories(ad_layers_test PRIVATE include)
if(APPLE)
  target_compile_definitions(ad_layers_test PRIVATE USE_ACCELERATE)
  target_link_libraries(ad_layers_test PRIVATE "-framework Accelerate")
endif()
add_test(NAME ad_layers_test COMMAND ad_layers_test)

# AD transformer test
add_executable(ad_transformer_test test/ad_transformer_test.cpp ${LIB_SOURCES})
target_include_directories(ad_transformer_test PRIVATE include)
if(APPLE)
  target_compile_definitions(ad_transformer_test PRIVATE USE_ACCELERATE)
  target_link_libraries(ad_transformer_test PRIVATE "-framework Accelerate")
endif()
add_test(NAME ad_transformer_test COMMAND ad_transformer_test)

# MoE test
add_executable(moe_test test/moe_test.cpp ${LIB_SOURCES})
target_include_directories(moe_test PRIVATE include)
if(APPLE)
  target_compile_definitions(moe_test PRIVATE USE_ACCELERATE)
  target_link_libraries(moe_test PRIVATE "-framework Accelerate")
endif()
add_test(NAME moe_test COMMAND moe_test)

# Optimizer test
add_executable(optimizer_test test/optimizer_test.cpp ${LIB_SOURCES})
target_include_directories(optimizer_test PRIVATE include)
if(APPLE)
  target_compile_definitions(optimizer_test PRIVATE USE_ACCELERATE)
  target_link_libraries(optimizer_test PRIVATE "-framework Accelerate")
endif()
add_test(NAME optimizer_test COMMAND optimizer_test)

# New modules test (RoPE, SwiGLU, RMSNorm, LR scheduler)
add_executable(new_modules_test test/new_modules_test.cpp ${LIB_SOURCES})
target_include_directories(new_modules_test PRIVATE include)
if(APPLE)
  target_compile_definitions(new_modules_test PRIVATE USE_ACCELERATE)
  target_link_libraries(new_modules_test PRIVATE "-framework Accelerate")
endif()
add_test(NAME new_modules_test COMMAND new_modules_test)

# Advanced modules test (GQA, LoRA, KV Cache, Rep Penalty, Flash Attention, Weight Tying)
add_executable(advanced_modules_test test/advanced_modules_test.cpp ${LIB_SOURCES})
target_include_directories(advanced_modules_test PRIVATE include)
if(APPLE)
  target_compile_definitions(advanced_modules_test PRIVATE USE_ACCELERATE)
  target_link_libraries(advanced_modules_test PRIVATE "-framework Accelerate")
endif()
add_test(NAME advanced_modules_test COMMAND advanced_modules_test)

# Vision test (Conv2D, Pool2D, BatchNorm2D, Flatten, CNN pipeline)
add_executable(vision_test test/vision_test.cpp ${LIB_SOURCES})
target_include_directories(vision_test PRIVATE include)
if(APPLE)
  target_compile_definitions(vision_test PRIVATE USE_ACCELERATE)
  target_link_libraries(vision_test PRIVATE "-framework Accelerate")
endif()
add_test(NAME vision_test COMMAND vision_test)

# BPE training tool
add_executable(train_bpe src/train_bpe.cpp)
target_include_directories(train_bpe PRIVATE include)

# Node graph server
file(GLOB SERVER_SOURCES "src/server/*.cpp")
add_executable(node_server src/node_server_main.cpp ${SERVER_SOURCES} ${LIB_SOURCES})
target_include_directories(node_server PRIVATE include)
# httplib requires threads
find_package(Threads REQUIRED)
target_link_libraries(node_server PRIVATE Threads::Threads)
if(APPLE)
  target_compile_definitions(node_server PRIVATE USE_ACCELERATE)
  target_link_libraries(node_server PRIVATE "-framework Accelerate")
endif()

# On Apple platforms, use Accelerate framework for optimized BLAS
if(APPLE)
  # Enable Accelerate-based GEMM implementation
  target_compile_definitions(deepseek_ai PRIVATE USE_ACCELERATE)
  target_compile_definitions(transformer_test PRIVATE USE_ACCELERATE)
  # Link against Apple's Accelerate framework for optimized BLAS/LAPACK
  target_link_libraries(deepseek_ai PRIVATE "-framework Accelerate")
  target_link_libraries(transformer_test PRIVATE "-framework Accelerate")
endif()