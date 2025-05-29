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

# Separate library sources (exclude main.cpp and train_bpe.cpp)
list(FILTER ALL_SRC EXCLUDE REGEX "src/main\\.cpp$")
list(FILTER ALL_SRC EXCLUDE REGEX "src/train_bpe\\.cpp$")
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
  
# BPE training tool
add_executable(train_bpe src/train_bpe.cpp)
target_include_directories(train_bpe PRIVATE include)

# On Apple platforms, use Accelerate framework for optimized BLAS
if(APPLE)
  # Enable Accelerate-based GEMM implementation
  target_compile_definitions(deepseek_ai PRIVATE USE_ACCELERATE)
  target_compile_definitions(transformer_test PRIVATE USE_ACCELERATE)
  # Link against Apple's Accelerate framework for optimized BLAS/LAPACK
  target_link_libraries(deepseek_ai PRIVATE "-framework Accelerate")
  target_link_libraries(transformer_test PRIVATE "-framework Accelerate")
endif()