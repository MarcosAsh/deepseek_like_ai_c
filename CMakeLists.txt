cmake_minimum_required(VERSION 3.10)
project(DeepSeek_AI)

set(CMAKE_CXX_STANDARD 17)

# Find all source files
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE TEST_SOURCES "test/*.cpp")

# Main executable
add_executable(deepseek_ai ${SOURCES})

# Test executable
add_executable(transformer_test ${TEST_SOURCES} ${SOURCES})

# Include directories
target_include_directories(deepseek_ai PRIVATE include)
target_include_directories(transformer_test PRIVATE include)