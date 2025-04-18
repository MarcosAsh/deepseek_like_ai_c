cmake_minimum_required(VERSION 3.10)
project(DeepSeek_AI)

set(CMAKE_CXX_STANDARD 17)

# Find all source files
# Collect all source files
file(GLOB_RECURSE ALL_SRC "src/*.cpp")
file(GLOB_RECURSE TEST_SOURCES "test/*.cpp")

# Separate library sources (exclude main.cpp)
list(FILTER ALL_SRC EXCLUDE REGEX "src/main\\.cpp$")
set(LIB_SOURCES ${ALL_SRC})

# Main executable (includes main.cpp)
add_executable(deepseek_ai src/main.cpp ${LIB_SOURCES})

# Test executable (uses test mains)
add_executable(transformer_test ${TEST_SOURCES} ${LIB_SOURCES})

# Include directories
target_include_directories(deepseek_ai PRIVATE include)
target_include_directories(transformer_test PRIVATE include)