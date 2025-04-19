cmake_minimum_required(VERSION 3.10)
project(DeepSeek_AI)

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