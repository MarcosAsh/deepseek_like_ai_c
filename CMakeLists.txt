cmake_minimum_required(VERSION 3.10)
project(GenAI)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(main main.cpp ${SOURCES})

# Add a separate executable for the test
file(GLOB_RECURSE TEST_SOURCES "test/*.cpp")
add_executable(transformer_test ${TEST_SOURCES} ${SOURCES})