cmake_minimum_required(VERSION 3.18)
project(ffn-probe LANGUAGES C CXX)

set(LLAMA_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../vendor/llama.cpp")
add_subdirectory("${LLAMA_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/llama-build" EXCLUDE_FROM_ALL)

add_executable(ffn_probe ffn_probe.cpp)
target_include_directories(ffn_probe PRIVATE
    "${LLAMA_SOURCE_DIR}/include"
    "${LLAMA_SOURCE_DIR}/ggml/include"
    "${LLAMA_SOURCE_DIR}/src"
)
target_link_libraries(ffn_probe PRIVATE llama ggml)
target_compile_features(ffn_probe PRIVATE cxx_std_17)
