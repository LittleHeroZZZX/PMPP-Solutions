cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0146 NEW)

add_library(ch01_q1_matmul SHARED ch01_q1_matmul.cu)
set_target_properties(ch01_q1_matmul PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)
target_compile_features(ch01_q1_matmul PRIVATE cuda_std_14)

add_custom_command(TARGET ch01_q1_matmul POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:ch01_q1_matmul> ${CMAKE_CURRENT_SOURCE_DIR}/tests/)


