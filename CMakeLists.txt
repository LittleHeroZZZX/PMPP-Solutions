cmake_minimum_required(VERSION 3.18)
project(PMPP CUDA)
set(CMAKE_CUDA_ARCHITECTURES native)
# Option to control debug info
option(CUDA_DEBUG_INFO "Add debug information to CUDA compiled .so files" ON)

# Set CUDA debug flags if option is ON
if(CUDA_DEBUG_INFO)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
endif()

add_subdirectory(Chapter02)