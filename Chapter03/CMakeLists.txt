cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0146 NEW)
# Find Python
find_package(Python COMPONENTS Interpreter REQUIRED)

# Find all .cu files in the current directory
file(GLOB CUDA_SOURCE_FILES "*.cu")

# Function to get the target name from a file path
function(get_target_name filepath out_var)
    get_filename_component(filename ${filepath} NAME_WE)
    set(${out_var} ${filename} PARENT_SCOPE)
endfunction()

# List to store all test targets
set(ALL_TEST_TARGETS "")

# Iterate over each .cu file and create targets
foreach(CUDA_SOURCE ${CUDA_SOURCE_FILES})
    get_target_name(${CUDA_SOURCE} TARGET_NAME)

    # Add library target
    add_library(${TARGET_NAME} SHARED ${CUDA_SOURCE})
    set_target_properties(${TARGET_NAME} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
    )
    target_compile_features(${TARGET_NAME} PRIVATE cuda_std_14)

    # Add custom command to copy the library
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TARGET_NAME}> ${CMAKE_CURRENT_SOURCE_DIR}/tests/
    )

    # Add custom target for running tests
    add_custom_target(test_${TARGET_NAME}
            COMMAND ${CMAKE_COMMAND} -E echo "Running test for ${TARGET_NAME}"
            COMMAND ${CMAKE_COMMAND} -E env PYTHONUNBUFFERED=1 ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_${TARGET_NAME}.py
            COMMAND ${CMAKE_COMMAND} -E echo "Test for ${TARGET_NAME} completed"
            DEPENDS ${TARGET_NAME}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/
    )

    # Add this test target to our list of all test targets
    list(APPEND ALL_TEST_TARGETS test_${TARGET_NAME})
endforeach()

# Extract the chapter number from the current directory name
get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REGEX MATCH "Chapter([0-9]+)" CHAPTER_MATCH ${CURRENT_DIR_NAME})
set(CHAPTER_NUMBER ${CMAKE_MATCH_1})

# Create a target to run all tests for this chapter
add_custom_target(test_ch${CHAPTER_NUMBER}
        COMMAND ${CMAKE_COMMAND} -E echo "Running all tests for Chapter ${CHAPTER_NUMBER}"
        DEPENDS ${ALL_TEST_TARGETS}
        COMMAND ${CMAKE_COMMAND} -E echo "All tests for Chapter ${CHAPTER_NUMBER} completed"
)







