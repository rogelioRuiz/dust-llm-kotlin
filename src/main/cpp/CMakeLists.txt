cmake_minimum_required(VERSION 3.22.1)
project(dust_llm_jni)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_CPP_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../../dust-llm-swift/native/llama.cpp)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_CPP_DIR} ${CMAKE_CURRENT_BINARY_DIR}/llama.cpp)

add_library(
    llava
    OBJECT
    ${LLAMA_CPP_DIR}/examples/llava/clip.cpp
    ${LLAMA_CPP_DIR}/examples/llava/llava.cpp
)

target_include_directories(
    llava
    PUBLIC
    ${LLAMA_CPP_DIR}/examples/llava
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/ggml/src
    ${LLAMA_CPP_DIR}/common
)

target_compile_options(
    llava
    PRIVATE
    -Wno-cast-qual
)

if(TARGET ggml)
    target_link_libraries(llava PRIVATE ggml)
endif()

target_link_libraries(llava PRIVATE llama)

add_library(
    llama_jni
    SHARED
    llama_jni.cpp
)

target_include_directories(
    llama_jni
    PRIVATE
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/examples/llava
)

target_link_libraries(
    llama_jni
    PRIVATE
    llama
    $<TARGET_OBJECTS:llava>
    log
)

if(TARGET ggml)
    target_link_libraries(llama_jni PRIVATE ggml)
endif()
