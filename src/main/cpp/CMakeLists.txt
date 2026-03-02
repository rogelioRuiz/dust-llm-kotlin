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

file(GLOB MTMD_MODEL_SOURCES ${LLAMA_CPP_DIR}/tools/mtmd/models/*.cpp)

add_library(
    mtmd
    OBJECT
    ${LLAMA_CPP_DIR}/tools/mtmd/clip.cpp
    ${LLAMA_CPP_DIR}/tools/mtmd/mtmd.cpp
    ${LLAMA_CPP_DIR}/tools/mtmd/mtmd-helper.cpp
    ${LLAMA_CPP_DIR}/tools/mtmd/mtmd-audio.cpp
    ${MTMD_MODEL_SOURCES}
)

target_include_directories(
    mtmd
    PUBLIC
    ${LLAMA_CPP_DIR}/tools/mtmd
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/ggml/src
    ${LLAMA_CPP_DIR}/vendor
)

target_compile_options(
    mtmd
    PRIVATE
    -Wno-cast-qual
)

if(TARGET ggml)
    target_link_libraries(mtmd PRIVATE ggml)
endif()

target_link_libraries(mtmd PRIVATE llama)

add_library(
    llama_jni
    SHARED
    llama_jni.cpp
)

target_include_directories(
    llama_jni
    PRIVATE
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/tools/mtmd
)

target_link_libraries(
    llama_jni
    PRIVATE
    llama
    $<TARGET_OBJECTS:mtmd>
    log
)

if(TARGET ggml)
    target_link_libraries(llama_jni PRIVATE ggml)
endif()
