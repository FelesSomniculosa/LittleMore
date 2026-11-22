# LittleMore top-level build.
#
# Builds the high-level C API (`include/littlemore.h`, `src/littlemore.cpp`) as one shared library,
# `littlemore`, on top of the patched llama.cpp in `engine/` (added here with add_subdirectory, so one
# configure builds the engine and the API with the same compiler and options).
#
# Packaging model: the ggml compute backends (CPU variants, CUDA, Vulkan, ...) are built as runtime-loaded
# plugins (GGML_BACKEND_DL). `littlemore` links only the engine's core libraries (llama, ggml, ggml-base) and
# loads every backend plugin it finds next to itself at the first API call, so one package runs on machines
# with or without a GPU: a missing NVIDIA driver only means the CUDA plugin is skipped.
#
# Typical configure (Windows, from a shell with the MSVC compiler; see docs/development.md):
#   cmake -G Ninja -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j 6
#   ctest --test-dir build --output-on-failure
#
# Everything (littlemore, llama, ggml, the backend plugins, the tests) lands in build/bin.

cmake_minimum_required(VERSION 3.21)

# The version lives in one place, the public header, so the library and the header can never disagree.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/littlemore.h" _lm_version_lines
     REGEX "^#define LM_VERSION_(MAJOR|MINOR|PATCH) +[0-9]+")
foreach(_line IN LISTS _lm_version_lines)
    if (_line MATCHES "^#define LM_VERSION_(MAJOR|MINOR|PATCH) +([0-9]+)")
        set(_lm_${CMAKE_MATCH_1} ${CMAKE_MATCH_2})
    endif()
endforeach()
if (NOT DEFINED _lm_MAJOR OR NOT DEFINED _lm_MINOR OR NOT DEFINED _lm_PATCH)
    message(FATAL_ERROR "could not read LM_VERSION_MAJOR/MINOR/PATCH from include/littlemore.h")
endif()

project(littlemore VERSION ${_lm_MAJOR}.${_lm_MINOR}.${_lm_PATCH} LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# The test of the C API is plain C on purpose (it proves the header has no C++ in it).
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(LITTLEMORE_BUILD_TESTS "littlemore: build the C API tests (CTest)" ${PROJECT_IS_TOP_LEVEL})

# ---------------------------------------------------------------------------------------------------------------
# Engine defaults. These are only defaults: a value given on the cmake command line wins (set(... CACHE) does
# not overwrite an existing cache entry).
#   BUILD_SHARED_LIBS      backend plugins need a shared ggml-base to link against.
#   GGML_BACKEND_DL        backends as runtime-loaded plugins (the packaging model above).
#   GGML_CPU_ALL_VARIANTS  one CPU plugin per instruction-set level; the best one for the running CPU is picked
#                          at load time (the AVX2 LittleBit kernel is in the AVX2 variants only).
# ---------------------------------------------------------------------------------------------------------------
set(BUILD_SHARED_LIBS ON CACHE BOOL "build shared libraries (required by GGML_BACKEND_DL)")
set(GGML_BACKEND_DL ON CACHE BOOL "ggml: build backends as dynamically loadable plugins")
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64)$")
    set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "ggml: build one CPU backend plugin per instruction-set level")
endif()
# The C API only needs libllama; the engine's tools, examples, tests and its `common` helper library are not
# built from here (build them from engine/ directly, see docs/development.md).
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "llama: build common utils library")
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "llama: build tests")
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "llama: build tools")
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "llama: build examples")
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "llama: build server example")

if (NOT BUILD_SHARED_LIBS)
    message(FATAL_ERROR "littlemore is a shared library over a shared engine: BUILD_SHARED_LIBS must be ON")
endif()
if (NOT GGML_BACKEND_DL)
    # Not fatal: a static-backend build still works (the backends are then linked into ggml), it just loses the
    # "one package for every machine" property. Say so loudly instead of failing.
    message(WARNING "GGML_BACKEND_DL is OFF: the backends are linked into ggml instead of loaded as plugins")
endif()

if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/engine/CMakeLists.txt")
    message(FATAL_ERROR "engine/ (the patched llama.cpp) is missing; see docs/engine-changes.md for how to create it")
endif()
# One output directory for everything (the engine sets the same for its own targets, but only inside its own
# directory scope): littlemore must sit next to llama/ggml and the backend plugins it loads at run time.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_subdirectory(engine)

# ---------------------------------------------------------------------------------------------------------------
# The C API library.
# ---------------------------------------------------------------------------------------------------------------
add_library(littlemore SHARED
    include/littlemore.h
    src/littlemore.cpp)
target_include_directories(littlemore PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
# PRIVATE: the public header exposes no engine type, so users of littlemore never need llama.h or ggml.h.
target_link_libraries(littlemore PRIVATE llama ggml ggml-base)
# LM_BUILD selects dllexport (Windows) / default visibility (elsewhere) in the header's LM_API macro.
target_compile_definitions(littlemore PRIVATE LM_BUILD)
set_target_properties(littlemore PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})
if (MSVC)
    # UTF-8 sources (comments contain non-ASCII math symbols); /W4 for the new code only.
    target_compile_options(littlemore PRIVATE /utf-8 /W4)
    # getenv/sscanf are used as the C standard defines them; silence MSVC's "use the _s variant" advice.
    target_compile_definitions(littlemore PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(littlemore PRIVATE -Wall -Wextra)
endif()
# With backend plugins the library must be able to find them next to itself: make sure they are built
# whenever littlemore is (ggml already depends on every plugin target, this just states it for readers).
add_dependencies(littlemore ggml)

# ---------------------------------------------------------------------------------------------------------------
# Tests (CTest): tests/c/test_api.c, compiled as C, run once on the CPU and once on CUDA (when built with CUDA;
# the test skips itself with code 77 when no CUDA device is present).
#
# The test needs a tiny GGUF model. A fixture test builds it first with tests/c/make_tiny_gguf.py, using the
# project's virtual environment (.venv, see docs/setup-windows.md) or any Python 3 found; the file is cached
# under .cache/gguf/ and rebuilt only when its sources change. Without Python the fixture is not added and
# the tests skip with a message (unless the model already exists).
# ---------------------------------------------------------------------------------------------------------------
if (LITTLEMORE_BUILD_TESTS)
    enable_testing()

    add_executable(test_api tests/c/test_api.c)
    set_source_files_properties(tests/c/test_api.c PROPERTIES LANGUAGE C)
    target_link_libraries(test_api PRIVATE littlemore)
    if (WIN32)
        target_link_libraries(test_api PRIVATE psapi)   # GetProcessMemoryInfo, for the leak check
    else()
        find_package(Threads REQUIRED)
        target_link_libraries(test_api PRIVATE Threads::Threads)
    endif()
    if (MSVC)
        target_compile_options(test_api PRIVATE /W4 /utf-8)
        target_compile_definitions(test_api PRIVATE _CRT_SECURE_NO_WARNINGS)
    else()
        target_compile_options(test_api PRIVATE -Wall -Wextra)
    endif()

    set(LITTLEMORE_TEST_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/.cache/gguf/tiny-llama-lb2-chatml.gguf"
        CACHE FILEPATH "tiny GGUF used by test_api (built by tests/c/make_tiny_gguf.py)")

    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Scripts/python.exe")
        set(_lm_python "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Scripts/python.exe")
    elseif (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
        set(_lm_python "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
    else()
        find_package(Python3 COMPONENTS Interpreter)
        set(_lm_python "${Python3_EXECUTABLE}")
    endif()
    set(LITTLEMORE_PYTHON "${_lm_python}" CACHE FILEPATH "Python used to build the tiny test model")

    set(_lm_api_tests test-api-cpu)
    add_test(NAME test-api-cpu COMMAND test_api "${LITTLEMORE_TEST_MODEL}" cpu)
    if (GGML_CUDA)
        list(APPEND _lm_api_tests test-api-cuda)
        add_test(NAME test-api-cuda COMMAND test_api "${LITTLEMORE_TEST_MODEL}" cuda)
    endif()
    set_tests_properties(${_lm_api_tests} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)

    if (LITTLEMORE_PYTHON)
        add_test(NAME littlemore-tiny-gguf
                 COMMAND "${LITTLEMORE_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/c/make_tiny_gguf.py" "${LITTLEMORE_TEST_MODEL}"
                 WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
        set_tests_properties(littlemore-tiny-gguf PROPERTIES FIXTURES_SETUP lm_tiny_gguf TIMEOUT 900)
        set_tests_properties(${_lm_api_tests} PROPERTIES FIXTURES_REQUIRED lm_tiny_gguf)
    endif()

    # The CUDA plugin links the CUDA runtime DLLs of the toolkit it was built with; put that toolkit's bin
    # directory first on PATH so it is found before any other installed CUDA version.
    if (GGML_CUDA AND DEFINED CUDAToolkit_BIN_DIR)
        set_tests_properties(${_lm_api_tests} PROPERTIES
            ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}")
    endif()
endif()
