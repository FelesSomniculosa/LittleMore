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
# Install (library, header, backend plugins, a CMake package): cmake --install build --prefix <dir>

# 3.22: the tests use the ENVIRONMENT_MODIFICATION test property (new in 3.22).
cmake_minimum_required(VERSION 3.22)

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
# The ABI version (LM_API_VERSION) is the shared library's SOVERSION: it changes exactly when binaries built
# against an older header stop working, which is what SOVERSION means.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/littlemore.h" _lm_api_line REGEX "^#define LM_API_VERSION +[0-9]+")
if (NOT _lm_api_line MATCHES "^#define LM_API_VERSION +([0-9]+)")
    message(FATAL_ERROR "could not read LM_API_VERSION from include/littlemore.h")
endif()
set(_lm_API ${CMAKE_MATCH_1})

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
include(GNUInstallDirs)
add_subdirectory(engine)

# ---------------------------------------------------------------------------------------------------------------
# The C API library.
# ---------------------------------------------------------------------------------------------------------------
add_library(littlemore SHARED
    include/littlemore.h
    src/littlemore.cpp
    src/utf8_boundary.h)
add_library(littlemore::littlemore ALIAS littlemore)
target_include_directories(littlemore PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
# PRIVATE: the public header exposes no engine type, so users of littlemore never need llama.h or ggml.h.
target_link_libraries(littlemore PRIVATE llama ggml ggml-base)
# LM_BUILD selects dllexport (Windows) / default visibility (elsewhere) in the header's LM_API macro.
target_compile_definitions(littlemore PRIVATE LM_BUILD)
set_target_properties(littlemore PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${_lm_API})
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
# Install: the library, its header and a CMake package (find_package(littlemore) -> littlemore::littlemore).
# The engine's own install rules (added by add_subdirectory) install llama, ggml, ggml-base and every backend
# plugin; the plugins go to ${CMAKE_INSTALL_BINDIR} (ggml's default when GGML_BACKEND_DIR is not set). The
# library loads its plugins from its own directory, so it is installed into that same directory on every
# platform (on Windows a DLL goes to bin anyway; on Linux/macOS this puts the .so/.dylib next to the plugins).
# Only the Windows layout is verified so far.
# ---------------------------------------------------------------------------------------------------------------
install(TARGETS littlemore
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES include/littlemore.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# The package file is written by hand instead of with install(EXPORT): an exported shared library would have to
# export its private engine dependencies (llama, ggml) too, which the engine does not provide as an export set.
# The imported target needs only the header, the library and (Windows) its import library.
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/littlemoreConfig.cmake" CONTENT
"# littlemoreConfig.cmake - CMake package of the LittleMore C API (generated by LittleMore's CMakeLists.txt).
# Defines the imported target littlemore::littlemore (header include/littlemore.h, shared library littlemore).
get_filename_component(_lm_prefix \"\${CMAKE_CURRENT_LIST_DIR}/../../..\" ABSOLUTE)
if (NOT TARGET littlemore::littlemore)
    add_library(littlemore::littlemore SHARED IMPORTED)
    set_target_properties(littlemore::littlemore PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES \"\${_lm_prefix}/${CMAKE_INSTALL_INCLUDEDIR}\"
        IMPORTED_LOCATION \"\${_lm_prefix}/${CMAKE_INSTALL_BINDIR}/$<TARGET_FILE_NAME:littlemore>\"
        IMPORTED_IMPLIB \"\${_lm_prefix}/${CMAKE_INSTALL_LIBDIR}/$<TARGET_LINKER_FILE_NAME:littlemore>\")
endif()
unset(_lm_prefix)
")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/littlemoreConfig.cmake" DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/littlemore)
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/littlemoreConfigVersion.cmake"
                                 VERSION ${PROJECT_VERSION} COMPATIBILITY SameMinorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/littlemoreConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/littlemore)

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

    # Unit test of the UTF-8 boundary rule of the streaming code (no model, no library needed).
    add_executable(test_utf8 tests/c/test_utf8.c)
    set_source_files_properties(tests/c/test_utf8.c PROPERTIES LANGUAGE C)
    add_test(NAME test-utf8 COMMAND test_utf8)

    # The same API test, but run from a copy of test_api in another directory with that directory as the
    # working directory: there are no plugins next to the executable or in the current directory (ggml's own
    # default search places), so it passes only if littlemore loads the plugins from ITS OWN directory.
    set(_lm_elsewhere "${CMAKE_BINARY_DIR}/elsewhere")
    add_custom_command(TARGET test_api POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_lm_elsewhere}"
        COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:test_api>" "${_lm_elsewhere}/"
        COMMENT "copying test_api to ${_lm_elsewhere} (plugin directory test)")

    set(_lm_api_tests test-api-cpu test-api-plugin-dir)
    add_test(NAME test-api-cpu COMMAND test_api "${LITTLEMORE_TEST_MODEL}" cpu)
    add_test(NAME test-api-plugin-dir COMMAND "${_lm_elsewhere}/test_api${CMAKE_EXECUTABLE_SUFFIX}" "${LITTLEMORE_TEST_MODEL}" cpu
             WORKING_DIRECTORY "${_lm_elsewhere}")
    if (GGML_CUDA)
        list(APPEND _lm_api_tests test-api-cuda)
        add_test(NAME test-api-cuda COMMAND test_api "${LITTLEMORE_TEST_MODEL}" cuda)
    endif()
    # TIMEOUT 120: a correct run takes about a second; a deadlock (e.g. a broken re-entrancy guard) must fail
    # the test instead of hanging the suite.
    set_tests_properties(${_lm_api_tests} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)

    if (LITTLEMORE_PYTHON)
        add_test(NAME littlemore-tiny-gguf
                 COMMAND "${LITTLEMORE_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/c/make_tiny_gguf.py" "${LITTLEMORE_TEST_MODEL}"
                 WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
        set_tests_properties(littlemore-tiny-gguf PROPERTIES FIXTURES_SETUP lm_tiny_gguf TIMEOUT 900)
        set_tests_properties(${_lm_api_tests} PROPERTIES FIXTURES_REQUIRED lm_tiny_gguf)
    endif()

    # The CUDA plugin links the CUDA runtime DLLs of the toolkit it was built with; put that toolkit's bin
    # directory first on PATH so it is found before any other installed CUDA version. (CUDAToolkit_BIN_DIR is a
    # cache variable set by the engine's find_package(CUDAToolkit), so it is visible here.) The plugin-directory
    # test also needs build/bin on PATH, to find littlemore itself from its copied executable.
    set(_lm_env_cuda "")
    if (GGML_CUDA AND DEFINED CUDAToolkit_BIN_DIR)
        set(_lm_env_cuda "PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}")
    endif()
    foreach(_t IN LISTS _lm_api_tests)
        set(_env ${_lm_env_cuda})
        if (_t STREQUAL "test-api-plugin-dir")
            list(APPEND _env "PATH=path_list_prepend:$<TARGET_FILE_DIR:littlemore>")
        endif()
        if (_env)
            set_tests_properties(${_t} PROPERTIES ENVIRONMENT_MODIFICATION "${_env}")
        endif()
    endforeach()
endif()
