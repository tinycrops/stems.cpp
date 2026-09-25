cmake_minimum_required(VERSION 3.20)
project(stems_cpp LANGUAGES C CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Put our binaries next to the ggml DLLs so tools run without PATH juggling.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Backend toggles (forwarded to the ggml submodule).
option(STEMS_CUDA   "Build the GGML CUDA backend"     OFF)
option(STEMS_VULKAN "Build the GGML Vulkan backend"   OFF)
option(STEMS_METAL  "Build the GGML Metal backend"    OFF)
option(STEMS_HIP    "Build the GGML HIP/ROCm backend" OFF)

option(STEMS_BUILD_TOOLS "Build the CLI tools and tests" ON)
option(STEMS_STATIC "Build libstems as a static library (for embedding in an app binary)" OFF)

# ggml is linked statically into everything, including libstems.so/.dll, so a host ships one
# file. That needs position-independent objects throughout.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(STEMS_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    # CUDA graphs need Ampere+ (CC >= 8.0). Only force them off when we can PROVE the build
    # targets exclusively pre-Ampere GPUs: every CMAKE_CUDA_ARCHITECTURES entry is a plain
    # number < 80. Any non-numeric token (native/all/all-major) means unknown at configure
    # time, so we leave the ggml default alone.
    if(CMAKE_CUDA_ARCHITECTURES)
        set(_stems_all_pre_ampere ON)
        set(_stems_any_numeric OFF)
        foreach(_arch ${CMAKE_CUDA_ARCHITECTURES})
            string(REGEX REPLACE "-(real|virtual)$" "" _arch_num "${_arch}")
            if(_arch_num MATCHES "^[0-9]+$")
                set(_stems_any_numeric ON)
                if(_arch_num GREATER_EQUAL 80)
                    set(_stems_all_pre_ampere OFF)
                    break()
                endif()
            else()
                set(_stems_all_pre_ampere OFF)
                break()
            endif()
        endforeach()
        if(_stems_any_numeric AND _stems_all_pre_ampere)
            set(GGML_CUDA_GRAPHS OFF CACHE BOOL "" FORCE)
            message(STATUS "STEMS_CUDA: CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} is all pre-Ampere; forcing GGML_CUDA_GRAPHS=OFF")
        endif()
    endif()
endif()
if(STEMS_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
if(STEMS_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()
if(STEMS_HIP)
    set(GGML_HIP ON CACHE BOOL "" FORCE)
endif()

# stems.cpp pins the same revision of the betweentwomidnights/ggml fork as sa3.cpp and
# audiocraft.cpp, so every gary model links against one ggml.
# Fresh checkouts not cloned with --recurse-submodules have an empty ggml/; initialize it here so
# clean configure runs do not depend on out-of-tree copies.
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt")
    find_package(Git QUIET)
    if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
        message(STATUS "Initializing ggml submodule")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" submodule update --init --recursive ggml
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt")
        message(FATAL_ERROR
            "ggml checkout is missing. Run `git submodule update --init --recursive ggml` "
            "from ${CMAKE_CURRENT_SOURCE_DIR} and reconfigure.")
    endif()
endif()

# Existing clones can retain an older submodule checkout after a parent-repository update.
# Detect it without modifying the checkout: a maintainer may have deliberate local ggml work.
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml/.git")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD:ggml
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE STEMS_GGML_PIN_RESULT OUTPUT_VARIABLE STEMS_GGML_PIN
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C ggml rev-parse HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE STEMS_GGML_HEAD_RESULT OUTPUT_VARIABLE STEMS_GGML_HEAD
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(STEMS_GGML_PIN_RESULT EQUAL 0 AND STEMS_GGML_HEAD_RESULT EQUAL 0 AND
       NOT STEMS_GGML_PIN STREQUAL STEMS_GGML_HEAD)
        message(WARNING
            "ggml is checked out at ${STEMS_GGML_HEAD}, but this stems.cpp revision pins "
            "${STEMS_GGML_PIN}. Run `git submodule sync --recursive` then "
            "`git submodule update --init --recursive`. Local ggml work was not modified.")
    endif()
endif()
add_subdirectory(ggml)

add_library(stems STATIC src/htdemucs.cpp)
target_include_directories(stems PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(stems PUBLIC ggml)
set_target_properties(stems PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- libstems: C ABI library (embed stems in a host app; see src/libstems_v1.h) -------------
# Windows: stems.dll + stems.lib (import lib).  Linux/macOS: libstems.so / libstems.dylib.
# STEMS_STATIC (forced on for iOS) builds libstems.a instead, to link straight into an app.
# Only stems_get_api is exported: ggml is linked in statically and kept private, so a host
# that also loads sa3 or audiocraft (each with its own ggml) cannot have symbols cross over.
if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(STEMS_STATIC ON CACHE BOOL "" FORCE)
endif()
if(STEMS_STATIC)
    add_library(stems_shared STATIC src/libstems.cpp)
else()
    add_library(stems_shared SHARED src/libstems.cpp)
    set_target_properties(stems_shared PROPERTIES C_VISIBILITY_PRESET hidden
                          CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
    if(APPLE)
        target_link_options(stems_shared PRIVATE "LINKER:-exported_symbol,_stems_get_api")
    elseif(UNIX)
        target_link_options(stems_shared PRIVATE "LINKER:--exclude-libs,ALL")
    endif()
endif()
set_target_properties(stems_shared PROPERTIES OUTPUT_NAME stems)
target_link_libraries(stems_shared PRIVATE stems)

if(STEMS_BUILD_TOOLS)
    add_executable(stems-split tools/stems-split.cpp)
    target_link_libraries(stems-split PRIVATE stems)

    # Pure C on purpose: proves the public header is C-clean and links through the ABI only.
    add_executable(stems-libtest tools/stems-libtest.c)
    target_include_directories(stems-libtest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(stems-libtest PRIVATE stems_shared)

    add_executable(stems-parity tools/stems-parity.cpp)
    target_link_libraries(stems-parity PRIVATE stems)

    # Vendored cpp-httplib (split implementation) and yyjson (C), warnings off.
    find_package(Threads REQUIRED)
    add_executable(stems-server tools/stems-server.cpp
        vendor/cpp-httplib/httplib.cpp
        vendor/yyjson/yyjson.c)
    target_link_libraries(stems-server PRIVATE stems Threads::Threads)
    target_include_directories(stems-server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/cpp-httplib
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/yyjson)
    if(MSVC)
        set_source_files_properties(vendor/yyjson/yyjson.c vendor/cpp-httplib/httplib.cpp
                                    PROPERTIES COMPILE_FLAGS /w)
    else()
        set_source_files_properties(vendor/yyjson/yyjson.c vendor/cpp-httplib/httplib.cpp
                                    PROPERTIES COMPILE_FLAGS -w)
    endif()
    if(WIN32)
        target_link_libraries(stems-server PRIVATE ws2_32)
    endif()
endif()
