# Auto-generated ESP32-P4 Vehicle Classifier Project
# Variant: current
# Quantization: qat
# Generated: 2025-10-24 01:01:30

cmake_minimum_required(VERSION 3.16)

add_compile_options(-fdiagnostics-color=always)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)  # ← MOVE THIS UP

set(DEPENDENCIES_LOCK_FILE "${CMAKE_SOURCE_DIR}/dependencies.lock.${IDF_TARGET}")
if(EXISTS ${DEPENDENCIES_LOCK_FILE})
    idf_build_set_property(DEPENDENCIES_LOCK ${DEPENDENCIES_LOCK_FILE})  # ← NOW IT WORKS
endif()

project(vehicle_current_qat)