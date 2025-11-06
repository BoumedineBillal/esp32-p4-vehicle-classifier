# Auto-generated CMakeLists for ESP32-P4 Vehicle Classifier
# Variant: pico
# Quantization: qat
# Generated: 2025-10-24 00:16:19

set(srcs app_main.cpp)

set(requires esp-dl)

idf_build_get_property(component_targets __COMPONENT_TARGETS)
if ("___idf_espressif__esp-dl" IN_LIST component_targets)
   idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif("___idf_esp-dl" IN_LIST component_targets)
   idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)

# Embed model and test images
set(embed_files models/p4/vehicle_classifier.espdl
                not_vehicle_0.jpg
                not_vehicle_1.jpg
                vehicle_0.jpg
                vehicle_1.jpg
)

idf_component_register(SRCS ${srcs}
                       REQUIRES ${requires}
                       EMBED_FILES ${embed_files})

target_add_aligned_binary_data(${COMPONENT_LIB} models/p4/vehicle_classifier.espdl BINARY)
