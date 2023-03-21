file(GLOB SRC_FILES "*.cpp" "*.cc")
add_library(model_utils ${SRC_FILES})

target_include_directories(model_utils PUBLIC include)
target_link_libraries(model_utils ${TF_LITE_LIB})
add_dependencies(model_utils tf)



