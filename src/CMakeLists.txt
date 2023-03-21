file(GLOB SRC_FILES "*.cpp" "*.cc")
add_library(gptpu ${SRC_FILES})

target_include_directories(gptpu PUBLIC ../include)
target_link_libraries(gptpu model_utils)
add_dependencies(gptpu model_utils)
