cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(infer)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(inf inf.cpp)
target_link_libraries(inf "${TORCH_LIBRARIES}")
set_property(TARGET inf PROPERTY CXX_STANDARD 14)
