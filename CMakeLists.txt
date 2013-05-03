cmake_minimum_required (VERSION 2.8)
project (PPSO)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-std=c++11 -O3")
endif()

add_subdirectory(lib)
add_subdirectory(demo)

