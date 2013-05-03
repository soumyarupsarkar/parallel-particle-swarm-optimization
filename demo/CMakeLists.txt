include_directories (${PPSO_SOURCE_DIR}/include)
link_directories (${PPSO_SOURCE_DIR}/lib)

add_executable (demo demo.cpp)
target_link_libraries (demo ppso pthread tcmalloc)

