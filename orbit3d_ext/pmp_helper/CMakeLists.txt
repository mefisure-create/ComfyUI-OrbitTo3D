cmake_minimum_required(VERSION 3.16)
project(orbit3d_pmp_polyline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CGAL REQUIRED)
find_package(OpenMP)

add_executable(orbit3d_pmp_polyline orbit3d_pmp_polyline.cpp)
target_link_libraries(orbit3d_pmp_polyline PRIVATE CGAL::CGAL)

if(OpenMP_CXX_FOUND)
    target_link_libraries(orbit3d_pmp_polyline PRIVATE OpenMP::OpenMP_CXX)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(orbit3d_pmp_polyline PRIVATE -O3 -DNDEBUG)
endif()
