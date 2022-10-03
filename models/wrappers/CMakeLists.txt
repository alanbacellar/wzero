cmake_minimum_required(VERSION 3.10)

project(wnnc)

set (CMAKE_CXX_FLAGS "-std=c++17 -O3 -Wall -shared  -fPIC -march=native -fopt-info-vec-optimized ${CMAKE_CXX_FLAGS}")

# add_subdirectory(pybind11)
find_package(pybind11 REQUIRED)
pybind11_add_module(wnnc 
    wnnc.cc
    ../cc/utils/mapping.cc
    ../cc/utils/utils.cc
    ../cc/wisard.cc
    ../cc/regression_wisard.cc
    ../cc/bleaching_wisard.cc
    ../cc/bloom_wisard.cc
    ../cc/bleaching_bloom_wisard.cc
    bleaching_bloom_wisard_bind.cc
    bleaching_wisard_bind.cc
    bloom_wisard_bind.cc
    wisard_bind.cc
    ../cc/wisard2.cc
    ../cc/wisard3.cc
    ../cc/bleaching_wisard2.cc
    ../cc/bleaching_wisard15.cc
    wisard2_bind.cc
    wisard3_bind.cc
    bleaching_wisard2_bind.cc
    bleaching_wisard15_bind.cc
    ../cc/pseudoconv1D.cc
    pseudoconv1D_bind.cc
    ../cc/pseudoconv2D.cc
    pseudoconv2D_bind.cc
    ../cc/bloom_ram_wisard.cc
    bloom_ram_wisard_bind.cc
    ../cc/bloom_wisard2.cc
    bloom_wisard2_bind.cc
    ../cc/bloom_wisard3.cc
    bloom_wisard3_bind.cc
    ../cc/bloom_wisard4.cc
    bloom_wisard4_bind.cc
    ../cc/bloom_wisard5.cc
    bloom_wisard5_bind.cc
    ../cc/bloom_wisard55.cc
    bloom_wisard55_bind.cc
    ../cc/prob_wisard.cc
    prob_wisard_bind.cc
)

target_compile_definitions(wnnc PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})