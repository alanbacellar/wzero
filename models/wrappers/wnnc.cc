#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "wisard_bind.cc"
#include "bleaching_wisard_bind.cc"
#include "bloom_wisard_bind.cc"
#include "bleaching_bloom_wisard_bind.cc"
#include "regression_wisard_bind.cc"

#include "wisard2_bind.cc"
#include "wisard3_bind.cc"

#include "bleaching_wisard2_bind.cc"
#include "bleaching_wisard15_bind.cc"

#include "pseudoconv1D_bind.cc"
#include "pseudoconv2D_bind.cc"

#include "bloom_ram_wisard_bind.cc"

#include "bloom_wisard2_bind.cc"
#include "bloom_wisard3_bind.cc"
#include "bloom_wisard4_bind.cc"
#include "bloom_wisard5_bind.cc"
#include "bloom_wisard55_bind.cc"

#include "prob_wisard_bind.cc"

namespace py = pybind11;

PYBIND11_MODULE(wnnc, m) {

    py::class_<BindWiSARD>(m, "CcWiSARD")
        .def(py::init<int, int, int, bool>())
        .def("train", (void (BindWiSARD::*)(py::array_t<bool>&, py::array_t<int>&)) &BindWiSARD::train)
        .def("predict", (py::array_t<int> (BindWiSARD::*)(py::array_t<bool>&)) &BindWiSARD::predict)
        .def("mental_images", (py::array_t<float> (BindWiSARD::*)(void)) &BindWiSARD::mental_images)
        .def("clear", (void (BindWiSARD::*)(void)) &BindWiSARD::clear)
        .def("get_size", (int (BindWiSARD::*)(void)) &BindWiSARD::get_size)
    ;

    py::class_<BindBleachingWiSARD>(m, "CcBleachingWiSARD")
        .def(py::init<int, int, int, bool>())
        .def("train", (void (BindBleachingWiSARD::*)(py::array_t<bool>&, py::array_t<int>&)) &BindBleachingWiSARD::train)
        .def("predict", (py::array_t<int> (BindBleachingWiSARD::*)(py::array_t<bool>&)) &BindBleachingWiSARD::predict)
        .def("predictb", (py::array_t<int> (BindBleachingWiSARD::*)(py::array_t<bool>&, int)) &BindBleachingWiSARD::predictb)
        .def("mental_images", (py::array_t<float> (BindBleachingWiSARD::*)(void)) &BindBleachingWiSARD::mental_images)
        .def("clear", (void (BindBleachingWiSARD::*)(void)) &BindBleachingWiSARD::clear)
        .def("get_size", (int (BindBleachingWiSARD::*)(void)) &BindBleachingWiSARD::get_size)
    ;

    py::class_<BindBloomWiSARD>(m, "CcBloomWiSARD")
        .def(py::init<int, int, int, int, int, bool>())
        .def("train", (void (BindBloomWiSARD::*)(py::array_t<bool>&, py::array_t<int>&)) &BindBloomWiSARD::train)
        .def("predict", (py::array_t<int> (BindBloomWiSARD::*)(py::array_t<bool>&)) &BindBloomWiSARD::predict)
        .def("mental_images", (py::array_t<float> (BindBloomWiSARD::*)(void)) &BindBloomWiSARD::mental_images)
        .def("clear", (void (BindBloomWiSARD::*)(void)) &BindBloomWiSARD::clear)
        .def("get_size", (int (BindBloomWiSARD::*)(void)) &BindBloomWiSARD::get_size)
    ;

    py::class_<BindBleachingBloomWiSARD>(m, "CcBleachingBloomWiSARD")
        .def(py::init<int, int, int, int, int, bool>())
        .def("train", (void (BindBleachingBloomWiSARD::*)(py::array_t<bool>&, py::array_t<int>&)) &BindBleachingBloomWiSARD::train)
        .def("predict", (py::array_t<int> (BindBleachingBloomWiSARD::*)(py::array_t<bool>&)) &BindBleachingBloomWiSARD::predict)
        .def("predictb", (py::array_t<int> (BindBleachingBloomWiSARD::*)(py::array_t<bool>&, int)) &BindBleachingBloomWiSARD::predictb)
        .def("mental_images", (py::array_t<float> (BindBleachingBloomWiSARD::*)(void)) &BindBleachingBloomWiSARD::mental_images)
        .def("clear", (void (BindBleachingBloomWiSARD::*)(void)) &BindBleachingBloomWiSARD::clear)
        .def("get_size", (int (BindBleachingBloomWiSARD::*)(void)) &BindBleachingBloomWiSARD::get_size)
    ;

    py::class_<BindRegressionWiSARD>(m, "CcRegressionWiSARD")
        .def(py::init<int, int, int, bool>())
        .def("train", (void (BindRegressionWiSARD::*)(py::array_t<bool>&, py::array_t<float>&)) &BindRegressionWiSARD::train)
        .def("predict", (py::array_t<int> (BindRegressionWiSARD::*)(py::array_t<bool>&)) &BindRegressionWiSARD::predict)
        .def("mental_images", (py::array_t<float> (BindRegressionWiSARD::*)(void)) &BindRegressionWiSARD::mental_images)
        .def("clear", (void (BindRegressionWiSARD::*)(void)) &BindRegressionWiSARD::clear)
        .def("get_size", (int (BindRegressionWiSARD::*)(void)) &BindRegressionWiSARD::get_size)
    ;

    py::class_<BindWiSARD3>(m, "CcWiSARD3")
        .def(py::init<int, int, int>())
        .def("train", (void (BindWiSARD3::*)(py::array_t<bool>&, py::array_t<int>&)) &BindWiSARD3::train)
        .def("predict", (py::array_t<int> (BindWiSARD3::*)(py::array_t<bool>&)) &BindWiSARD3::predict)
    ;

    py::class_<BindWiSARD2>(m, "CcWiSARD2")
        .def(py::init<int, int, int>())
        .def("train", (void (BindWiSARD2::*)(py::array_t<bool>&, py::array_t<int>&)) &BindWiSARD2::train)
        .def("predict", (py::array_t<int> (BindWiSARD2::*)(py::array_t<bool>&)) &BindWiSARD2::predict)
    ;

    py::class_<BindBleachingWiSARD2>(m, "CcBleachingWiSARD2")
        .def(py::init<int, int, int, bool>())
        .def("train", (void (BindBleachingWiSARD2::*)(py::array_t<bool>&, py::array_t<int>&)) &BindBleachingWiSARD2::train)
        .def("predict", (py::array_t<int> (BindBleachingWiSARD2::*)(py::array_t<bool>&)) &BindBleachingWiSARD2::predict)
        .def("mental_images", (py::array_t<float> (BindBleachingWiSARD2::*)(void)) &BindBleachingWiSARD2::mental_images)
        .def("clear", (void (BindBleachingWiSARD2::*)(void)) &BindBleachingWiSARD2::clear)
        .def("get_size", (int (BindBleachingWiSARD2::*)(void)) &BindBleachingWiSARD2::get_size)
    ;

     py::class_<BindBleachingWiSARD15>(m, "CcBleachingWiSARD15")
        .def(py::init<int, int, int, bool>())
        .def("train", (void (BindBleachingWiSARD15::*)(py::array_t<bool>&, py::array_t<int>&)) &BindBleachingWiSARD15::train)
        .def("predict", (py::array_t<int> (BindBleachingWiSARD15::*)(py::array_t<bool>&)) &BindBleachingWiSARD15::predict)
        .def("mental_images", (py::array_t<float> (BindBleachingWiSARD15::*)(void)) &BindBleachingWiSARD15::mental_images)
        .def("clear", (void (BindBleachingWiSARD15::*)(void)) &BindBleachingWiSARD15::clear)
        .def("get_size", (int (BindBleachingWiSARD15::*)(void)) &BindBleachingWiSARD15::get_size)
    ;

    py::class_<BindPC1D>(m, "CcWiSARDPC1D")
        .def(py::init<int, int, int, int, int , int>())
        .def("train", (void (BindPC1D::*)(py::array_t<bool>&, py::array_t<int>&)) &BindPC1D::train)
        .def("predict", (py::array_t<int> (BindPC1D::*)(py::array_t<bool>&)) &BindPC1D::predict)
    ;

    py::class_<BindPC2D>(m, "CcWiSARDPC2D")
        .def(py::init<int, int, int, int, int, int , int>())
        .def("train", (void (BindPC2D::*)(py::array_t<bool>&, py::array_t<int>&)) &BindPC2D::train)
        .def("predict", (py::array_t<int> (BindPC2D::*)(py::array_t<bool>&)) &BindPC2D::predict)
    ;

    py::class_<BindBloomRamWiSARD>(m, "CcBloomRamWiSARD")
        .def(py::init<int, int, int, int, bool>())
        .def("train", (void (BindBloomRamWiSARD::*)(py::array_t<bool>&, py::array_t<int>&)) &BindBloomRamWiSARD::train)
        .def("predict", (py::array_t<int> (BindBloomRamWiSARD::*)(py::array_t<bool>&)) &BindBloomRamWiSARD::predict)
        .def("mental_images", (py::array_t<float> (BindBloomRamWiSARD::*)(void)) &BindBloomRamWiSARD::mental_images)
        .def("clear", (void (BindBloomRamWiSARD::*)(void)) &BindBloomRamWiSARD::clear)
        .def("get_size", (int (BindBloomRamWiSARD::*)(void)) &BindBloomRamWiSARD::get_size)
    ;

    py::class_<BindBloomWiSARD2>(m, "CcBloomWiSARD2")
        .def(py::init<int, int, int, int, int>())
        .def("train", (void (BindBloomWiSARD2::*)(py::array_t<bool>&, py::array_t<int>&)) &BindBloomWiSARD2::train)
        .def("predict", (py::array_t<int> (BindBloomWiSARD2::*)(py::array_t<bool>&)) &BindBloomWiSARD2::predict)
    ;

    py::class_<BindBloomWiSARD3>(m, "CcBloomWiSARD3")
        .def(py::init<int, int, int, int, int>())
        .def("train", (void (BindBloomWiSARD3::*)(py::array_t<bool>&, py::array_t<int>&)) &BindBloomWiSARD3::train)
        .def("predict", (py::array_t<int> (BindBloomWiSARD3::*)(py::array_t<bool>&)) &BindBloomWiSARD3::predict)
    ;

    py::class_<BindBloomWiSARD4>(m, "CcBloomWiSARD4")
        .def(py::init<int, int, int, int, int>())
        .def("train", (void (BindBloomWiSARD4::*)(py::array_t<bool>&, py::array_t<int>&)) &BindBloomWiSARD4::train)
        .def("predict", (py::array_t<int> (BindBloomWiSARD4::*)(py::array_t<bool>&)) &BindBloomWiSARD4::predict)
    ;

    py::class_<BindBloomWiSARD5>(m, "CcBloomWiSARD5")
        .def(py::init<int, int, int, int, int>())
        .def("train", (void (BindBloomWiSARD5::*)(py::array_t<bool>&, py::array_t<int>&)) &BindBloomWiSARD5::train)
        .def("predict", (py::array_t<int> (BindBloomWiSARD5::*)(py::array_t<bool>&)) &BindBloomWiSARD5::predict)
    ;

    py::class_<BindBloomWiSARD55>(m, "CcBloomWiSARD55")
        .def(py::init<int, int, int, int, int>())
        .def("train", (void (BindBloomWiSARD55::*)(py::array_t<bool>&, py::array_t<int>&)) &BindBloomWiSARD55::train)
        .def("predict", (py::array_t<int> (BindBloomWiSARD55::*)(py::array_t<bool>&)) &BindBloomWiSARD55::predict)
    ;

    py::class_<BindProbWiSARD>(m, "CcProbWiSARD")
        .def(py::init<int, int, int, bool>())
        .def("train", (void (BindProbWiSARD::*)(py::array_t<bool>&, py::array_t<int>&)) &BindProbWiSARD::train)
        .def("predict", (py::array_t<float> (BindProbWiSARD::*)(py::array_t<bool>&)) &BindProbWiSARD::predict)
    ;
    
  
};