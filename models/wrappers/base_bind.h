#ifndef BASE_BIND_H
#define BASE_BIND_H

#include <atomic>

#include "../cc/utils/array.h"
#include "utils/array_conversion.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

using namespace std;

// template <class WISARD_TYPE, class OUTPUT_VALUE_TYPE>
// class BindBase : public WISARD_TYPE {
// public:

//     BindBase(int input_lenght, int tuple_lenght, int num_classes, bool canonical)
//     : WISARD_TYPE(input_lenght, tuple_lenght, num_classes, canonical) {};

//     void train(py::array_t<bool>& input_array, py::array_t<int>& classes_array) {
//         ArrayND<bool> input = to_array_nd<bool>(input_array);
//         ArrayND<int> classes = to_array_nd<int>(classes_array);
//         WISARD_TYPE::train(input, classes);
//     };

//     py::array_t<OUTPUT_VALUE_TYPE> predict(py::array_t<bool>& input_array) {
//         ArrayND<bool> input = to_array_nd<bool>(input_array);
//         ArrayND<atomic<OUTPUT_VALUE_TYPE>> output = WISARD_TYPE::predict(input);
//         return to_np_array<atomic<OUTPUT_VALUE_TYPE>, OUTPUT_VALUE_TYPE>(output);
//     };

//     uint64_t get_size() {
//         return WISARD_TYPE::get_size();
//     };

//     py::array_t<OUTPUT_VALUE_TYPE> mental_images() {
//         ArrayND<atomic<OUTPUT_VALUE_TYPE>> output = WISARD_TYPE::mental_images();
//         return to_np_array<atomic<OUTPUT_VALUE_TYPE>, OUTPUT_VALUE_TYPE>(output);
//     };

//     void clear() {
//         WISARD_TYPE::clear();
//     };

// };


template <class WISARD_TYPE, class OUTPUT_VALUE_TYPE>
class BindBase : public WISARD_TYPE {
public:

    template <typename... T>
    BindBase(T... args)
    : WISARD_TYPE(args...) {};

    void train(py::array_t<bool>& input_array, py::array_t<int>& classes_array) {
        ArrayND<bool> input = to_array_nd<bool>(input_array);
        ArrayND<int> classes = to_array_nd<int>(classes_array);
        WISARD_TYPE::train(input, classes);
    };

    py::array_t<OUTPUT_VALUE_TYPE> predict(py::array_t<bool>& input_array) {
        ArrayND<bool> input = to_array_nd<bool>(input_array);
        ArrayND<atomic<OUTPUT_VALUE_TYPE>> output = WISARD_TYPE::predict(input);
        return to_np_array<atomic<OUTPUT_VALUE_TYPE>, OUTPUT_VALUE_TYPE>(output);
    };

    uint64_t get_size() {
        return WISARD_TYPE::get_size();
    };

    py::array_t<OUTPUT_VALUE_TYPE> mental_images() {
        ArrayND<atomic<OUTPUT_VALUE_TYPE>> output = WISARD_TYPE::mental_images();
        return to_np_array<atomic<OUTPUT_VALUE_TYPE>, OUTPUT_VALUE_TYPE>(output);
    };

    void clear() {
        WISARD_TYPE::clear();
    };

};

#endif