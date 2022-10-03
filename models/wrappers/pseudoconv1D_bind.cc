#include <atomic>

#include "../cc/utils/array.h"
#include "utils/array_conversion.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "../cc/pseudoconv1D.cc"

#include <iostream>

using namespace std;

class BindPC1D : public WiSARDPC1D {
public:

    BindPC1D(int x_dim, int z_dim, int window_size, int stride, int tuple_lenght, int num_classes)
    : WiSARDPC1D(x_dim, z_dim, window_size, stride, tuple_lenght, num_classes) {};

    void train(py::array_t<bool>& input_array, py::array_t<int>& classes_array) {
        ArrayND<bool> input = to_array_nd<bool>(input_array);
        ArrayND<int> classes = to_array_nd<int>(classes_array);
        WiSARDPC1D::train(input, classes);
    };

    py::array_t<int> predict(py::array_t<bool>& input_array) {
        ArrayND<bool> input = to_array_nd<bool>(input_array);
        ArrayND<atomic<int>> output = WiSARDPC1D::predict(input);
        return to_np_array<atomic<int>, int>(output);
    };

};