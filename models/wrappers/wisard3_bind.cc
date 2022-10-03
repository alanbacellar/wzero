#include <atomic>

#include "../cc/utils/array.h"
#include "utils/array_conversion.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "../cc/wisard3.cc"

using namespace std;

class BindWiSARD3 : public WiSARD3 {
public:

    BindWiSARD3(int input_lenght, int tuple_lenght, int num_classes)
    : WiSARD3(input_lenght, tuple_lenght, num_classes) {};

    void train(py::array_t<bool>& input_array, py::array_t<int>& classes_array) {
        ArrayND<bool> input = to_array_nd<bool>(input_array);
        ArrayND<int> classes = to_array_nd<int>(classes_array);
        WiSARD3::train(input, classes);
    };

    py::array_t<int> predict(py::array_t<bool>& input_array) {
        ArrayND<bool> input = to_array_nd<bool>(input_array);
        ArrayND<atomic<int>> output = WiSARD3::predict(input);
        return to_np_array<atomic<int>, int>(output);
    };

};