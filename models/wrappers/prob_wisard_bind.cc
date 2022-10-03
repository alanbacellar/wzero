#include "base_bind.h"
#include "../cc/utils/array.h"
#include "utils/array_conversion.h"

#include "../cc/prob_wisard.cc" 

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;


class BindProbWiSARD : public BindBase<ProbWiSARD, int> {
public:

    BindProbWiSARD(int input_lenght, int tuple_lenght, int num_classes, bool canonical)
    : BindBase(input_lenght, tuple_lenght, num_classes, canonical) {};

    py::array_t<float> predict(py::array_t<bool>& input_array) {
        ArrayND<bool> input = to_array_nd<bool>(input_array);
        ArrayND<float> output = ProbWiSARD::predict2(input);
        return to_np_array<float, float>(output);
    };

};