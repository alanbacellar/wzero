#include "base_bind.h"
#include "../cc/utils/array.h"
#include "utils/array_conversion.h"

#include "../cc/bleaching_wisard15.cc" 

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;


class BindBleachingWiSARD15 : public BindBase<BleachingWiSARD15, int> {
public:

    BindBleachingWiSARD15(int input_lenght, int tuple_lenght, int num_classes, bool canonical)
    : BindBase(input_lenght, tuple_lenght, num_classes, canonical) {};

    py::array_t<int> predict(py::array_t<bool>& input_array) {
        ArrayND<bool> input = to_array_nd<bool>(input_array);
        ArrayND<atomic<int>> output = BleachingWiSARD15::predict15(input);
        return to_np_array<atomic<int>, int>(output);
    };

};