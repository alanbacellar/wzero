#include "base_bind.h"
#include "../cc/utils/array.h"
#include "utils/array_conversion.h"

#include "../cc/bleaching_bloom_wisard.cc"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

class BindBleachingBloomWiSARD : public BindBase<BleachingBloomWiSARD, int> {
public:

    BindBleachingBloomWiSARD(int input_lenght, int tuple_lenght, int num_filters, int filter_tuple_lenght, int num_classes, bool canonical)
    : BindBase(input_lenght, tuple_lenght, num_filters, filter_tuple_lenght, num_classes, canonical) {};

    py::array_t<int> predictb(py::array_t<bool>& input_array, int bleaching) {
        ArrayND<bool> input = to_array_nd<bool>(input_array);
        ArrayND<atomic<int>> output = BleachingBloomWiSARD::predictb(input, bleaching);
        return to_np_array<atomic<int>, int>(output);
    };

};