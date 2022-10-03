#ifndef ARRAY_CONVERSION_H
#define ARRAY_CONVERSION_H

#include <vector>

#include "../../cc/utils/array.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;


template <class T>
ArrayND<T> to_array_nd(py::array_t<T>& arr) {
    py::buffer_info buf = arr.request();
    std::vector<int> dims(buf.shape.begin(), buf.shape.end());
    return ArrayND<T>((T*)buf.ptr, dims);
};

template <class T>
py::array_t<T> to_np_array(ArrayND<T>& arr) {
    
    ssize_t ndim = arr.num_dims;
    
    std::vector<ssize_t> shape(arr.shape, arr.shape + arr.num_dims);
    std::vector<ssize_t> strides;
    for(int i = 0; i < arr.num_dims; ++i)
        strides.push_back(sizeof(T) * arr.strides[i]);

    return py::array_t<T>(py::buffer_info(
        arr.data,
        sizeof(T),
        py::format_descriptor<T>::format(),
        ndim,
        shape,
        strides
    ));

};

// Used to convert between two types of same sizeof.
template <class T1, class T2>
py::array_t<T2> to_np_array(ArrayND<T1>& arr) {
    
    ssize_t ndim = arr.num_dims;
    
    std::vector<ssize_t> shape(arr.shape, arr.shape + arr.num_dims);
    std::vector<ssize_t> strides;
    for(int i = 0; i < arr.num_dims; ++i)
        strides.push_back(sizeof(T1) * arr.strides[i]);

    return py::array_t<T2>(py::buffer_info(
        arr.data,
        sizeof(T1),
        py::format_descriptor<T2>::format(),
        ndim,
        shape,
        strides
    ));

};


#endif