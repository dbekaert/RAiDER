#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "sys/types.h"

#ifndef PY_FAST_INTERP_H
#define PY_FAST_INTERP_H

namespace py = pybind11;

// TODO: Don't store grid points as an array, just derive them from a formula?
// TODO: Same for interpolation points?
void interpolate_1d(
    double * data_xs,
    size_t data_N,
    double * data_ys,
    double * xs,
    double * out,
    size_t N,
    bool assume_sorted
);

void interpolate_2d(
    double * data_xs,
    size_t data_x_N,
    double * data_ys,
    size_t data_y_N,
    double * data_zs,
    double * interpolation_points,
    double * out,
    size_t N,
    bool assume_sorted
);

void interpolate_3d(
    double * data_xs,
    size_t data_x_N,
    double * data_ys,
    size_t data_y_N,
    double * data_zs,
    size_t data_z_N,
    double * data_ws,
    double * interpolation_points,
    double * out,
    size_t N,
    bool assume_sorted
);

template <typename T>
struct slice {
    size_t size;
    T * ptr;
};

// Any dimension, but slower
void interpolate(
    const std::vector<slice<double>> &grid,
    const slice<double> &values,
    const slice<double> &interpolation_points,
    slice<double> &out,
    bool assume_sorted
);

template <typename T>
inline size_t offset(const ssize_t *strides, const std::vector<size_t> &index) {
    size_t offset = 0;
    for (size_t dim = 0; dim < index.size(); dim++) {
        offset += index[dim] * strides[dim];
    }
    return offset / sizeof(T);
}

void interpolate_1d_along_axis(
    const py::array_t<double> points,
    const py::array_t<double> values,
    const py::array_t<double> interp_points,
    py::array_t<double> out,
    size_t axis,
    bool assume_sorted
);

#endif
