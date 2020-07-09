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
    size_t N
);

void interpolate_2d(
    double * data_xs,
    size_t data_x_N,
    double * data_ys,
    size_t data_y_N,
    double * data_zs,
    double * interpolation_points,
    double * out,
    size_t N
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
    size_t N
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
    slice<double> &out
);

#endif
