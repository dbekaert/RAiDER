// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Author: Rohan Weeden
// Copyright 2020, by the California Institute of Technology. ALL RIGHTS
// RESERVED. United States Government Sponsorship acknowledged.
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iterator>
#include <optional>

#include "sys/types.h"
#include "assert.h"

#ifndef PY_FAST_INTERP_H
#define PY_FAST_INTERP_H

namespace py = pybind11;

template<typename RAIter>
inline size_t bisect_left(RAIter begin, RAIter end, double x) {
    size_t left = 0;
    size_t right = std::distance(begin, end);

    while (right != left) {
        size_t mid = (left + right) / 2;
        if (x < begin[mid]) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }

    return right;
}

// Like bisect_left but optimized for when we expect the target to appear near
// the start of the array.
//
// Tries a small scan first and then switches to bisection
template<typename RAIter>
inline size_t find_left(RAIter begin, RAIter end, double x) {
    size_t left = 0;
    size_t data_N = std::distance(begin, end);

    for (; left < 5; left++) {
        if (left == data_N || x < begin[left]) {
            return left;
        }
    }

    return bisect_left(begin + left, end, x) + left;
}

// TODO: Don't store grid points as an array, just derive them from a formula?
// TODO: Same for interpolation points?
template<typename T, typename RAIter>
void interpolate_1d(
    const RAIter data_xs,
    size_t data_N,
    const RAIter data_ys,
    const RAIter xs,
    RAIter out,
    size_t N,
    std::optional<double> fill_value,
    bool assume_sorted
) {
    size_t lo = 0;
    for (size_t i = 0; i < N; i++) {
        T x = xs[i];
        size_t hi;
        if (assume_sorted) {
            hi = find_left(data_xs + lo, data_xs + data_N, x) + lo;
        } else {
            hi = bisect_left(data_xs, data_xs + data_N, x);
        }
        if (hi < 1) {
            if (fill_value.has_value()) {
                out[i] = *fill_value;
                continue;
            }
            hi = 1;
        } else if (hi > data_N - 1) {
            if (fill_value.has_value()) {
                out[i] = *fill_value;
                continue;
            }
            hi = data_N - 1;
        }

        lo = hi - 1;

        T   x0 = data_xs[lo],
            x1 = data_xs[hi],
            // Output
            y0 = data_ys[lo],
            y1 = data_ys[hi];

        T slope = (y1 - y0) / (x1 - x0);
        out[i] = y0 + slope * (x - x0);
    }
}

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

// Helper for handling the striding required to iterate along a given axis
template<typename T>
class axis_iterator {
    const unsigned char *data;
    size_t ndim;
    size_t axis;
    size_t axis_stride;
    size_t offset_base = 0;
    size_t position = 0;
public:
    axis_iterator(
        const T *data,
        size_t ndim,
        size_t axis,
        const size_t * index,
        const ssize_t * strides
    ) : data((unsigned char*) data), ndim(ndim), axis(axis) {
        assert(axis < ndim);
        axis_stride = strides[axis];
        for (size_t dim = 0; dim < ndim; dim++) {
            if (dim != axis) {
                offset_base += index[dim] * strides[dim];
            }
        }
    }

    axis_iterator(
        const py::array_t<T> &array,
        size_t axis,
        const size_t * index
    ) : axis_iterator(
            array.data(),
            (size_t) array.ndim(),
            axis,
            index,
            array.strides()
        ) { assert(array.ndim() > 0); }

    axis_iterator(
        const T *data,
        size_t ndim,
        size_t axis,
        size_t axis_stride,
        size_t offset_base
    ) : data((unsigned char*) data),
        ndim(ndim),
        axis(axis),
        axis_stride(axis_stride),
        offset_base(offset_base) { }

    axis_iterator& operator++() {
        position += 1;
        return *this;
    }
    axis_iterator operator++(int) {
        axis_iterator retval = *this;
        ++(*this);
        return retval;
    }
    axis_iterator& operator+=(size_t n) {
        this->position += n;
        return *this;
    }
    axis_iterator operator+(size_t n) const {
        axis_iterator next(*this);
        next += n;
        return next;
    }
    long operator-(axis_iterator &other) const {
        return ((ssize_t) position) - other.position;
    }
    bool operator==(axis_iterator &other) const {
        return data == other.data
            && ndim == other.ndim
            && axis == other.axis
            && axis_stride == other.axis_stride
            && offset_base == other.offset_base
            && position == other.position;
    }
    bool operator!=(axis_iterator &other) const {
        return !(*this == other);
    }
    T operator*() {
        return (*this)[position];
    }
    T & operator[](const ssize_t &n) const {
        size_t offset = offset_base + (position + n) * axis_stride;
        return *((T*) (data + offset));
    }
    // iterator traits
    using difference_type = long;
    using value_type = T;
    using pointer = const T*;
    using reference = const T&;
    using iterator_category = std::random_access_iterator_tag;
};

void interpolate_1d_along_axis(
    const py::buffer_info grid,
    const py::buffer_info values,
    const py::buffer_info interp_points,
    py::buffer_info out,
    size_t axis,
    std::optional<double> fill_value,
    bool assume_sorted
);

#endif
