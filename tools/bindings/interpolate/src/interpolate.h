#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iterator>

#include "sys/types.h"

#ifndef PY_FAST_INTERP_H
#define PY_FAST_INTERP_H

namespace py = pybind11;

template<typename RAIter>
inline size_t bisect_left(RAIter begin, RAIter end, double x) {
    size_t left = 0;
    size_t right = std::distance(begin, end) - 1;

    while (right - left > 1) {
        size_t mid = left + (right - left) / 2;
        if (x < begin[mid]) {
            right = mid;
        } else {
            left = mid;
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
        axis_stride = strides[axis];
        for (size_t dim = 0; dim < ndim; dim++) {
            if (dim != axis) {
                offset_base += index[dim] * strides[dim];
            }
        }
    }

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
    const py::array_t<double> points,
    const py::array_t<double> values,
    const py::array_t<double> interp_points,
    py::array_t<double> out,
    size_t axis,
    bool assume_sorted
);

#endif
