// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Author: Rohan Weeden
// Copyright 2020, by the California Institute of Technology. ALL RIGHTS
// RESERVED. United States Government Sponsorship acknowledged.
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <future>
#include <sstream>
#include <optional>

#include "interpolate.h"


namespace py = pybind11;

PYBIND11_MODULE(interpolate, m) {
    m.doc() = "Fast linear interpolator over a regular grid";

    m.def("interpolate", [](
            std::vector<py::array_t<double, py::array::c_style>> points,
            py::array_t<double, py::array::c_style> values,
            py::array_t<double, py::array::c_style> interp_points,
            std::optional<double> fill_value,
            bool assume_sorted,
            size_t max_threads
        ) {
            size_t num_dims = points.size();

            if (values.ndim() == 0 || interp_points.ndim() == 0) {
                throw py::type_error("Only arrays are supported, not scalar values!");
            }

            for (auto arr : points) {
                if (arr.ndim() != 1) {
                    throw py::type_error("'points' must be a list of 1D arrays!");
                }
            }

            if (num_dims != values.ndim()) {
                std::stringstream ss;
                ss << "Dimension mismatch! Grid is " << num_dims
                   << "D but values are " << values.ndim() << "D!";
                throw py::type_error(ss.str());
            }

            if (interp_points.ndim() != 2) {
                throw py::type_error("'interp_points' should have shape (N, ndim).");
            }

            size_t interp_dims = interp_points.shape()[1];
            if (num_dims != interp_dims) {
                std::stringstream ss;
                ss << "Dimension mismatch! Grid is " << num_dims
                   << "D but interpolation points are " << interp_dims << "D!";
                throw py::type_error(ss.str());
            }
            size_t num_elements = interp_points.shape()[0];
            double * out = new double[num_elements];

            auto values_info = values.request();
            auto interp_points_info = interp_points.request();


            // Reasonable thread defaults based on profiling. It seems that spawning
            // threads in powers of 2 yields optimal performance.
            size_t desired_threads;
            if (num_elements < 10000) {
                desired_threads = 1;
            } else if (num_elements < 4000000) {
                desired_threads = 2;
            } else if (num_elements < 160000000){
                desired_threads = 4;
            } else {
                desired_threads = 8;
            }
            size_t num_threads = std::min(desired_threads, max_threads);
            if (num_threads == 0) {
                num_threads = 1;
            }
            size_t stride = (num_elements / num_threads);
            if (stride * num_threads < num_elements) {
                stride += 1;
            }

            double * values_ptr = (double *) values_info.ptr,
                   * interp_points_ptr = (double *) interp_points_info.ptr;

            if (num_dims == 1) {
                auto xs_info = points[0].request();

                double * xs_ptr = (double *) xs_info.ptr;

                if (num_threads == 1) {
                    interpolate_1d<double>(
                        xs_ptr,
                        points[0].size(),
                        values_ptr,
                        interp_points_ptr,
                        out,
                        num_elements,
                        fill_value,
                        assume_sorted
                    );
                } else {
                    std::vector<std::future<void>> tasks;

                    for (size_t i = 0; i < num_threads; i++) {
                        size_t index = i * stride;
                        tasks.push_back(
                            std::async(
                                &interpolate_1d<double, double *>,
                                xs_ptr,
                                xs_info.shape[0],
                                values_ptr,
                                &interp_points_ptr[index],
                                &out[index],
                                index + stride < num_elements ? stride : num_elements - index,
                                fill_value,
                                assume_sorted
                            )
                        );
                    }
                    for (auto &future : tasks) {
                        std::move(future);
                    }
                }
            } else if (num_dims == 2) {
                auto xs_info = points[0].request();
                auto ys_info = points[1].request();

                double * xs_ptr = (double *) xs_info.ptr,
                       * ys_ptr = (double *) ys_info.ptr;

                if (num_threads == 1) {
                    interpolate_2d(
                        xs_ptr,
                        points[0].size(),
                        ys_ptr,
                        points[1].size(),
                        values_ptr,
                        interp_points_ptr,
                        out,
                        num_elements,
                        fill_value,
                        assume_sorted
                    );
                } else {
                    std::vector<std::future<void>> tasks;

                    for (size_t i = 0; i < num_threads; i++) {
                        size_t index = i * stride;
                        tasks.push_back(
                            std::async(
                                &interpolate_2d,
                                xs_ptr,
                                points[0].size(),
                                ys_ptr,
                                points[1].size(),
                                values_ptr,
                                &interp_points_ptr[index * num_dims],
                                &out[index],
                                index + stride < num_elements ? stride : num_elements - index,
                                fill_value,
                                assume_sorted
                            )
                        );
                    }
                    for (auto &future : tasks) {
                        std::move(future);
                    }
                }
            } else if (num_dims == 3) {
                auto xs_info = points[0].request();
                auto ys_info = points[1].request();
                auto zs_info = points[2].request();

                double * xs_ptr = (double *) xs_info.ptr,
                       * ys_ptr = (double *) ys_info.ptr,
                       * zs_ptr = (double *) zs_info.ptr;

                if (num_threads == 1) {
                    interpolate_3d(
                        xs_ptr,
                        points[0].size(),
                        ys_ptr,
                        points[1].size(),
                        zs_ptr,
                        points[2].size(),
                        values_ptr,
                        interp_points_ptr,
                        out,
                        num_elements,
                        fill_value,
                        assume_sorted
                    );
                } else {
                    std::vector<std::future<void>> tasks;

                    for (size_t i = 0; i < num_threads; i++) {
                        size_t index = i * stride;
                        tasks.push_back(
                            std::async(
                                &interpolate_3d,
                                xs_ptr,
                                points[0].size(),
                                ys_ptr,
                                points[1].size(),
                                zs_ptr,
                                points[2].size(),
                                values_ptr,
                                &interp_points_ptr[index * num_dims],
                                &out[index],
                                index + stride < num_elements ? stride : num_elements - index,
                                fill_value,
                                assume_sorted
                            )
                        );
                    }
                    for (auto &future : tasks) {
                        std::move(future);
                    }
                }
            } else {
                std::vector<slice<double>> grid;
                for (auto axis : points) {
                    auto info = axis.request();
                    grid.push_back(slice<double> {
                        (size_t) info.shape[0],
                        (double *) info.ptr
                    });
                }
                slice<double> values_slice = {
                    values_info.size / sizeof(double),
                    values_ptr
                };
                slice<double> interpolation_points_slice = {
                    values_info.size / sizeof(double),
                    interp_points_ptr
                };
                slice<double> out_slice = {num_elements, out};

                interpolate(
                    grid,
                    values_slice,
                    interpolation_points_slice,
                    out_slice,
                    fill_value,
                    assume_sorted
                );
            }


            py::capsule free_when_done(out, [](void *f) {
                double *out = reinterpret_cast<double *>(f);
                delete[] out;
            });

            return py::array_t<double>(
                {num_elements}, // Shape
                {sizeof(double)}, // Strides
                out, // the data pointer
                free_when_done
            ); // numpy array references this parent
        },
        R"pbdoc(
            Linear interpolator in any dimension. Arguments are similar to
            scipy.interpolate.RegularGridInterpolator

            :param points: Tuple of N axis coordinates specifying the grid.
            :param values: Nd array containing the grid point values.
            :param interp_points: List of points to interpolate, should have
                dimension (x, N). If this list is guaranteed to be sorted make sure
                to use the `assume_sorted` option.
            :param fill_value: The value to return for interpolation points
                  outside of the grid range.
            :param assume_sorted: Enable optimization when the list of interpolation
                points is sorted.
            :param max_threads: Limit the number of threads to a certain amount.
                Note: The number of threads will always be one of {1, 2, 4, 8}
        )pbdoc",
        py::arg("points"),
        py::arg("values"),
        py::arg("interp_points"),
        py::arg("fill_value") = std::nullopt,
        py::arg("assume_sorted") = false,
        py::arg("max_threads") = 8
    );

    m.def("interpolate_along_axis", [](
            py::array_t<double, py::array::c_style> points,
            py::array_t<double, py::array::c_style> values,
            py::array_t<double, py::array::c_style> interp_points,
            ssize_t axis_in,
            std::optional<double> fill_value,
            bool assume_sorted,
            size_t max_threads
        ) {
            py::buffer_info points_info = points.request();
            py::buffer_info values_info = values.request();
            py::buffer_info interp_points_info = interp_points.request();

            if (values.ndim() == 0 || interp_points.ndim() == 0) {
                throw py::type_error("Only arrays are supported, not scalar values!");
            }

            if (points.ndim() != values.ndim() || points.ndim() != interp_points.ndim()) {
                throw py::type_error(
                    "'points', 'values' and 'interp_points' must all have the "
                    "same number of dimensions!"
                );
            }
            size_t dimensions = (size_t) points.ndim();

            for (size_t i = 0; i < dimensions; i++) {
                if (points.shape(i) != values.shape(i)) {
                    throw py::type_error("'points' and 'values' must have the same shape!");
                }
            }

            size_t num_threads = std::max((size_t) 1, max_threads);

            if (axis_in < 0) { axis_in += dimensions; }
            if (axis_in >= dimensions || axis_in < 0) {
                throw py::type_error("'axis' out of range!");
            } else if (axis_in == 0 && max_threads > 1) {
                throw std::runtime_error(
                    "Cannot interpolate along axis 0 with multiple threads!"
                );
            }
            size_t axis = (size_t) axis_in;

            for (size_t i = 0; i < dimensions; i++) {
                if (i != axis && interp_points_info.shape[i] != points_info.shape[i]) {
                    std::stringstream ss;
                    ss << "Dimension mismatch at axis " << i << "! 'points' is "
                    << points_info.shape[i] << " but interp_points is "
                    << interp_points_info.shape[i] << "!";
                    throw py::type_error(ss.str());
                }
            }


            size_t interp_points_size = 1;
            for (size_t i = 0; i < dimensions; i++) {
                interp_points_size *= interp_points_info.shape[i];
            }
            double *out = new double[interp_points_size];

            py::capsule free_when_done(out, [](void *f) {
                double *out = reinterpret_cast<double *>(f);
                delete[] out;
            });

            std::vector<ssize_t> shape(interp_points_info.shape);
            std::vector<ssize_t> strides(interp_points_info.strides);

            auto out_array = py::array_t<double>(
                shape, // Shape
                strides, // Strides
                out, // the data pointer
                free_when_done
            ); // numpy array references this parent

            py::buffer_info out_info = out_array.request();

            num_threads = std::min(num_threads, (size_t) points_info.shape[0]);

            size_t thread_stride = points_info.shape[0] / num_threads;
            if (thread_stride * num_threads < points_info.shape[0]) {
                thread_stride += 1;
            }


            if (num_threads == 1) {
                interpolate_1d_along_axis(
                    std::move(points_info),
                    std::move(values_info),
                    std::move(interp_points_info),
                    std::move(out_info),
                    axis,
                    fill_value,
                    assume_sorted
                );
            } else {
                std::vector<std::future<void>> tasks;

                for (size_t i = 0; i < num_threads; i++) {
                    size_t index = i * thread_stride;
                    if (index >= points_info.shape[0]) {
                        break;
                    }
                    size_t num_elements =
                        index + thread_stride < points_info.shape[0]
                        ? thread_stride : points_info.shape[0] - index;

                    if (num_elements == 0) {
                        break;
                    }

                    std::vector<ssize_t> points_view_shape(points_info.shape);
                    points_view_shape[0] = num_elements;
                    py::buffer_info points_view_info(
                        (unsigned char *) points_info.ptr + index * points_info.strides[0],
                        points_info.itemsize,
                        points_info.format,
                        points_info.ndim,
                        points_view_shape,
                        std::vector<ssize_t>(points_info.strides)
                    );
                    std::vector<ssize_t> values_view_shape(values_info.shape);
                    values_view_shape[0] = num_elements;
                    py::buffer_info values_view_info(
                        (unsigned char *) values_info.ptr + index * values_info.strides[0],
                        values_info.itemsize,
                        values_info.format,
                        values_info.ndim,
                        values_view_shape,
                        std::vector<ssize_t>(values_info.strides)
                    );
                    std::vector<ssize_t> interp_points_view_shape(interp_points_info.shape);
                    interp_points_view_shape[0] = num_elements;
                    py::buffer_info interp_points_view_info(
                        (unsigned char *) interp_points_info.ptr + index * interp_points_info.strides[0],
                        interp_points_info.itemsize,
                        interp_points_info.format,
                        interp_points_info.ndim,
                        interp_points_view_shape,
                        std::vector<ssize_t>(interp_points_info.strides)
                    );
                    std::vector<ssize_t> out_view_shape(out_info.shape);
                    out_view_shape[0] = num_elements;
                    py::buffer_info out_view_info(
                        (unsigned char *) out_info.ptr + index * out_info.strides[0],
                        out_info.itemsize,
                        out_info.format,
                        out_info.ndim,
                        out_view_shape,
                        std::vector<ssize_t>(out_info.strides)
                    );
                    tasks.push_back(
                        std::async(
                            &interpolate_1d_along_axis,
                            std::move(points_view_info),
                            std::move(values_view_info),
                            std::move(interp_points_view_info),
                            std::move(out_view_info),
                            axis,
                            fill_value,
                            assume_sorted
                        )
                    );
                }
                for (auto &future : tasks) {
                    std::move(future);
                }
            }

            return out_array;
        },
        R"pbdoc(
          1D linear interpolator along a specific axis.

          :param points: N-dimensional x coordinates. Axis specified by
                'axis' must contain at least 2 points.
          :param values: N-dimensional y values. Must have the same shape as
                'points'.
          :param interp_points: N-dimensional x coordinates to interpolate at.
                The shape may only differ from that of 'points' at the axis
                specified by 'axis'. For example if 'points' has shape (1, 2, 3)
                and 'axis' is 2, then and shape like (1, 2, X) is valid.
          :param axis: The axis to interpolate along.
          :param fill_value: The value to return for interpolation points
                outside of the grid range.
          :param assume_sorted: Enable optimization when the list of interpolation
                points is sorted along the axis of interpolation.
          :param max_threads: Limit the number of threads to a certain amount.
        )pbdoc",
        py::arg("points"),
        py::arg("values"),
        py::arg("interp_points"),
        py::arg("axis") = -1,
        py::arg("fill_value") = std::nullopt,
        py::arg("assume_sorted") = false,
        py::arg("max_threads") = 8,
        py::return_value_policy::move
    );
}
