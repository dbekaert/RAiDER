#include "assert.h"
#include "stdio.h"
#include "interpolate.h"

#include <iostream>


// TODO, just take start and size
inline size_t bisect_left(double * data, size_t data_N, double x, size_t left, size_t right) {
    assert(left > 0);
    assert(right < data_N);

    while (right - left > 1) {
        size_t mid = left + (right - left) / 2;
        if (x < data[mid]) {
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
inline size_t find_left(double * data, size_t data_N, double x, size_t left, size_t right) {
    for (size_t i = 0; i < 5; i++) {
        if (left == data_N || x < data[left]) {
            return left;
        }
        left += 1;
    }

    return bisect_left(data, data_N, x, left, right);
}


void interpolate_1d(
    double * data_xs,
    size_t data_N,
    double * data_ys,
    double * xs,
    double * out,
    size_t N
) {
    size_t lo = 0;
    size_t hi = 0;
    for (size_t i = 0; i < N; i++) {
        double x = xs[i];
        hi = find_left(data_xs, data_N, x, lo, data_N);
        if (hi < 1) {
            hi = 1;
        } else if (hi > data_N - 1) {
            hi = data_N - 1;
        }

        lo = hi - 1;

        double x0 = data_xs[lo],
               x1 = data_xs[hi],
               // Output
               y0 = data_ys[lo],
               y1 = data_ys[hi];

        double slope = (y1 - y0) / (x1 - x0);
        out[i] = y0 + slope * (x - x0);
    }
}


// data_zs must have length data_x_N * data_y_N
// out must have length N
// interpolation_points must have length 2N
void interpolate_2d(
    double * data_xs,
    size_t data_x_N,
    double * data_ys,
    size_t data_y_N,
    double * data_zs,
    double * interpolation_points,
    double * out,
    size_t N
) {
    size_t lox = 0;
    size_t loy = 0;
    for (size_t i = 0; i < N; i++) {
        double x = interpolation_points[i * 2];
        double y = interpolation_points[i * 2 + 1];
        size_t hix = find_left(data_xs, data_x_N, x, lox, data_x_N);
        size_t hiy = find_left(data_ys, data_y_N, y, loy, data_y_N);
        if (hix < 1) {
            hix = 1;
        } else if (hix > data_x_N - 1) {
            hix = data_x_N - 1;
        }
        if (hiy < 1) {
            hiy = 1;
        } else if (hiy > data_y_N - 1) {
            hiy = data_y_N - 1;
        }

        lox = hix - 1;
        loy = hiy - 1;

        // https://en.wikipedia.org/wiki/Bilinear_interpolation
        double x0 = data_xs[lox],
               y0 = data_ys[loy],
               x1 = data_xs[hix],
               y1 = data_ys[hiy],
               // Outputs
               z00 = data_zs[lox * data_y_N + loy],
               z01 = data_zs[lox * data_y_N + hiy],
               z10 = data_zs[hix * data_y_N + loy],
               z11 = data_zs[hix * data_y_N + hiy];

        double dx = x1 - x0,
               dy = y1 - y0,
               dist_x0 = x - x0,
               dist_x1 = x1 - x,
               dist_y0 = y - y0,
               dist_y1 = y1 - y;

        out[i] = (
            dist_x1 * (z00 * dist_y1 + z01 * dist_y0) +
            dist_x0 * (z10 * dist_y1 + z11 * dist_y0)
        ) / (dx * dy);
    }
}

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
) {
    size_t lox = 0;
    size_t loy = 0;
    size_t loz = 0;
    for (size_t i = 0; i < N; i++) {
        double x = interpolation_points[i * 3];
        double y = interpolation_points[i * 3 + 1];
        double z = interpolation_points[i * 3 + 2];
        size_t hix = find_left(data_xs, data_x_N, x, lox, data_x_N);
        size_t hiy = find_left(data_ys, data_y_N, y, loy, data_y_N);
        size_t hiz = find_left(data_zs, data_z_N, z, loz, data_z_N);
        if (hix < 1) {
            hix = 1;
        } else if (hix > data_x_N - 1) {
            hix = data_x_N - 1;
        }
        if (hiy < 1) {
            hiy = 1;
        } else if (hiy > data_y_N - 1) {
            hiy = data_y_N - 1;
        }
        if (hiz < 1) {
            hiz = 1;
        } else if (hiz > data_z_N - 1) {
            hiz = data_z_N - 1;
        }

        lox = hix - 1;
        loy = hiy - 1;
        loz = hiz - 1;

        // https://en.wikipedia.org/wiki/Trilinear_interpolation
        size_t data_yz_N = data_y_N * data_z_N;
        double x0 = data_xs[lox],
               y0 = data_ys[loy],
               z0 = data_zs[loz],
               x1 = data_xs[hix],
               y1 = data_ys[hiy],
               z1 = data_zs[hiz],
               // Outputs
               w000 = data_ws[lox * data_yz_N + loy * data_z_N + loz],
               w001 = data_ws[lox * data_yz_N + loy * data_z_N + hiz],
               w010 = data_ws[lox * data_yz_N + hiy * data_z_N + loz],
               w011 = data_ws[lox * data_yz_N + hiy * data_z_N + hiz],
               w100 = data_ws[hix * data_yz_N + loy * data_z_N + loz],
               w101 = data_ws[hix * data_yz_N + loy * data_z_N + hiz],
               w110 = data_ws[hix * data_yz_N + hiy * data_z_N + loz],
               w111 = data_ws[hix * data_yz_N + hiy * data_z_N + hiz];

        double dx = x1 - x0,
               dy = y1 - y0,
               dz = z1 - z0,
               dist_x0 = x - x0,
               dist_x1 = x1 - x,
               dist_y0 = y - y0,
               dist_y1 = y1 - y,
               dist_z0 = z - z0,
               dist_z1 = z1 - z;

        out[i] = (
            dist_x1 * (
                dist_y1 * (dist_z1 * w000 + dist_z0 * w001) +
                dist_y0 * (dist_z1 * w010 + dist_z0 * w011)
            ) +
            dist_x0 * (
                dist_y1 * (dist_z1 * w100 + dist_z0 * w101) +
                dist_y0 * (dist_z1 * w110 + dist_z0 * w111)
            )
        ) / (dx * dy * dz);
    }
}

void interpolate(
    const std::vector<slice<double>> &grid,
    const slice<double> &values,
    const slice<double> &interpolation_points,
    slice<double> &out
) {
    // Only up to 64 dimensions is supported
    assert(grid.size() < 64);

    size_t dimensions = grid.size();
    size_t num_points = out.size;

    std::vector<size_t> los(dimensions);
    std::vector<size_t> his(dimensions);

    std::vector<double> lower_bounds(dimensions);
    std::vector<double> upper_bounds(dimensions);
    std::vector<double> lower_dist(dimensions);
    std::vector<double> upper_dist(dimensions);


    std::vector<double> corner_points(1 << dimensions);


    for (size_t i = 0; i < num_points; i++) {
        double total_volume = 1;
        for (size_t dim = 0; dim < dimensions; dim++) {
            auto xs = grid[dim];
            double x = interpolation_points.ptr[i * dimensions + dim];
            size_t hi = find_left(xs.ptr, xs.size, x, los[dim], xs.size);
            if (hi < 1) {
                hi = 1;
            } else if (hi > xs.size - 1) {
                hi = xs.size - 1;
            }
            size_t lo = hi - 1;

            los[dim] = lo;
            his[dim] = hi;

            double x0 = xs.ptr[lo];
            double x1 = xs.ptr[hi];

            lower_bounds[dim] = x0;
            upper_bounds[dim] = x1;
            total_volume *= x1 - x0;
            lower_dist[dim] = x - x0;
            upper_dist[dim] = x1 - x;
        }

        out.ptr[i] = 0;
        for (unsigned long j = 0; j < corner_points.size(); j++) {
            size_t index = 0;
            // lox * data_y * data_z + loy * data_z + loz
            // (data_z (data_y * (lox) + loy) + loz) * 1
            for (size_t dim = 0; dim < dimensions; dim++) {
                index += (
                    (j >> (dim)) & 1 ? his[dim] : los[dim]
                );
                index *= (dim + 1 < dimensions ? grid[dim + 1].size : 1);
            }
            double term = values.ptr[index];
            for (size_t dim = 0; dim < dimensions; dim++) {
                term *= (j >> (dim)) & 1 ? lower_dist[dim] : upper_dist[dim];
            }
            out.ptr[i] += term;
        }

        out.ptr[i] /= total_volume;
    }
}
