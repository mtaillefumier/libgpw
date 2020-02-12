#include <iostream>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>

#include "mdarray.hpp"

// number of polinomials x^a y ^ beta z ^ gamma with alpha + beta + gamma = l
// we need to add 3 for the first derivatives along x, y, z (if needed)
// we go from l = 0 to l = 9

static const int length_[] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55};
// the possible indices are encoded as z * 2 ^ 16 + y * 2 ^ 8 + x.
static const int exponents[] = {0, 1, 256, 65536, 2, 257, 512, 65537, 65792, 131072, 3, 258, 513, \
                                768, 65538, 65793, 66048, 131073, 131328, 196608, 4, 259, 514, 769, \
                                1024, 65539, 65794, 66049, 66304, 131074, 131329, 131584, 196609, \
                                196864, 262144, 5, 260, 515, 770, 1025, 1280, 65540, 65795, 66050, \
                                66305, 66560, 131075, 131330, 131585, 131840, 196610, 196865, 197120, \
                                262145, 262400, 327680, 6, 261, 516, 771, 1026, 1281, 1536, 65541, \
                                65796, 66051, 66306, 66561, 66816, 131076, 131331, 131586, 131841, \
                                132096, 196611, 196866, 197121, 197376, 262146, 262401, 262656, \
                                327681, 327936, 393216, 7, 262, 517, 772, 1027, 1282, 1537, 1792, \
                                65542, 65797, 66052, 66307, 66562, 66817, 67072, 131077, 131332, \
                                131587, 131842, 132097, 132352, 196612, 196867, 197122, 197377, \
                                197632, 262147, 262402, 262657, 262912, 327682, 327937, 328192, \
                                393217, 393472, 458752, 8, 263, 518, 773, 1028, 1283, 1538, 1793, \
                                2048, 65543, 65798, 66053, 66308, 66563, 66818, 67073, 67328, 131078, \
                                131333, 131588, 131843, 132098, 132353, 132608, 196613, 196868, \
                                197123, 197378, 197633, 197888, 262148, 262403, 262658, 262913, \
                                263168, 327683, 327938, 328193, 328448, 393218, 393473, 393728, \
                                458753, 459008, 524288, 9, 264, 519, 774, 1029, 1284, 1539, 1794, \
                                2049, 2304, 65544, 65799, 66054, 66309, 66564, 66819, 67074, 67329, \
                                67584, 131079, 131334, 131589, 131844, 132099, 132354, 132609, \
                                132864, 196614, 196869, 197124, 197379, 197634, 197889, 198144, \
                                262149, 262404, 262659, 262914, 263169, 263424, 327684, 327939, \
                                328194, 328449, 328704, 393219, 393474, 393729, 393984, 458754, \
                                459009, 459264, 524289, 524544, 589824};

static const int offset_[10] = {0, 1, 4, 10, 20, 35, 56, 84, 120, 165};

template<typename type = float, typename period = std::milli>
class stopwatch
{
public:
    using clock      = std::chrono::high_resolution_clock;
    using duration   = std::chrono::duration<type, period>;
    using time_point = std::chrono::time_point<clock, duration>;

    stopwatch           () : time_(clock::now()) { }
    stopwatch           (const stopwatch&  that) = default;
    stopwatch           (      stopwatch&& temp) = default;
    ~stopwatch           ()                       = default;
    stopwatch& operator=(const stopwatch&  that) = default;
    stopwatch& operator=(      stopwatch&& temp) = default;

    duration tick ()
        {
            time_point time  = clock::now();
            duration   delta = time - time_;
            time_            = time;
            return delta;
        }

private:
    time_point time_;
};

template <typename T> void compute_folded_integrals(const int center,
                                                    const int period,
                                                    const int length,
                                                    const T *__restrict unfolded_table,
                                                    T *__restrict folded_table)
{
    const int p = (center - (length + 1) / 2 + 16 * period) % period;
    const int start = period - p;

    for (int i = 0; i < std::min(start, length); i++)
        folded_table[i + p] += unfolded_table[i];

    if (length > start) {
    // we need the remaining elements of the unfolded table : # of elements length - start
        for (int i = 0; i < length - start;) {
             if (i + start + period < length) {
                 for (int k = 0; k < period; k++) {
                     folded_table[k] += unfolded_table[i + start + k];
                 }
             } else {
                 for (int k = 0; (k + i) < length - start; k++)
                     folded_table[k] += unfolded_table[i + start + k];
             }
             i += period;
         }
    }
}

void compute_nonzero_values(const int center, const int period, const int length, int *non_zeroes_)
{
    for (int  i = 0; i < period; i++) {
        non_zeroes_[i] = 0;
    }

    if ((period - center) >= (length + 1) / 2) {
        for (int i = center; i < center + (length + 1) / 2 - 1; i++) {
            non_zeroes_[i]  = 1;
        }
    } else {
        if ((length + 1) / 2 >= period) {
            for (int i = 0; i < period; i++) {
                non_zeroes_[i] = 1;
            }
            return;
        } else {
            for (int i = center; i < period; i++) {
                non_zeroes_[i] = 1;
            }
            for (int i = 0; i < (length + 1) / 2 - period + center - 1; i++) {
                non_zeroes_[i] = 1;
            }
        }
    }

    if (center >= (length + 1) / 2) {
        for (int i = 0; i < (length + 1) / 2; i++) {
            non_zeroes_[i + center - (length + 1) / 2] = 1;
        }
        return;
    } else {
        if ((length + 1) / 2 >= period) {
            for (int i = 0; i < period; i++) {
                non_zeroes_[i] = 1;
            }
        } else {
            for (int i = 0; i < center; i++) {
                non_zeroes_[i] = 1;
            }

            for (int i = period + center - (length + 1) / 2; i < period; i++) {
                non_zeroes_[i] = 1;
            }
        }
    }
}

void compute_folded_integrals(const int center,
                              const int period,
                              const int length,
                              const double *__restrict unfolded_table,
                              double *__restrict folded_table)
{
    const int p = (center - (length + 1) / 2 + 16 * period) % period;
    const int start = period - p;

    for (int i = 0; i < std::min(start, length); i++)
        folded_table[i + p] += unfolded_table[i];

    if (length > start) {
        // we need the remaining elements of the unfolded table : # of elements length - start
        for (int i = 0; i < length - start;) {
            if (i + start + period < length) {
                for (int k = 0; k < period; k++) {
                    folded_table[k] += unfolded_table[i + start + k];
                }
            } else {
                for (int k = 0; (k + i) < length - start; k++)
                    folded_table[k] += unfolded_table[i + start + k];
            }
            i += period;
        }
    }
}


void compute_sliding_window(const double *rab,
                            const int *radius_int,
                            const double *dr,
                            mdarray<double, 3, CblasRowMajor> &V,
                            mdarray<double, 3, CblasRowMajor> &V_reduced);

inline void compute_indice(const int n, int &alpha, int &beta, int &gamma)
{
    gamma = (n & 0xFF0000) >> 16;
    beta = (n & 0xFF00) >> 8;
    alpha = (n & 0xFF);
}
void calculate_center_of_two_gaussians(const double *r_a,
                                       const double *r_b,
                                       const double sigma_a,
                                       const double sigma_b,
                                       double *rab,
                                       double *kappa_ab,
                                       double *mu_ab,
                                       double *mu_mean)
{
    *mu_mean = (sigma_a + sigma_b);
    for (int i = 0; i < 3; i++)
        rab[i] = (sigma_a * r_a[i] + sigma_b * r_b[i])/(*mu_mean);

    *mu_ab = sigma_a * sigma_b /(*mu_mean);
    double tmp = (r_a[0] - r_b[0]) * (r_a[0] - r_b[0]) +
        (r_a[1] - r_b[1]) * (r_a[1] - r_b[1]) +
        (r_a[2] - r_b[2]) * (r_a[2] - r_b[2]);
    kappa_ab[0] = exp(-tmp * mu_ab[0]);
}

void calculate_polynomials_variant1(const int l,
                                    const int *n_alpha,
                                    const double *dr,
                                    const double *r_a,
                                    const double *r_ab,
                                    const double mu_mean,
                                    mdarray<double, 3, CblasRowMajor> &p_alpha)
{
    // l is the angular momentum. This gives the degree of the polynomial. For
    // the forces we need polynomials of degree l + 1

    // first compute the max of n_alpha

    const int n_max = std::max(n_alpha[0], std::max(n_alpha[1], n_alpha[2]));
    // we compute for -n_max * dr to n_max * dr
    p_alpha = mdarray<double, 3, CblasRowMajor>(3, 2 * n_max + 1, l + 2);
    p_alpha.zero();
    for (int i = -n_max; i <= n_max; i++) {
        p_alpha(0, i + n_max, 0) = exp(-mu_mean * (r_ab[0] - i * dr[0]) * (r_ab[0] - i * dr[0]));
        p_alpha(1, i + n_max, 0) = exp(-mu_mean * (r_ab[1] - i * dr[1]) * (r_ab[1] - i * dr[1]));
        p_alpha(2, i + n_max, 0) = exp(-mu_mean * (r_ab[2] - i * dr[2]) * (r_ab[2] - i * dr[2]));
    }

    // we add the extra exponent for the forces
        // [0..n_max-1] positive r_alpha from 0..n_max * dr_alpha
        // [n_max,.. 2 n_max + 1] negative r_alpha
        // fft convention for the frequency domain
    for (int i = -n_max; i <= n_max; i++) {
        for (int m = 1; m < l + 2; m++) {
            p_alpha(0, i + n_max, m) = (r_a[0] - i * dr[0]) * p_alpha(0, i + n_max, m - 1);
            p_alpha(1, i + n_max, m) = (r_a[1] - i * dr[1]) * p_alpha(1, i + n_max, m - 1);
            p_alpha(2, i + n_max, m) = (r_a[2] - i * dr[2]) * p_alpha(2, i + n_max, m - 1);
        }
    }
}

void calculate_polynomials_variant2(const int l,
                                    const double radius,
                                    const double *dr,
                                    const double *r_a,
                                    const double *r_ab,
                                    const double mu_mean,
                                    mdarray<double, 3, CblasRowMajor> &p_alpha)
{
    // l is the angular momentum. This gives the degree of the polynomial. For
    // the forces we need polynomials of degree l + 1

    // first compute the max of n_alpha

    const int n_max = radius / std::min(dr[0], std::min(dr[1], dr[2])) + 1;
    // we compute for -n_max * dr to n_max * dr
    p_alpha = mdarray<double, 3, CblasRowMajor>(3, l + 2, 2 * n_max + 1);
    p_alpha.zero();
    for (int i = -n_max; i <= n_max; i++) {
        p_alpha(0, 0, i + n_max) = exp(-mu_mean * (r_ab[0] - i * dr[0]) * (r_ab[0] - i * dr[0]));
        p_alpha(1, 0, i + n_max) = exp(-mu_mean * (r_ab[1] - i * dr[1]) * (r_ab[1] - i * dr[1]));
        p_alpha(2, 0, i + n_max) = exp(-mu_mean * (r_ab[2] - i * dr[2]) * (r_ab[2] - i * dr[2]));
    }

    // we add the extra exponent for the forces
        // [0..n_max-1] positive r_alpha from 0..n_max * dr_alpha
        // [n_max,.. 2 n_max + 1] negative r_alpha
        // fft convention for the frequency domain
    for (int i = -n_max; i <= n_max; i++) {
        for (int m = 1; m < l + 2; m++) {
            p_alpha(0, m, i + n_max) = (r_a[0] - i * dr[0]) * p_alpha(0, m - 1, i + n_max);
            p_alpha(1, m, i + n_max) = (r_a[1] - i * dr[1]) * p_alpha(1, m - 1, i + n_max);
            p_alpha(2, m, i + n_max) = (r_a[2] - i * dr[2]) * p_alpha(2, m - 1, i + n_max);
        }
    }
}

void calculate_integral_sphere(const int l1,
                               const int l2,
                               const double *dr,
                               const double *r_ab,
                               const double radius,
                               mdarray<double, 3, CblasRowMajor> &p_alpha,
                               mdarray<double, 3, CblasRowMajor> &p_beta,
                               mdarray<double, 3, CblasRowMajor> &V,
                               const int center[3],
                               const double kappa_ab,
                               mdarray<double, 2, CblasRowMajor> &res)
{
    res.zero();
    // I assume for now that the potential is centered around 0
    // first compute the radius in integer coordinates

    int radius_int[3];
    const int n_max = p_alpha.size(1) / 2;
    for (int i = 0; i < 3; i++)
        radius_int[i] = lrint(radius / dr[i]);

    for (int z = -radius_int[0]; z <= radius_int[0]; z++) {
        // compute the radius of the disc at fixed z
        double radius_y = sqrt(radius * radius - z * dr[0] * z * dr[0]);
        int y_r = lrint(radius_y / dr[1]);
        for (int y = -y_r; y <= y_r; y++) {
            // compute the segment length for the x direction
            int x_r = lrint(sqrt(radius_y * radius_y - y * dr[1] * y * dr[1]) / dr[2]);
            for (int x = - x_r; x <= x_r; x++) {
                double pot = V(z + center[0], y + center[1], x + center[2]);
                for (int m1 = 0; m1 <  length_[l1]; m1++) {
                    int gamma1, beta1, alpha1;
                    compute_indice(exponents[offset_[l1] + m1], alpha1, beta1, gamma1);
                    double pot1 = pot *
                        p_beta(0, x + n_max, alpha1) *
                        p_beta(1, y + n_max, beta1) *
                        p_beta(2, z + n_max, gamma1);
                    for (int m2 = 0; m2 <  length_[l2]; m2++) {
                        int gamma2, beta2, alpha2;
                        compute_indice(exponents[offset_[l2] + m2], alpha2, beta2, gamma2);
                        res(m1, m2) += pot1 *
                            p_alpha(0, x + n_max, alpha2) *
                            p_alpha(1, y + n_max, beta2) *
                            p_alpha(2, z + n_max, gamma2);
                    }
                }
            }
        }
    }
}

void calculate_integral_sphere_variant2(const int l1,
                                        const int l2,
                                        const double *dr,
                                        const double *r_ab,
                                        const double radius,
                                        mdarray<double, 3, CblasRowMajor> &p_alpha,
                                        mdarray<double, 3, CblasRowMajor> &p_beta,
                                        mdarray<double, 3, CblasRowMajor> &V,
                                        const int center[3],
                                        const double kappa_ab,
                                        mdarray<double, 2, CblasRowMajor> &res)
{
    // I assume for now that the potential is centered around 0
    // first compute the radius in integer coordinates

    int radius_int[3];
    const int n_max = p_alpha.size(2) / 2;
    for (int i = 0; i < 3; i++)
        radius_int[i] = lrint(radius / dr[i]);
    std::vector<double> tmp(V.size(2), 0.0);
    std::vector<double> tmp_xy(V.size(2) , 0.0);
    std::vector<double> x_12(p_alpha.size(2), 0.0);
    std::vector<double> y_12(p_alpha.size(2), 0.0);
    std::vector<double> z_12(p_alpha.size(2), 0.0);
    for (int m1 = 0; m1 <  length_[l1]; m1++) {
        int gamma1, beta1, alpha1;
        compute_indice(exponents[offset_[l1] + m1], alpha1, beta1, gamma1);
        for (int m2 = 0; m2 <  length_[l2]; m2++) {
            int gamma2, beta2, alpha2;
            compute_indice(exponents[offset_[l2] + m2], alpha2, beta2, gamma2);

            {
                const double *__restrict pb = p_beta.at<CPU>(0, alpha1, 0);
                const double *__restrict pa = p_alpha.at<CPU>(0, alpha2, 0);
                for (int x = 0; x < x_12.size(); x++) {
                    x_12[x] = pa[x] * pb[x];
                }
            }

            {
                const double *__restrict pb = p_beta.at<CPU>(1, beta1, 0);
                const double *__restrict pa = p_alpha.at<CPU>(1, beta2, 0);
                for (int x = 0; x < x_12.size(); x++) {
                    y_12[x] = pa[x] * pb[x];
                }
            }

            {
                const double *__restrict pb = p_beta.at<CPU>(2, gamma1, 0);
                const double *__restrict pa = p_alpha.at<CPU>(2, gamma2, 0);
                for (int x = 0; x < x_12.size(); x++) {
                    z_12[x] = pa[x] * pb[x];
                }
            }

            for (int z = -radius_int[0]; z <= radius_int[0]; z++) {
                // compute the radius of the disc at fixed z
                double radius_y = sqrt(radius * radius - z * dr[0] * z * dr[0]);
                int y_r = lrint(radius_y / dr[1]);

                for (int y = -y_r; y <= y_r; y++) {
                    // compute the segment length for the x direction
                    int x_r = lrint(sqrt(radius_y * radius_y - y * dr[1] * y * dr[1]) / dr[2]);
                    const double pre_factor2 = z_12[z + n_max] * y_12[y + n_max];
                    const double *__restrict const pot1 = V.at<CPU>(z + center[0], y + center[1], center[2] - x_r);
                    double *__restrict tp = &tmp_xy[-x_r + tmp.size() >> 1];
                    double *__restrict x12 = &x_12[-x_r + n_max];
                    for (int x = 0; x <= 2 * x_r + 1; x++) {
                        tp[x] += pre_factor2 * pot1[x] * x12[x];
                    }
                }


            }

            double res1 = 0.0;
            for (int s = 0; s < tmp.size(); s++)
                res1 += tmp[s];

            res(m1, m2) = res1;
            std::memset(&tmp[0], 0, sizeof(double) * tmp.size());
        }
    }
}

void calculate_integral_sphere_variant3(const int l1,
                                        const int l2,
                                        const double *dr,
                                        const double *r_ab,
                                        const double radius,
                                        mdarray<double, 3, CblasRowMajor> &p_alpha,
                                        mdarray<double, 3, CblasRowMajor> &p_beta,
                                        mdarray<double, 3, CblasRowMajor> &V,
                                        const int center[3],
                                        const double kappa_ab,
                                        mdarray<double, 2, CblasRowMajor> &res)
{
    // I assume for now that the potential is centered around 0
    // first compute the radius in integer coordinates

    int radius_int[3];
    const int n_max = p_alpha.size(2) / 2;
    for (int i = 0; i < 3; i++)
        radius_int[i] = lrint(radius / dr[i]);
    std::vector<double> tmp(2 * radius_int[2] + 1, 0.0);
    std::vector<double> x_12(2 * radius_int[2] + 1, 0.0);
    std::vector<double> y_12(2 * radius_int[1] + 1, 0.0);
    std::vector<double> z_12(2 * radius_int[0] + 1, 0.0);

    mdarray<double, 3, CblasRowMajor> V_reduced(2 * radius_int[0] + 1, 2 * radius_int[1] + 1, 2 * radius_int[2] + 1);

    compute_sliding_window(r_ab,
                           radius_int,
                           dr,
                           V,
                           V_reduced);

    for (int m1 = 0; m1 <  length_[l1]; m1++) {
        int gamma1, beta1, alpha1;
        compute_indice(exponents[offset_[l1] + m1], alpha1, beta1, gamma1);
        for (int m2 = 0; m2 <  length_[l2]; m2++) {
            int gamma2, beta2, alpha2;
            compute_indice(exponents[offset_[l2] + m2], alpha2, beta2, gamma2);

            {
                const double *__restrict pb = p_beta.at<CPU>(0, alpha1, n_max - radius_int[2]);
                const double *__restrict pa = p_alpha.at<CPU>(0, alpha2, n_max - radius_int[2]);
                double *__restrict x12 = &x_12[0];
                for (int x = 0; x < x_12.size(); x++) {
                    x12[x] = pa[x] * pb[x];
                }
            }

            {
                const double *__restrict pb = p_beta.at<CPU>(1, beta1, n_max - radius_int[1]);
                const double *__restrict pa = p_alpha.at<CPU>(1, beta2, n_max - radius_int[1]);
                for (int x = 0; x < x_12.size(); x++) {
                    y_12[x] = pa[x] * pb[x];
                }
            }

            {
                const double *__restrict pb = p_beta.at<CPU>(2, gamma1, n_max - radius_int[0]);
                const double *__restrict pa = p_alpha.at<CPU>(2, gamma2, n_max - radius_int[0]);
                for (int x = 0; x < x_12.size(); x++) {
                    z_12[x] = pa[x] * pb[x];
                }
            }

            for (int z = -radius_int[0]; z <= radius_int[0]; z++) {
                // compute the radius of the disc at fixed z
                double radius_y = sqrt(radius * radius - z * dr[0] * z * dr[0]);
                int y_r = lrint(radius_y / dr[1]);

                // compute t_zx = sum_{y} Pot_{zyx} * pot_y


                for (int y = -y_r; y <= y_r; y++) {
                    // compute the segment length for the x direction
                    int x_r = lrint(sqrt(radius_y * radius_y - y * dr[1] * y * dr[1]) / dr[2]);
                    const double pre_factor = y_12[y + radius_int[1]] * z_12[z + radius_int[0]];
                    const double *__restrict const pot1 = V_reduced.at<CPU>(z + radius_int[0], y + radius_int[1], radius_int[2] - x_r);
                    double *__restrict tp = &tmp[radius_int[2] - x_r];
#pragma ivdep
                    for (int x = 0; x < 2 * x_r + 1; x++) {
                        tp[x] += pre_factor * pot1[x];
                    }
                }
            }


            double res1 = 0.0;
            double *__restrict tp = &tmp[0];

#pragma ivdep
            for (int s = 0; s < tmp.size(); s++)
                res1 += tp[s] * x_12[s];

            res(m1, m2) = res1;
            std::memset(&tmp[0], 0, sizeof(double) * tmp.size());
        }
    }
    V_reduced.clear();
}


void calculate_integral_box(const int l1,
                            const int l2,
                            const double *dr,
                            const double *r_ab,
                            const double radius,
                            mdarray<double, 3, CblasRowMajor> &p_alpha,
                            mdarray<double, 3, CblasRowMajor> &p_beta,
                            mdarray<double, 3, CblasRowMajor> &V,
                            const int center[3],
                            const double kappa_ab,
                            mdarray<double, 2, CblasRowMajor> &res)
{
    res.zero();
    // I assume for now that the potential is centered around 0
    // first compute the radius in integer coordinates

    int radius_int[3];
    const int n_max = (p_alpha.size(1) - 1) / 2;
    for (int i = 0; i < 3; i++)
        radius_int[i] = radius / dr[i];
    const int *expo1 = &exponents[offset_[l1]];
    const int *expo2 = &exponents[offset_[l2]];

    for (int z = -radius_int[0]; z <= radius_int[0]; z++) {
        for (int y = -radius_int[1]; y <= radius_int[1]; y++) {
            // compute the segment length for the x direction
            for (int x = - radius_int[2]; x <= radius_int[2]; x++) {
                double pot = V(z + center[0], y + center[1], x + center[2]);
#pragma ivdep
                for (int m1 = 0; m1 <  length_[l1]; m1++) {
                    int gamma1, beta1, alpha1;
                    compute_indice(expo1[m1], alpha1, beta1, gamma1);
                    const double pot1 = pot *
                        p_beta(0, x + n_max, alpha1) *
                        p_beta(1, y + n_max, beta1) *
                        p_beta(2, z + n_max, gamma1);
                    for (int m2 = 0; m2 <  length_[l2]; m2++) {
                        int gamma2, beta2, alpha2;
                        compute_indice(expo2[m2], alpha2, beta2, gamma2);
                        res(m1, m2) += pot1 *
                            p_alpha(0, x + n_max, alpha2) *
                            p_alpha(1, y + n_max, beta2) *
                            p_alpha(2, z + n_max, gamma2);
                    }
                }
            }
        }
    }
}

void calculate_integral_box_variant2(const int l1,
                                     const int l2,
                                     const double *dr,
                                     const double *r_ab,
                                     const double radius,
                                     mdarray<double, 3, CblasRowMajor> &p_alpha,
                                     mdarray<double, 3, CblasRowMajor> &p_beta,
                                     mdarray<double, 3, CblasRowMajor> &V,
                                     const int center[3],
                                     const double kappa_ab,
                                     mdarray<double, 2, CblasRowMajor> &res)
{
    res.zero();
    // I assume for now that the potential is centered around 0
    // first compute the radius in integer coordinates

    int radius_int[3];
    const int n_max = (p_alpha.size(2) - 1) / 2;
    for (int i = 0; i < 3; i++)
        radius_int[i] = radius / dr[i];
    const int *expo1 = &exponents[offset_[l1]];
    const int *expo2 = &exponents[offset_[l2]];
    std::vector<double> tmp(V.size(2), 0.0);
    std::vector<double> x_12(p_alpha.size(2), 0.0);
    std::vector<double> y_12(p_alpha.size(2), 0.0);
    std::vector<double> z_12(p_alpha.size(2), 0.0);

    for (int m1 = 0; m1 <  length_[l1]; m1++) {
        int gamma1, beta1, alpha1;
        compute_indice(expo1[m1], alpha1, beta1, gamma1);
        for (int m2 = 0; m2 <  length_[l2]; m2++) {
            int gamma2, beta2, alpha2;
            compute_indice(expo1[m2], alpha2, beta2, gamma2);

            {
                const double *__restrict pb = p_beta.at<CPU>(0, alpha1, 0);
                const double *__restrict pa = p_alpha.at<CPU>(0, alpha2, 0);
                for (int x = 0; x < x_12.size(); x++) {
                    x_12[x] = pa[x] * pb[x];
                }
            }

            {
                const double *__restrict pb = p_beta.at<CPU>(1, beta1, 0);
                const double *__restrict pa = p_alpha.at<CPU>(1, beta2, 0);
                for (int x = 0; x < x_12.size(); x++) {
                    y_12[x] = pa[x] * pb[x];
                }
            }

            {
                const double *__restrict pb = p_beta.at<CPU>(2, gamma1, 0);
                const double *__restrict pa = p_alpha.at<CPU>(2, gamma2, 0);
                for (int x = 0; x < x_12.size(); x++) {
                    z_12[x] = pa[x] * pb[x];
                }
            }

            for (int z = -radius_int[0]; z <= radius_int[0]; z++) {
                // compute the radius of the disc at fixed z
                for (int y = -radius_int[1]; y <= radius_int[1]; y++) {
                    // compute the segment length for the x direction
                    const double pre_factor2 = z_12[z + n_max] * y_12[y + n_max];
                    const double *__restrict const pot1 = V.at<CPU>(z + center[0], y + center[1], center[2] - radius_int[2]);
                    double *__restrict tp = &tmp[-radius_int[2] + tmp.size() >> 1];
                    double *__restrict x12 = &x_12[-radius_int[2] + n_max];
                    for (int x = 0; x <= 2 * radius_int[2] + 1; x++) {
                        tp[x] += pre_factor2 * pot1[x] * x12[x];
                    }
                }
            }

            double rs = 0.0;
            for (int s = 0; s < tmp.size(); s++)
                 rs += tmp[s];
            res(m1, m2) = rs;

            memset(&tmp[0], 0, sizeof(double) * tmp.size());
        }
    }
}

void calculate_integral_box_variant3(const int l1,
                                     const int l2,
                                     const double *dr,
                                     const double *r_ab,
                                     const double radius,
                                     mdarray<double, 3, CblasRowMajor> &p_alpha,
                                     mdarray<double, 3, CblasRowMajor> &p_beta,
                                     mdarray<double, 3, CblasRowMajor> &V,
                                     const int center[3],
                                     const double kappa_ab,
                                     mdarray<double, 2, CblasRowMajor> &res)
{
    res.zero();
    // I assume for now that the potential is centered around 0
    // first compute the radius in integer coordinates

    int radius_int[3];
    const int n_max = (p_alpha.size(2) - 1) / 2;
    for (int i = 0; i < 3; i++)
        radius_int[i] = radius / dr[i];
    const int *expo1 = &exponents[offset_[l1]];
    const int *expo2 = &exponents[offset_[l2]];
    std::vector<double> tmp(3 * radius_int[2], 0.0);
    std::vector<double> x_12(p_alpha.size(2), 0.0);
    std::vector<double> y_12(p_alpha.size(2), 0.0);
    std::vector<double> z_12(p_alpha.size(2), 0.0);
    mdarray<double, 3, CblasRowMajor> V_reduced(2 * radius_int[0] + 1, 2 * radius_int[1] + 1, 2 * radius_int[2] + 1);

    compute_sliding_window(r_ab,
                           radius_int,
                           dr,
                           V,
                           V_reduced);

    for (int m1 = 0; m1 <  length_[l1]; m1++) {
        int gamma1, beta1, alpha1;
        compute_indice(expo1[m1], alpha1, beta1, gamma1);
        for (int m2 = 0; m2 <  length_[l2]; m2++) {
            int gamma2, beta2, alpha2;
            compute_indice(expo1[m2], alpha2, beta2, gamma2);

            {
                const double *__restrict pb = p_beta.at<CPU>(0, alpha1, 0);
                const double *__restrict pa = p_alpha.at<CPU>(0, alpha2, 0);
                for (int x = 0; x < x_12.size(); x++) {
                    x_12[x] = pa[x] * pb[x];
                }
            }

            {
                const double *__restrict pb = p_beta.at<CPU>(1, beta1, 0);
                const double *__restrict pa = p_alpha.at<CPU>(1, beta2, 0);
                for (int x = 0; x < x_12.size(); x++) {
                    y_12[x] = pa[x] * pb[x];
                }
            }

            {
                const double *__restrict pb = p_beta.at<CPU>(2, gamma1, 0);
                const double *__restrict pa = p_alpha.at<CPU>(2, gamma2, 0);
                for (int x = 0; x < x_12.size(); x++) {
                    z_12[x] = pa[x] * pb[x];
                }
            }

            for (int z = -radius_int[0]; z < radius_int[0]; z++) {
                // compute the radius of the disc at fixed z
                for (int y = -radius_int[1]; y < radius_int[1]; y++) {
                    // compute the segment length for the x direction
                    const double pre_factor2 = z_12[z + n_max] * y_12[y + n_max];
                    const double *__restrict const pot1 = V_reduced.at<CPU>(z + radius_int[0], y + radius_int[1], 0);
                    double *__restrict tp = &tmp[-radius_int[2] + tmp.size() >> 1];
                    double *__restrict x12 = &x_12[-radius_int[2] + n_max];
                    for (int x = 0; x < 2 * radius_int[2] + 1; x++) {
                        tp[x] += pre_factor2 * pot1[x] * x12[x];
                    }
                }
            }

            double rs = 0.0;
            for (int s = 0; s < tmp.size(); s++)
                 rs += tmp[s];
            res(m1, m2) = rs;

            memset(&tmp[0], 0, sizeof(double) * tmp.size());
        }
    }
}

void calculate_integral_box_variant4(const int l1,
                                     const int l2,
                                     const double *dr,
                                     const double *r_ab,
                                     const double radius,
                                     mdarray<double, 3, CblasRowMajor> &p_alpha,
                                     mdarray<double, 3, CblasRowMajor> &p_beta,
                                     mdarray<double, 3, CblasRowMajor> &V,
                                     const int center[3],
                                     const double kappa_ab,
                                     mdarray<double, 2, CblasRowMajor> &res)
{
    res.zero();
    // I assume for now that the potential is centered around 0
    // first compute the radius in integer coordinates

    int radius_int[3];
    const int n_max = (p_alpha.size(2) - 1) / 2;
    for (int i = 0; i < 3; i++)
        radius_int[i] = radius / dr[i];
    const int *expo1 = &exponents[offset_[l1]];
    const int *expo2 = &exponents[offset_[l2]];
    std::vector<double> x_12(p_alpha.size(2), 0.0);
    std::vector<double> y_12(p_alpha.size(2), 0.0);
    std::vector<double> z_12(p_alpha.size(2), 0.0);

    mdarray<double, 3, CblasRowMajor> V_reduced(2 * radius_int[0] + 1, 2 * radius_int[1] + 1, 2 * radius_int[2] + 1);
    mdarray<double, 3, CblasRowMajor> weight(2 * radius_int[0] + 1, 2 * radius_int[1] + 1, 2 * radius_int[2] + 1);

    compute_sliding_window(r_ab,
                           radius_int,
                           dr,
                           V,
                           V_reduced);
    // for (int z = -radius_int[0]; z < radius_int[0]; z++) {
    //     for (int y = -radius_int[1]; y < radius_int[1]; y++) {
    //         memcpy(V_reduced.at<CPU>(z + radius_int[0], y + radius_int[1], 0),
    //                V.at<CPU>(z + center[0], y + center[1], center[2] - radius_int[2]),
    //                sizeof(double) * (2 * radius_int[2] + 1));
    //     }
    // }

    for (int m1 = 0; m1 <  length_[l1]; m1++) {
        int gamma1, beta1, alpha1;
        compute_indice(expo1[m1], alpha1, beta1, gamma1);
        for (int m2 = 0; m2 <  length_[l2]; m2++) {
            int gamma2, beta2, alpha2;
            compute_indice(expo1[m2], alpha2, beta2, gamma2);

            {
                const double *__restrict pb = p_beta.at<CPU>(0, alpha1, 0);
                const double *__restrict pa = p_alpha.at<CPU>(0, alpha2, 0);
                for (int x = 0; x < x_12.size(); x++) {
                    x_12[x] = pa[x] * pb[x];
                }
            }

            {
                const double *__restrict pb = p_beta.at<CPU>(1, beta1, 0);
                const double *__restrict pa = p_alpha.at<CPU>(1, beta2, 0);
                for (int x = 0; x < x_12.size(); x++) {
                    y_12[x] = pa[x] * pb[x];
                }
            }

            {
                const double *__restrict pb = p_beta.at<CPU>(2, gamma1, 0);
                const double *__restrict pa = p_alpha.at<CPU>(2, gamma2, 0);
                for (int x = 0; x < x_12.size(); x++) {
                    z_12[x] = pa[x] * pb[x];
                }
            }

            const double *__restrict x12 = &x_12[-radius_int[2] + n_max];
            const double *__restrict y12 = &y_12[-radius_int[1] + n_max];
            const double *__restrict z12 = &z_12[-radius_int[0] + n_max];

            for (int z = 0; z <  2 * radius_int[0] + 1; z++) {
                for (int y = 0; y < 2 * radius_int[1] + 1; y++) {
                    const double pre_factor2 = z12[z] * y12[y];
                    for (int x = 0; x < 2 * radius_int[2] + 1; x++) {
                        weight(z, y, x) = pre_factor2 * x12[x];
                    }
                }
            }

            double rs[4] = { 0.0, 0.0, 0.0, 0.0};
            const double *__restrict__ rs1 = weight.at<CPU>();
            const double *__restrict__ rs2 = V_reduced.at<CPU>();

            res(m1, m2) = cblas_ddot(weight.size(), rs1, 1, rs2, 1);
        }
    }

    V_reduced.clear();
    weight.clear();
}

// NB : basis should be a 4x3 matrix
void compute_periodic_boundaries_conditions(double *ra, double *rb, double *basis, int *disp_vectors)
{
    double dist = 100000000000000;
    double rb1[3];
    for (int z = -1; z < 1; z++) {
        for (int y = -1; y < 1; y++) {
            for (int x = -1; x < 1; x++) {
                // first compute displacement vector
                double d_v[3];
                double r[3];
                d_v[0] = basis[0] * z + basis[1] * y + basis[2] * x;
                d_v[1] = basis[4] * z + basis[5] * y + basis[6] * x;
                d_v[2] = basis[8] * z + basis[9] * y + basis[10] * x;

                r[0] = ra[0] - rb[0] + d_v[0];
                r[1] = ra[1] - rb[1] + d_v[1];
                r[2] = ra[2] - rb[2] + d_v[2];
                double dist1 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
                if (dist1 < dist) {
                    dist = dist1;
                    disp_vectors[0] = z;
                    disp_vectors[1] = y;
                    disp_vectors[2] = x;
                    rb1[0] = rb[0] -  d_v[0];
                    rb1[1] = rb[1] -  d_v[1];
                    rb1[2] = rb[2] -  d_v[2];
                }
            }
        }
    }

    rb[0] = rb1[0];
    rb[1] = rb1[1];
    rb[2] = rb1[2];
}

// compute the index of the point that closest to rab. these coordinates are
// then used to shift the grid using periodic boundaries conditions. Then
// compute the reduced box

void compute_sliding_window(const double *rab,
                            const int *radius_int,
                            const double *dr,
                            mdarray<double, 3, CblasRowMajor> &V,
                            mdarray<double, 3, CblasRowMajor> &V_reduced)
{
    // first compute the center nearest to rab = alpha ra + beta rb / (a + b)
    int ind[3];

    ind[0] = rab[0] / dr[0];
    ind[1] = rab[1] / dr[1];
    ind[2] = rab[2] / dr[2] + 3 * V.size(2) / 2;

    std::vector<double> tmp (3 * V.size(2));
    // now I want to go from ind - L/2 to ind + L/2 in each direction
    for (int z = -radius_int[0]; z < radius_int[0]; z++) {
        int z1 = (ind[0] - z + V.size(0)) % V.size(0);
        for (int y = -radius_int[1]; y < radius_int[1]; y++) {
            int y1 = (ind[1] - y + V.size(1)) % V.size(1);

            // I want to shift / rotate the grid along x such that the datas are
            // centered around rab. Then I can extract the data more easily
            // the rotating point is actually rab
            // so I need to compute rab_int % L and shift / rotate around that point

            // for (int x = ind[2]; x < grid_size[2]; x++) {
            //     tmp[x - ind[2]] = V(z1, y1, x);
            // }
            // for (int x = 0; x < ind[2]; x++) {
            //     tmp[grid_size[2] - ind[2] + x] = V(z1, y1, x);
            // }

            // now I can do a memcpy

            // memcpy(V_reduced.at<CPU>(z + radius_int[0], z + radius_int[1], 0),
            //        &tmp[grid_size[2] - radius_int[2]],
            //        sizeof(double) * (2 * radius_int[2] + 1));

            // an other option is to copy V_x three times and take
            // -radius_int[0]...radius_int[0] points from that table with center ind[2]

            // if we do that it is simply four memcpy
            memcpy(&tmp[0], V.at<CPU>(z1, y1, 0), sizeof(double) * V.size(2));
            memcpy(&tmp[V.size(2)], V.at<CPU>(z1, y1, 0), sizeof(double) * V.size(2));
            memcpy(&tmp[2 * V.size(2)], V.at<CPU>(z1, y1, 0), sizeof(double) * V.size(2));

            memcpy(V_reduced.at<CPU>(z + radius_int[0], y + radius_int[1], 0),
                   &tmp[ind[2] - radius_int[2]],
                   sizeof(double) * V_reduced.size(2));
        }
    }
    tmp.clear();
}

int main(int argc, char **argv)
{
    const int l1 = 2;
    const int l2 = 2;
    const int L[3] = {30, 30, 30};
    stopwatch<float, std::milli> watch;
    double ra[] = {0.1, 0.1 , 0.1};
    double rb[] = {0.2, 0.2 ,0.2};
    double dr[] = {0.3, 0.3, 0.3};
    double alpha_a = 0.25;
    double alpha_b = 0.34;
    double mu_ab, mu_mean, r_ab[3], kappa_ab;
    double radius = 5;
    mdarray<double, 3, CblasRowMajor> Potential;
    mdarray< double, 3, CblasRowMajor> p_alpha;
    mdarray<double, 3, CblasRowMajor> p_beta;

    mdarray<double, 2, CblasRowMajor> res;
    const int center[3] = {105, 105, 105};

    Potential = mdarray<double, 3, CblasRowMajor>(106, 106, 106);
    res = mdarray<double, 2, CblasRowMajor>(length_[l1], length_[l2]);

    for (int i = 0; i < Potential.size(0); i++)
        for (int j = 0; j < Potential.size(1); j++)
            for (int k = 0; k < Potential.size(2); k++)
                Potential(i, j, k) = 1.0;

    calculate_center_of_two_gaussians(ra, rb,
                                      alpha_a, alpha_b,
                                      r_ab, &kappa_ab,
                                      &mu_ab, &mu_mean);

    calculate_polynomials_variant1(l1, L, dr, ra, r_ab, mu_mean * 0.5, p_alpha);
    calculate_polynomials_variant1(l2, L, dr, rb, r_ab, mu_mean  * 0.5, p_beta);
    std::cout.precision(15);

    double dV = dr[0] * dr[1] * dr[2];
    double th_res  = std::pow(M_PI / mu_mean, 1.5);
    std::vector<std::string> v_s({"z^2", "y*z", "y^2", "x*z", "x*y", "x^2", " "});

//    for (radius = 1; radius < 10; radius+= 1.0) {
        // calculate_integral_sphere(l1, l2, dr, r_ab, radius, p_alpha, p_beta,
        //                           Potential, center, kappa_ab, res);

        // std::cout << "sphere variant 1 " << radius << "\n";
        // std::cout << "    ";
        // for (int m2 = 0; m2 < length_[l2]; m2++)
        //     std::cout << v_s[m2] << "                ";
        // std::cout << "\n";
        // for (int m1 = 0; m1 < length_[l1]; m1++) {
        //     std::cout << v_s[m1] << " ";
        //     for (int m2 = 0; m2 < length_[l2]; m2++)
        //         std::cout << res(m1, m2) * dV * kappa_ab << " ";
        //     std::cout << "\n";
        // }

        // calculate_integral_sphere_variant2(l1, l2, dr, r_ab, radius,
        //                                    p_alpha_variant2, p_beta_variant2,
        //                                    Potential, center, kappa_ab, res);
        // std::cout << "sphere variant 2\n";
        // std::cout << "    ";
        // for (int m2 = 0; m2 < length_[l2]; m2++)
        //     std::cout << v_s[m2] << "                ";
        // std::cout << "\n";

        // for (int m1 = 0; m1 < length_[l1]; m1++) {
        //     std::cout << v_s[m1] << " ";
        //     for (int m2 = 0; m2 < length_[l2]; m2++)
        //         std::cout << res(m1, m2) * dV * kappa_ab << " ";
        //     std::cout << "\n";
        // }

    //stopwatch<float, std::milli> t1;

    for (int i = 0; i < 10000; i++) {
        mdarray< double, 3, CblasRowMajor> p_alpha_variant2;
        mdarray<double, 3, CblasRowMajor> p_beta_variant2;
        calculate_polynomials_variant2(l1, 7, dr, ra, r_ab, mu_mean * 0.5, p_alpha_variant2);
        calculate_polynomials_variant2(l2, 7, dr, rb, r_ab, mu_mean  * 0.5, p_beta_variant2);
        calculate_integral_sphere_variant3(l1, l2, dr, r_ab, radius,
                                           p_alpha_variant2, p_beta_variant2,
                                           Potential, center, kappa_ab, res);
        p_alpha_variant2.clear();
        p_beta_variant2.clear();
    }

    //std::cout <<  t1.tick().count()  << std::endl;

    std::cout << "sphere variant 3\n";
    std::cout << "    ";
    for (int m2 = 0; m2 < length_[l2]; m2++)
        std::cout << v_s[m2] << "                ";
    std::cout << "\n";

    for (int m1 = 0; m1 < length_[l1]; m1++) {
        std::cout << v_s[m1] << " ";
        for (int m2 = 0; m2 < length_[l2]; m2++)
            std::cout << res(m1, m2) * dV * kappa_ab << " ";
        std::cout << "\n";
    }

    // std::cout << "box variant 1\n";

        // calculate_integral_box(l1, l2, dr, r_ab, radius, p_alpha, p_beta,
        //                        Potential, center, kappa_ab, res);
        // std::cout << "    ";
        // for (int m2 = 0; m2 < length_[l2]; m2++)
        //     std::cout << v_s[m2] << "                ";
        // std::cout << "\n";

        // for (int m1 = 0; m1 < length_[l1]; m1++) {
        //     std::cout << v_s[m1] << " ";
        //     for (int m2 = 0; m2 < length_[l2]; m2++)
        //         std::cout << res(m1, m2) * dV * kappa_ab << " ";
        //     std::cout << "\n";
        // }

        // std::cout << "box variant 2\n";
        // calculate_integral_box_variant2(l1, l2, dr, r_ab, radius, p_alpha_variant2, p_beta_variant2,
        //                                 Potential, center, kappa_ab, res);

        // std::cout << "    ";
        // for (int m2 = 0; m2 < length_[l2]; m2++)
        //     std::cout << v_s[m2] << "                ";
        // std::cout << "\n";

        // for (int m1 = 0; m1 < length_[l1]; m1++) {
        //     std::cout << v_s[m1] << " ";
        //     for (int m2 = 0; m2 < length_[l2]; m2++)
        //         std::cout << res(m1, m2) * dV * kappa_ab << " ";
        //     std::cout << "\n";
        // }

        // std::cout << "box variant 3\n";
        // calculate_integral_box_variant3(l1, l2, dr, r_ab, radius, p_alpha_variant2, p_beta_variant2,
        //                                 Potential, center, kappa_ab, res);
        // std::cout << "    ";
        // for (int m2 = 0; m2 < length_[l2]; m2++)
        //     std::cout << v_s[m2] << "                ";
        // std::cout << "\n";

        // for (int m1 = 0; m1 < length_[l1]; m1++) {
        //     std::cout << v_s[m1] << " ";
        //     for (int m2 = 0; m2 < length_[l2]; m2++)
        //         std::cout << res(m1, m2) * dV * kappa_ab << " ";
        //     std::cout << "\n";
        // }
        // std::cout << "box variant 4\n";
        // calculate_integral_box_variant4(l1, l2, dr, r_ab, radius, p_alpha_variant2, p_beta_variant2,
        //                                 Potential, center, kappa_ab, res);
        // std::cout << "    ";
        // for (int m2 = 0; m2 < length_[l2]; m2++)
        //     std::cout << v_s[m2] << "                ";
        // std::cout << "\n";

        // for (int m1 = 0; m1 < length_[l1]; m1++) {
        //     std::cout << v_s[m1] << " ";
        //     for (int m2 = 0; m2 < length_[l2]; m2++)
        //         std::cout << res(m1, m2) * dV * kappa_ab << " ";
        //     std::cout << "\n";
        // }
        //  }
    return 0;
}
