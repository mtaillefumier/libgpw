#include <iostream>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

#ifdef HAVE_LIBXSMM
#include <libxsmm.h>
#endif

// we need to replace the mdarray class by something a bit lighter. The problem
// comes from computation of the indices. We usually use tensors of order 3 so
// something like this ***a.

#include "mdarray.hpp"
#include "rt_graph.hpp"

rt_graph::Timer timer;



/* tabulated binomials up to n = 20 */

inline const int binomial(const int a, const int b)
{
    const int bin[21][21] = {{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                             {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                             {1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                             {1, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                             {1, 4, 6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                             {1, 5, 10, 10, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                             {1, 6, 15, 20, 15, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                             {1, 7, 21, 35, 35, 21, 7, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                             {1, 8, 28, 56, 70, 56, 28, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                             {1, 9, 36,  84, 126, 126, 84, 36, 9, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                             {1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0},
                             {1, 11, 55, 165, 330, 462, 462, 330, 165, 55, 11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                             {1, 12, 66, 220, 495, 792, 924, 792, 495, 220, 66, 12, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                             {1, 13, 78, 286, 715, 1287, 1716, 1716, 1287, 715, 286, 78, 13, 1, 0, 0, 0, 0, 0, 0, 0},
                             {1, 14, 91, 364, 1001, 2002, 3003, 3432, 3003, 2002, 1001, 364, 91, 14, 1, 0, 0, 0, 0, 0, 0},
                             {1, 15, 105, 455, 1365, 3003, 5005, 6435, 6435, 5005, 3003, 1365, 455, 105, 15, 1, 0, 0, 0, 0, 0},
                             {1, 16, 120, 560, 1820, 4368, 8008, 11440, 12870, 11440, 8008, 4368, 1820, 560, 120, 16, 1, 0, 0, 0, 0},
                             {1, 17, 136, 680, 2380, 6188, 12376, 19448, 24310, 24310, 19448, 12376, 6188, 2380, 680, 136, 17, 1, 0, 0, 0},
                             {1, 18, 153, 816, 3060, 8568, 18564, 31824, 43758, 48620, 43758, 31824, 18564, 8568, 3060, 816, 153, 18, 1, 0, 0},
                             {1, 19, 171, 969, 3876, 11628, 27132, 50388, 75582, 92378, 92378, 75582, 50388,  27132, 11628, 3876, 969, 171, 19, 1, 0},
                             {1, 20, 190, 1140, 4845, 15504, 38760, 77520, 125970, 167960, 184756, 167960, 125970, 77520, 38760, 15504, 4845, 1140, 190, 20, 1}};

    if ((a < 0) || (b < 0))
        return 0;

    return bin[a][b];
}

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


/* Use the code generated by the The Tensor Algebra Compiler (taco)

   http://tensor-compiler.org/codegen

   enter

   C(i,j,k) = x(l,k) * z(m, i) * A(l, m, n) * y(n,j)

   and hit the generate button

   I adapted the toca_tensor_t to mdarray manually and templated it
*/

int popcount(unsigned x)
{
    int c = 0;
    for (; x != 0; x >>= 1)
        if (x & 1)
            c++;
    return c;
}

template <typename T> class gaussian {
    //
    bool orthorombic_{true};

    mdarray<T, 3, CblasRowMajor> Vtmp;

    int lmax_{0};
    T basis_[3][3];
    int period_[3];
    int grid_size_[3];

    // angular momentum for the cartesian harmonics
    std::vector<int> li_lj_;

    // coefficients of the gaussians [always by pair]
    std::vector<T> eta_ij;

    // radius. They will be direction dependent for the non orthorombic case
    std::vector<T> radius_;

    // position of the atoms
    std::vector<T> &ri_;
    std::vector<T> &rj_;

    std::vector<size_t> offset_;
    std::vector<T> coef_data_;

    // weights of the gaussians
    mdarray<T, 4, CblasRowMajor> coefs_;
    mdarray<T, 3, CblasRowMajor> transformed_coefs_;

    // contain the nonzero elements of the polynomials along x, y, z
    mdarray<T, 3, CblasRowMajor> p_alpha_reduced_;

    // contain the polynomials computed along x,y,z with periodic boundaries
    // conditions
    mdarray<T, 4, CblasRowMajor> p_alpha_folded_;

    // polynomials computed from -radius..radius
    mdarray<T, 3, CblasRowMajor> p_alpha_;

    // non vanishing elements of the polynomials (only the position)
    mdarray<int, 2, CblasRowMajor> yz_non_vanishing_elements_;

    gaussian(const int grid_size[3],
             const int lower_bound[3],
             const int upper_bound[3],
             const int period[3],
             const T dr[3],
             const T *basis,
             const int lmax,
             const bool ortho=true)
        {
            orthorombic_ = ortho;

            period_[0] = period[0];
            period_[1] = period[1];
            period_[2] = period[2];

            grid_size_[0] = grid_size[0];
            grid_size_[1] = grid_size[1];
            grid_size_[2] = grid_size[2];

            basis_[0][0] = basis[0];
            basis_[0][1] = basis[1];
            basis_[0][2] = basis[2];
            basis_[1][0] = basis[3];
            basis_[1][1] = basis[4];
            basis_[1][2] = basis[5];
            basis_[2][0] = basis[6];
            basis_[2][1] = basis[7];
            basis_[2][2] = basis[8];

            lmax_ = lmax;

            p_alpha_folded_ = mdarray<T, 4, CblasRowMajor>(3, lmax_ + 1, lmax_ + 1, std::max(std::max(period[0],
                                                                                                           period[1]),
                                                                                                  period[2]));

            p_alpha_reduced_ = mdarray<T, 4, CblasRowMajor>(3, lmax_ + 1, lmax_ + 1, std::max(std::max(grid_size_[0],
                                                                                                       grid_size_[1]),
                                                                                              grid_size_[2]));

            // p_alpha_ depends on external parameters we do not know to begin
            // with
        }
};

template<typename T> void collocate_core(mdarray<T, 3, CblasRowMajor> &Vtmp,  const mdarray<T, 3, CblasRowMajor> &co, const mdarray<T, 3, CblasRowMajor> &p_alpha_beta_reduced_)
{

    timer.start("init");

    mdarray<T, 3, CblasRowMajor> C = mdarray<T, 3, CblasRowMajor>(co.size(0), co.size(1), Vtmp.size(1));
    mdarray<T, 3, CblasRowMajor> xyz_alpha_beta = mdarray<T, 3, CblasRowMajor>(co.size(0), Vtmp.size(0), Vtmp.size(1));

    // C.zero();
    // xyz_alpha_beta.zero();
    timer.stop("init");

    if (co.size(0) > 1) {
        timer.start("dgemm");
#ifdef HAVE_LIBXSMM
        int prefetch = LIBXSMM_PREFETCH_AUTO;
        int flags = 0; /* LIBXSMM_FLAGS */
        libxsmm_dmmfunction xmm = NULL;
        double alpha = 1, beta = 0;
        xmm = libxsmm_dmmdispatch(co.size(2)/*m*/, n[1]/*n*/, co.size(2)/*k*/,
                                  NULL, NULL/*ldb*/, NULL/*ldc*/,
                                  &alpha, &beta, &flags, &prefetch);
        const int asize = co.ld() * co.size(1);
        const int bsize = p_alpha_beta_reduced_.ld() * co.size(2);
        const int csize = C.ld() * c.size(1);

        for (i = 0; i < (n - 1); ++i) {
            const double *const ai = a + i * asize;
            const double *const bi = b + i * bsize;
            double *const ci = c + i * csize;
            xmm(ai, bi, ci, ai + asize, bi + bsize, ci + csize);
        }

        xmm(a + (n - 1) * asize, b + (n - 1) * bsize, c + (n - 1) * csize,
            /* pseudo prefetch for last element of batch (avoids page fault) */
            a + (n - 1) * asize, b + (n - 1) * bsize, c + (n - 1) * csize);
#else
// we can batch this easily
        for (int a1 = 0; a1 < co.size(0); a1++) {

            // we need to replace this with libxsmm
            cblas_dgemm(CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        co.size(2),
                        Vtmp.size(1),
                        co.size(2),
                        1.0,
                        co.template at<CPU>(a1, 0, 0), // Coef_{alpha,gamma,beta}
                        co.ld(),
                        p_alpha_beta_reduced_.template at<CPU>(1, 0, 0), // Y_{beta,j}
                        p_alpha_beta_reduced_.ld(),
                        0.0,
                        C.template at<CPU>(a1, 0, 0), // tmp_{alpha, gamma, j}
                        C.ld());
        }
#endif

        for (int a1 = 0; a1 < co.size(0); a1++) {
            cblas_dgemm(CblasRowMajor,
                        CblasTrans,
                        CblasNoTrans,
                        Vtmp.size(0),
                        Vtmp.size(1),
                        co.size(1),
                        1.0,
                        p_alpha_beta_reduced_.template at<CPU>(0, 0, 0), // Z_{gamma,k} -> I need to transpose it I want Z_{k,gamma}
                        p_alpha_beta_reduced_.ld(),
                        C.template at<CPU>(a1, 0, 0), // C_{gamma, j} = Coef_{alpha,gamma,beta} Y_{beta,j} (fixed alpha)
                        C.ld(),
                        0.0,
                        xyz_alpha_beta.template at<CPU>(a1, 0, 0), // contains xyz_{alpha, kj} the order kj is important
                        xyz_alpha_beta.ld());
        }

        cblas_dgemm(CblasRowMajor,
                    CblasTrans,
                    CblasNoTrans,
                    Vtmp.size(0) * Vtmp.size(1),
                    Vtmp.size(2),
                    co.size(2),
                    1.0,
                    xyz_alpha_beta.template at<CPU>(0, 0, 0),
                    xyz_alpha_beta.size(1) * xyz_alpha_beta.size(2),
                    p_alpha_beta_reduced_.template at<CPU>(2, 0, 0),
                    p_alpha_beta_reduced_.ld(),
                    0.0,
                    Vtmp.template at<CPU>(0, 0, 0),
                    Vtmp.ld());
        timer.stop("dgemm");
    } else {
        for (int z1 = 0; z1 < Vtmp.size(0); z1++) {
            const T tz = co(0, 0, 0) * p_alpha_beta_reduced_(0, 0, z1);
            for (int y1 = 0; y1 < Vtmp.size(1); y1++) {
                const T tmp = tz * p_alpha_beta_reduced_(1, 0, y1);
                const T *__restrict src = p_alpha_beta_reduced_.template at<CPU>(2, 0, 0);
                T *__restrict dst = Vtmp.template at<CPU>(z1, y1, 0);
                for (int x1 = 0; x1 < Vtmp.size(2); x1++) {
                    dst[x1] = tmp * src[x1];
                }
            }
        }
    }
}


// the indices are encoded that way for a given l, alpha, beta, gamma (gamma = l - alpha - beta)
// coef[offset_[l] - offset_[lmin] + a1 (l - a1 + 1) b1]
// compute_nonzero_values(center[0], period[0], length_p[0], corner[0], Potential.size(0), 0, yz_non_vanishing_elements);

void compute_nonzero_values(const int center, // center of the gaussian
                            const int period, // size of the global grid, which
                                              // might corresponds to the period
                            const int length, // it is the [-cutoff..cutoff] of the gaussian in unit of dr
                            const int lower_corner, // starting point of the local grid
                            const int length_dir, // not used
                            const int dir, // direction x, y, z
                            mdarray<int, 2, CblasRowMajor> &non_zeroes_)
{
    for (int  i = 0; i < non_zeroes_.size(1); i++) {
        non_zeroes_(dir, i) = 0;
    }

    if (length >= period) {
        for (int  i = 0; i < non_zeroes_.size(1); i++) {
            non_zeroes_(dir, i) = 1;
        }
        return;
    }

    int p0 = center - (length - 1) / 2;
    for (int i = 0; i < length; i++) {
        non_zeroes_(dir, (i + p0 + period) % period) = 1;
    }
    return;
}

void compute_nonzero_values(const int center,
                            const int period,
                            const int length,
                            const int lower_corner,
                            const int length_dir,
                            const int dir,
                            std::vector<int> &non_zeroes_)
{
    std::vector<int> tmp(period, 0);
    for (int  i = 0; i < non_zeroes_.size(); i++) {
        non_zeroes_[i] = 0;
    }

    if (length >= period) {
        for (int  i = 0; i < non_zeroes_.size(); i++) {
            non_zeroes_[i] = 1;
        }
        goto label;
    }

    if ((period - center) >= (length + 1) / 2) {
        for (int i = center; i < center + (length + 1) / 2 - 1; i++) {
            tmp[i]  = 1;
        }
    } else {
        if ((length + 1) / 2 >= period) {
            for (int i = 0; i < period; i++) {
                tmp[i] = 1;
            }
            goto label;
        } else {
            for (int i = center; i < period; i++) {
                tmp[i] = 1;
            }
            for (int i = 0; i < (length + 1) / 2 - period + center - 1; i++) {
                tmp[i] = 1;
            }
        }
    }

    if (center >= (length + 1) / 2) {
        for (int i = 0; i < (length + 1) / 2; i++) {
            tmp[i + center - (length + 1) / 2] = 1;
        }
        goto label;
    } else {
        if ((length + 1) / 2 >= period) {
            for (int i = 0; i < period; i++) {
                tmp[i] = 1;
            }
        } else {
            for (int i = 0; i < center; i++) {
                tmp[i] = 1;
            }

            for (int i = period + center - (length + 1) / 2; i < period; i++) {
                tmp[i] = 1;
            }
        }
    }

label:
    for (int  i = 0; i < length_dir; i++)
        non_zeroes_[i] = tmp[i + lower_corner];

    tmp.clear();
    return;
}


// the initial table contains a gaussian. But periodic boundaries conditions
// screw up thing a bit and we have to sum up multiple copies of the gaussian
// (the center been shifted by a period). this boils down to split the table in
// intervals of length corresponding to the period of the grid and then add them
// together elementwise

// basically we do this F(x) = \sum_{n} G_a(n L + x), where G_a is a "gaussian"
// centered around x_a and x is the grid coordinates.


template <typename T> void compute_folded_integrals(const int center,
                                                    const std::vector<T> &unfolded_table,
                                                    std::vector<T> &folded_table)
{
    const int length = unfolded_table.size();
    const int period = folded_table.size();

    const int p = (center - (length + 1) / 2 + 16 * period) % period;
    const int start = period - p;

    for (int i = 0; i < std::min(start, length); i++)
        folded_table[i + p] += unfolded_table[i];

    if (length > start) {
    // we need the remaining elements of the unfolded table : # of elements length - start
        for (int i = 0; i < length - start;) {
            const T *__restrict src = &unfolded_table[i + start];
            T *__restrict dst = &folded_table[0];
            if (i + start + period < length) {
                for (int k = 0; k < period; k++) {
                    dst[k] += src[k];
                }
            } else {
                for (int k = 0; (k + i) < length - start; k++)
                    dst[k] += src[k];
            }
            i += period;
        }
    }
}

template <typename T> void compute_folded_integrals(const int center,
                                                    const int period,
                                                    const int length,
                                                    const T *__restrict unfolded_table,
                                                    T *__restrict folded_table)
{
    // it compute the following thing. Take the gaussian extension such as
    //
    // | ----------------------------------------------------| gaussian table
    //
    //                |------------------------| size of the grid
    //
    // the periodic boundary condition applied to the grid imposes to
    // fold the tail and head of the gaussian or the grid.
    // we do simply this in unoptimised loop

    // for (int i = 0; i < gaussian.size(); i++) {
    //     grid[(i + center + 32 * grid.size()) % grid.size()] += gaussian[i]
    //}

    // the modulo operation is costly so basically to avoid it we
    // artificially split the gaussian table like this

    // | --------------)-----------------------(---------------| gaussian table
    //
    //                |------------------------| size of the grid

    // and do this
    //                |------------------------|
    //
    //         +
    //                )------------------------(
    //
    //         +
    //                         | --------------)
    //
    //         +      (---------------|

    // we only have to do that 3 times because the integral or
    // (collocation here) can be decomposed into x, y, and z.

    // computational cost : alpha n (n being the size of the collocation
    // grid in a given direction), alpha is a small number.

    const int start = period - ((length - 1) / 2 - center + 32 * period) % period;
    const int p0 = ((length - 1) / 2 - center + 32 * period) % period;
    const T *__restrict src = &unfolded_table[0];

    for (int i = start; i < std::min(period, length + start); i++) {
        folded_table[i] += src[i - start];
    }

    src += period - start;

    // we need the remaining elements of the unfolded table : # of elements length - start

    for (int i = 0; i < length - p0; i += period) {
        for (int k = 0; k < std::min(period, length - i - p0); k++)
            folded_table[k] += src[k];
        src += period;
    }
}

void find_interval(const std::vector<int> &folded_table_index, int *__restrict interval_, int &kmax_) {

    for (int i = 0; i < 16; i++)
        interval_[i] = 0;

    int k = 0;

    k += folded_table_index[0];

    for (int i = 0; i < folded_table_index.size() - 1; i++) {
        int test1 = folded_table_index[i + 1] > folded_table_index[i];
        int test2 = folded_table_index[i + 1] < folded_table_index[i];
        interval_[k] += test1 * (i + 1) + test2 * i;
        k += test1 + test2;
    }
    interval_[k] += folded_table_index[folded_table_index.size() - 1] * (folded_table_index.size() - 1);
    kmax_ = k;
}

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

template<typename T> void calculate_polynomials(const int l,
                                                const T radius,
                                                const T *dr,
                                                const T *r_a,
                                                const T mu_mean,
                                                mdarray<T, 3, CblasRowMajor> &p_alpha,
                                                int *length__)
{
    // l is the angular momentum. This gives the degree of the polynomial. For
    // the forces we need polynomials of degree l + 1

    // first compute the max of n_alpha

    const int n_max = radius / std::min(dr[0], std::min(dr[1], dr[2])) + 1;
    length__[0] = (radius / dr[0] + 1);
    length__[1] = (radius / dr[1] + 1);
    length__[2] = (radius / dr[2] + 1);
    // we compute for -n_max * dr to n_max * dr

    if (p_alpha.size())
        p_alpha.clear();

    p_alpha = mdarray<T, 3, CblasRowMajor>(3, l + 1, 2 * n_max + 1);
    p_alpha.zero();

    // std::cout << r_a[0] << " " << r_a[1] << " " << r_a[2] << "\n";
    // std::cout << std::fmod(r_a[0], dr[0]) << " " << std::fmod(r_a[1], dr[1]) << " " << std::fmod(r_a[2], dr[2]) << "\n";

    for (int d = 0; d < 3; d++) {
        T *__restrict dst = p_alpha.template at<CPU>(d, 0, 0) + length__[d];
        double rab = std::fmod(r_a[d], dr[d]);
        for (int i = - length__[0]; i <= length__[0]; i++) {
            dst[i] = exp(-mu_mean * (((double)i) * dr[d] + rab) * (((double)i) * dr[d] + rab));
        }

        // we add the extra exponent for the forces
        // [0..n_max-1] positive r_alpha from 0..n_max * dr_alpha
        // [n_max,.. 2 n_max + 1] negative r_alpha
        // fft convention for the frequency domain
        rab = std::fmod(r_a[d], dr[d]);
        for (int m = 1; m < l + 1; m++) {
            // ugly as hell. I shift the pointer addresses
            const T *__restrict src = p_alpha.template at<CPU>(d, m - 1, 0) + length__[d];
            T *__restrict dst = p_alpha.template at<CPU>(d, m, 0) + length__[d];
            for (int i = - length__[d]; i <= length__[d]; i++) {
                dst[i] = (((double)i) * dr[d] + rab) * src[i];
            }
        }
    }

    length__[0] = 2 * length__[0] + 1;
    length__[1] = 2 * length__[1] + 1;
    length__[2] = 2 * length__[2] + 1;
}

template <typename T> void compute_folded_integrals_naive(const int center,
                                                          const std::vector<T> &unfolded_table,
                                                          std::vector<T> &folded_table)
{
    const int period = folded_table.size();
    const int length = unfolded_table.size();
    for (int i = 0; i < unfolded_table.size(); i++)
        folded_table[(center - (length + 1) / 2 + i + 16 * period) % period] += unfolded_table[i];
}

/*  specific to the orthorombic case. We can not apply it for the generic case */
template <typename T> void compute_folded_integrals_all_powers(const int center[3], // center of the gaussian
                                                               const int period[3], // period of the grid.
                                                               const int length[3], // number of points in each direction where the gaussian is evaluated.
                                                               const mdarray<T, 3, CblasRowMajor> &p_alpha,
                                                               const mdarray<T, 3, CblasRowMajor> &p_beta,
                                                               mdarray<T, 4, CblasRowMajor> &p_alpha_beta_folded)
{
    std::vector<T> tmp(p_alpha.size(2), 0.0);
    p_alpha_beta_folded.zero();
    for (int deg = 0; deg < 3; deg++) {
        for (int a1 = 0; a1 < p_alpha.size(1); a1++) {
            const T *__restrict p_a = p_alpha.template at<CPU>(deg, a1, 0);
            for (int a2 = 0; a2 < p_beta.size(1); a2++) {
                T *__restrict p_ab = &tmp[0];
                const T *__restrict p_b = p_beta.template at<CPU>(deg, a2, 0);
                for (int s = 0; s < p_alpha.size(2); s++)
                    tmp[s] = p_a[s] * p_b[s];

                compute_folded_integrals<T>(center[deg],
                                            period[deg],
                                            p_alpha.size(2),
                                            &tmp[0],
                                            p_alpha_beta_folded.template at<CPU>(deg, a1, a2, 0));
            }
        }
    }

    tmp.clear();
}

template <typename T> void compute_folded_integrals_all_powers(const int center[3],
                                                               const int period[3],
                                                               const int length[3],
                                                               const mdarray<T, 3, CblasRowMajor> &p_alpha,
                                                               mdarray<T, 3, CblasRowMajor> &p_alpha_beta_folded)
{
    p_alpha_beta_folded.zero();
    for (int deg = 0; deg < 3; deg++) {
        for (int a1 = 0; a1 < p_alpha.size(1); a1++) {
            const T *__restrict p_a = p_alpha.template at<CPU>(deg, a1, 0);
            compute_folded_integrals<T>(center[deg],
                                        period[deg],
                                        length[deg],
                                        p_alpha.template at<CPU>(deg, a1, 0),
                                        p_alpha_beta_folded.template at<CPU>(deg, a1, 0));
        }
    }
}

template <typename T> void apply_non_orthorombic_corrections(const T*__restrict r_ab,
                                                             const T mu_mean,
                                                             const T *dr,
                                                             const T*__restrict basis,
                                                             const int *corner,
                                                             const int *period,
                                                             mdarray<T, 3, CblasRowMajor> &Potential)
{
    const int max_size = std::max(std::max(Potential.size(0), Potential.size(1)), Potential.size(2));
    mdarray<T, 3, CblasRowMajor> tmp(3, max_size, max_size);
    const mdarray<T, 2, CblasRowMajor> basis_c(basis, 3, 3);
    const T c[3];
    const int n[3][2] = {{0, 1},
                         {0, 2},
                         {1, 2}};
    /* alpha beta */
    c[0] = 2.0 * (basis_c(0, 0) * basis_c(1, 0) + basis_c(0, 1) * basis_c(1, 1) + basis_c(0, 2) * basis_c(1, 2));
    /* alpha gamma */
    c[1] = 2.0 * (basis_c(0, 0) * basis_c(2, 0) + basis_c(0, 1) * basis_c(2, 1) + basis_c(0, 2) * basis_c(2, 2));
    /* beta gamma */
    c[2] = 2.0 * (basis_c(1, 0) * basis_c(2, 0) + basis_c(1, 1) * basis_c(2, 1) + basis_c(1, 2) * basis_c(2, 2));

    tmp.zero();

    for (int dir = 0; dir < 3; dir++) {
        int d1 = n[dir][0];
        int d2 = n[dir][1];

        const T coef = c[0] * ((d1 == 0) && (d2 == 1)) +
            c[1] * ((d1 == 0) && (d2 == 2)) +
            c[2] * ((d1 == 1) && (d2 == 2));

        for (int alpha = 0; alpha < Potential.size(d1); alpha++) {
            double alpha_d = (alpha + corner[d1]) * dr[d1] - period[d1] * dr[d1] / 2.0 - r_ab[d1];
            for (int beta = 0; beta < Potential.size(d2); beta++) {
                int beta_d = (beta + corner[d2]) * dr[d2] - period[d2] * dr[d2] / 2.0  - r_ab[d2];
                tmp(dir, alpha, beta) = exp(-coef * mu_mean * alpha_d * beta_d);
            }
        }
    }

    /*
      for non orthorombic case n_ijk = n_ijk ^ortho T^z_ij T^y_ik T^x_jk
     */

    for (int z = 0; z < Potential.size(0); z++) {
        const T*__restrict src1 = tmp(1, z, 0); // z, x
        for (int y = 0; y < Potential.size(1); y++) {
            const T constant = tmp(0, z, y); // z, y
            const T*__restrict src2 = tmp(2, y, 0); // y, x
            T*__restrict dst = Potential.template at<CPU>(z, y, 0);
            for (int x = 0; x < Potential.size(2); x++) {
                dst[x] *= src1[x] * src2[x] * constant;
            }
        }
    }
}

inline int return_linear_index_from_exponents(const int alpha, const int beta, const int gamma)
{
    const int l = alpha + beta + gamma;
    return (l - alpha) * (l - alpha - 1) / 2 + gamma;
}

/* precompute the coefficients of the polynomials for a fixed z coordinates and
 * values of the exponents a1, a2. the result is then something like this in
 * math notation

  A^z_{\alpha_1, \alpha_2, y}.

  to obtain the final result we just do this

  dens_{x,y,z} = \sum_{\alpha_1\alpha_2} Px_{\alpha_1\alpha_2,x} A^z_{\alpha1,alpha2,y}

  it is just a matrix-matrix multiplication when we collapse the index alpha_{1,2} together

*/
template <typename T> void compute_polynomial_coefficients(const mdarray<T, 2, CblasRowMajor> &coef,
                                                           const int lmin[2], const int lmax[2],
                                                           mdarray<T, 3, CblasRowMajor> &co)
{
    co.zero();
    for (int a1 = 0; a1 <= lmax[0]; a1++) {
        for (int a2 = 0; a2 <= lmax[1]; a2++) {
            double res = 0.0;
            for (int l1 = lmin[0]; l1 <= lmax[0]; l1++) {
                for (int l2 = lmin[1]; l2 <= lmax[1]; l2++) {
                    for (int b1 = 0; b1 <= l1 - a1; b1++) {
                        for (int b2 = 0; b2 <= l2 - a2; b2++) {
                            const int g1 = l1 - b1 - a1;
                            const int g2 = l2 - b2 - a2;
                            int i1 = offset_[l1] + return_linear_index_from_exponents(a1, b1, g1);
                            int i2 = offset_[l2] + return_linear_index_from_exponents(a2, b2, g2);
                            co(a1 * (lmax[1] + 1) + a2, g1 * (lmax[1] + 1) + g2, b1 * (lmax[1] + 1) + b2) += coef(i1, i2);
                        }
                    }
                }
            }
        }
    }
}

/* precompute the coefficients of the polynomials derivatives (needed for the
 * kinetic energy) for a fixed z coordinates and values of the exponents a1, a2.
 * the result is then something like this in math notation

  A^z_{\alpha_1, \alpha_2, y}.

  to obtain the final result we just do this

  dens_{x,y,z} = \sum_{\alpha_1\alpha_2} Px_{\alpha_1\alpha_2,x} A^z_{\alpha1,alpha2,y}

  it is just a matrix-matrix multiplication when we collapse the index alpha_{1,2} together

*/
template <typename T> void compute_polynomial_coefficients_derivatives(const mdarray<T, 2, CblasRowMajor> &coef,
                                                                       const double eta_1, const double eta_2,
                                                                       const int lmin[2], const int lmax[2],
                                                                       mdarray<T, 3, CblasRowMajor> &co)
{
    co.zero();
    for (int a1 = 0; a1 <= lmax[0]; a1++) {
        for (int a2 = 0; a2 <= lmax[1]; a2++) {
            double res = 0.0;
            for (int l1 = lmin[0]; l1 <= lmax[0]; l1++) {
                for (int l2 = lmin[1]; l2 <= lmax[1]; l2++) {
                    for (int b1 = 0; b1 <= l1 - a1; b1++) {
                        for (int b2 = 0; b2 <= l2 - a2; b2++) {
                            const int g1 = l1 - b1 - a1;
                            const int g2 = l2 - b2 - a2;
                            const int i1 = offset_[l1] + return_linear_index_from_exponents(a1, b1, g1);
                            const int i2 = offset_[l2] + return_linear_index_from_exponents(a2, b2, g2);
                            const T coefs = coef(i1, i2);
                            {
                                const int a11 = a1 - 1;
                                const int a12 = a1 + 1;
                                const int a21 = a2 - 1;
                                const int a22 = a2 + 1;

                                /* -2 eta_2 alpha_1  (alpha_1 - 1, alpha_2 + 1)*/
                                if (a11 >= 0)
                                    co(a11 * (lmax[1] + 2) + a22 + 1,
                                       g1 * (lmax[1] + 2) + g2,
                                       b1 * (lmax[1] + 2) + b2) -= 2.0 * a1 * eta_2 * coefs;

                                /* -2 eta_1 alpha_2 (alpha_1 + 1, alpha_2 - 1) */

                                if (a21 >= 0)
                                    co(a12 * (lmax[1] + 2) + a21,
                                       g1 * (lmax[1] + 2) + g2,
                                       b1 * (lmax[1] + 2) + b2) -= 2.0 * a2 * eta_1 * coefs;

                                /*
                                  alpha_1 alpha_2 (a1 - 1, a2 - 1)
                                */

                                if ((a11 >= 0) && (a21 >= 0))
                                    co(a11 * (lmax[1] + 2) + a22,
                                       g1 * (lmax[1] + 2) + g2,
                                       b1 * (lmax[1] + 2) + b2) += a2 * a1 * coefs;

                                /*
                                  eta_1 * eta_2 * 4 (a1 + 1, a2 + 1)
                                */

                                co(a12 * (lmax[1] + 2) + a22,
                                   g1 * (lmax[1] + 2) + g2,
                                   b1 * (lmax[1] + 2) + b2) += 4.0 * eta_1 * eta_2 * coefs;
                            }

                            {
                                /* same with beta D[P, y]*/
                                const int b11 = b1 - 1;
                                const int b12 = b1 + 1;
                                const int b21 = b2 - 1;
                                const int b22 = b2 + 1;

                                /* -2 eta_2 alpha_1  (x)^a1-1 x^ a2 + 1*/
                                if (b11 >= 0)
                                    co(a1 * (lmax[1] + 2) + a2,
                                       g1 * (lmax[1] + 2) + g2,
                                       b11 * (lmax[1] + 2) + b22) -= 2.0 * b1 * eta_2 * coefs;

                                /* -2 eta_1 alpha_2 */
                                if (b21 >= 0)
                                    co((a1 + 1) * (lmax[1] + 2) + a2,
                                       g1 * (lmax[1] + 2) + g2,
                                       b12 * (lmax[1] + 2) + b21) -= 2.0 * b2 * eta_1 * coefs;

                                /* mixed term alpha_1 alpha_2 x ^ (a1 - 1) x ^ (a2 -
                                   1)
                                */

                                if ((b11 >= 0) && (b21 >= 0))
                                    co(a1 * (lmax[1] + 2) + a2,
                                       g1 * (lmax[1] + 2) + g2,
                                       b11 * (lmax[1] + 2) + b21) += b2 * b1 * coefs;

                                /* -2 eta_1 eta_2 */
                                co(a1 * (lmax[1] + 2) + a2,
                                   g1 * (lmax[1] + 2) + g2,
                                   b12 * (lmax[1] + 2) + b22) += 4.0 * eta_1 * eta_2 * coefs;
                            }
                            {
                                /* same with beta D[P, z]*/
                                const int g11 = g1 - 1;
                                const int g12 = g1 + 1;
                                const int g21 = g2 - 1;
                                const int g22 = g2 + 1;

                                /* -2 eta_2 alpha_1  (x)^a1-1 x^ a2 + 1*/
                                if (g11 >= 0)
                                    co(a1 * (lmax[1] + 2) + a2,
                                       g11 * (lmax[1] + 2) + g22,
                                       b1 * (lmax[1] + 2) + b2) -= 2.0 * g1 * eta_2 * coefs;

                                /* -2 eta_1 alpha_2 */
                                if (g21 >= 0)
                                    co((a1 + 1) * (lmax[1] + 2) + a2,
                                       g12 * (lmax[1] + 2) + g21,
                                       b1 * (lmax[1] + 2) + b2) -= 2.0 * g2 * eta_1 * coefs;

                                /* mixed term alpha_1 alpha_2 x ^ (a1 - 1) x ^ (a2 -
                                   1)
                                */

                                if ((g11 >= 0) && (g21 >= 0))
                                    co(a1 * (lmax[1] + 2) + a2,
                                       g11 * (lmax[1] + 2) + g21,
                                       b1 * (lmax[1] + 2) + b2) += g2 * g1 * coefs;

                                /* -2 eta_1 eta_2 */
                                co(a1 * (lmax[1] + 2) + a2,
                                   g12 * (lmax[1] + 2) + g22,
                                   b1 * (lmax[1] + 2) + b2) += 4.0 * eta_1 * eta_2 * coefs;
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename T> void compute_compact_polynomial_coefficients(const mdarray<T, 2, CblasRowMajor> &coef,
                                                                   const int lmin[2], const int lmax[2],
                                                                   const T *ra, const T *rb, const T *rab,
                                                                   mdarray<T, 3, CblasRowMajor> &co)
{
    // binomial coefficients n = 0 ... 20
    const int binomial[21][21] = {{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                  {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                  {1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                  {1, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                  {1, 4, 6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                  {1, 5, 10, 10, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                  {1, 6, 15, 20, 15, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                  {1, 7, 21, 35, 35, 21, 7, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                  {1, 8, 28, 56, 70, 56, 28, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                  {1, 9, 36,  84, 126, 126, 84, 36, 9, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                  {1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0},
                                  {1, 11, 55, 165, 330, 462, 462, 330, 165, 55, 11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                  {1, 12, 66, 220, 495, 792, 924, 792, 495, 220, 66, 12, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                                  {1, 13, 78, 286, 715, 1287, 1716, 1716, 1287, 715, 286, 78, 13, 1, 0, 0, 0, 0, 0, 0, 0},
                                  {1, 14, 91, 364, 1001, 2002, 3003, 3432, 3003, 2002, 1001, 364, 91, 14, 1, 0, 0, 0, 0, 0, 0},
                                  {1, 15, 105, 455, 1365, 3003, 5005, 6435, 6435, 5005, 3003, 1365, 455, 105, 15, 1, 0, 0, 0, 0, 0},
                                  {1, 16, 120, 560, 1820, 4368, 8008, 11440, 12870, 11440, 8008, 4368, 1820, 560, 120, 16, 1, 0, 0, 0, 0},
                                  {1, 17, 136, 680, 2380, 6188, 12376, 19448, 24310, 24310, 19448, 12376, 6188, 2380, 680, 136, 17, 1, 0, 0, 0},
                                  {1, 18, 153, 816, 3060, 8568, 18564, 31824, 43758, 48620, 43758, 31824, 18564, 8568, 3060, 816, 153, 18, 1, 0, 0},
                                  {1, 19, 171, 969, 3876, 11628, 27132, 50388, 75582, 92378, 92378, 75582, 50388,  27132, 11628, 3876, 969, 171, 19, 1, 0},
                                  {1, 20, 190, 1140, 4845, 15504, 38760, 77520, 125970, 167960, 184756, 167960, 125970, 77520, 38760, 15504, 4845, 1140, 190, 20, 1}};

    mdarray<T, 4, CblasRowMajor> power(2, 3, lmax[0] + lmax[1] + 1, lmax[0] + lmax[1] + 1);

    /* I compute (x - xa) ^ k Binomial(alpha, k), for alpha = 0.. l1 + l2 + 1
     * and k = 0 .. l1 + l2 + 1. It is used everywhere here and make the economy
     * of multiplications and function calls
     */
    timer.start("compute_compact_polynomial_coefficients_init");
    for (int dir = 0; dir < 3; dir++) {
        T tmp = rab[dir] - ra[dir];
        power(0, dir, 0, 0) = 1.0;
        for (int k = 1; k < lmax[0] + lmax[1] + 1; k++)
            power(0, dir, 0, k) = tmp * power(0, dir, 0, k - 1);

        for (int k = 1; k < lmax[0] + lmax[1] + 1; k++)
            memcpy(power.template at<CPU>(0, dir, k ,0),
                   power.template at<CPU>(0, dir, k - 1, 0),
                   sizeof(T) * power.size(3));

        tmp = rab[dir] - rb[dir];

        power(1, dir, 0, 0) = 1.0;
        for (int k = 1; k < lmax[0] + lmax[1] + 1; k++)
            power(1, dir, 0, k) = tmp * power(1, dir, 0, k - 1);

        for (int k = 1; k < lmax[0] + lmax[1] + 1; k++)
            memcpy(power.template at<CPU>(1, dir, k ,0),
                   power.template at<CPU>(1, dir, k - 1, 0),
                   sizeof(T) * power.size(3));

        for (int a1 = 0; a1 < lmax[0] + lmax[1] + 1; a1++)
            for (int k = 0; k < lmax[0] + lmax[1] + 1; k++)
                power(0, dir, a1, k) *= binomial[a1][k];

        for (int a1 = 0; a1 < lmax[0] + lmax[1] + 1; a1++)
            for (int k = 0; k < lmax[0] + lmax[1] + 1; k++)
                power(1, dir, a1, k) *= binomial[a1][k];

    }
    timer.stop("compute_compact_polynomial_coefficients_init");

    for (int l1 = lmin[0]; l1 <= lmax[0]; l1++) {
        for (int l2 = lmin[0]; l2 <= lmax[1]; l2++) {
            for (int a1 = 0; a1 <= l1; a1++) {
                for (int a2 = 0; a2 <= l2; a2++) {
                    for (int b1 = 0; b1 <= l1 - a1; b1++) {
                        for (int b2 = 0; b2 <= l2 - a2; b2++) {
                            const int g1 = l1 - a1 - b1;
                            const int g2 = l2 - a2 - b2;

                            /* Warning Warning Warning. I permute the gamma and
                             * beta coefficients. I am in Row major order !!!!!

                               Note : the polynomials are \f[
                               \sum_\alpha\beta\gamma C_\alpha\beta\gamma (x - x_1)^alpha (y - y_1)^beta (z - z_1) ^ gamma
                               ]

                               // the fast index is x, the slowest one is z. but
                               // I have to go backward since the grid is stored
                               // latest index is the fastest.

                               So I do first \sum_\beta C_\alpha\beta\gamma (y -
                               y_1)^beta to have a table (\alpha\gamma , y_j),
                               which means I have to store C with the second and
                               third dimensions transposed

                               I effectively compute

                               T1_{\alpha, \gamma, j} = \sum_\beta C_\alpha\gamma\beta (y - y_1)^beta

                               and then I do (for fixed alpha)

                               T1_{\alpha,\gamma,j} ^ T . ((z - z_1) ^gamma)



                            */
                            const int i1 = offset_[l1] + return_linear_index_from_exponents(a1, b1, g1);
                            const int i2 = offset_[l2] + return_linear_index_from_exponents(a2, b2, g2);
                            const T coef_i1_i2 = coef(i1, i2);
                            for (int k1 = 0; k1 <= a1; k1++) {
                                const T cst1 = coef_i1_i2 * power(0, 0, a1, a1 - k1);
                                for (int k2 = 0; k2 <= a2; k2++) {
                                    const T cst2 =  cst1 * power(1, 0, a2, a2 - k2);
                                    for (int k5 = 0; k5 <= g1; k5++) {
                                        const T cst3 = cst2 * power(0, 2, g1, g1 - k5);
                                        // Binomial(a, a - b) = Binomial(a, b)
                                        for (int k6 = 0; k6 <= g2; k6++) {
                                            const T cst4 = cst3 * power(1, 2, g2, g2 - k6);
                                            for (int k3 = 0; k3 <= b1; k3++) {
                                                const T cst5 = cst4 * power(0, 1, b1, b1 - k3);
                                                T *__restrict dst = co.template at<CPU>(k1 + k2, k5 + k6, k3);
                                                for (int k4 = 0; k4 <= b2; k4++) {
                                                    dst[k4] +=  cst5 * power(1, 1, b2, b2 - k4);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename T> void compute_polynomials_non_zero_elems(const mdarray<T, 3, CblasRowMajor> &p_alpha_beta_folded,
                                                              const mdarray<int, 2, CblasRowMajor> &yz_non_vanishing_elements,
                                                              const int grid_lower_corner[3],
                                                              const int grid_size[3],
                                                              int *n,
                                                              mdarray<T, 3, CblasRowMajor> &p_alpha_beta_reduced_)
{
    // This works only if we can break the sum in product of sums along x, y, z

    // compute the number of gaussian products I have to consider for this part
    // of the grid. The grid might be distributed and the gaussian sum may only
    // be partial (for instance if the gaussians sit at the boundaries between
    // neighboring mpi ranks).

    for (int d = 0; d < 3; d++) {
        n[d] = 0;
        for (int i = grid_lower_corner[d]; i < grid_size[d]; i++) {
            n[d] += yz_non_vanishing_elements(d, grid_lower_corner[d] + i);
        }
    }

    // imediately return if there is no element in any direction (this mean that
    // the gaussian sum has no contribution on this part of the grid)

    if ((n[0] == 0) || (n[1] == 0) || (n[2] == 0))
        return;

    p_alpha_beta_reduced_ = mdarray<T, 3 , CblasRowMajor>(3,
                                                         p_alpha_beta_folded.size(1),
                                                         std::max(std::max(n[0], n[1]), n[2]));

    for (int d = 0; d < 3; d++) { // loop over x, y, z

        /* basically I take all non zero elements (marked with a x, zero elements are marked with -)
           the initial interval is (this is not a possible case).

           |-------xxxxxxxxx---------xxxxxxxxx-----|
                      s                 e

           and store contiguously the non zero elements between the s and e markers (e excluded)

           The only possible case for the initial table is

           |--------xxxxxxxxxxxxxxxxxxxxxxxx-------|

           or

           |xxxxxxxxx-----------------------xxxxxxx|

           nothing else
        */

        int offset = 0;
        for (int z = 0; z < grid_size[d]; z++) {

            // continue loop without doing anything if the contribution to the
            // total sum is zero
            if (!yz_non_vanishing_elements(d, grid_lower_corner[d] + z))
                continue;

            // we found the first non vanishing element
            int zmin = z + grid_lower_corner[d];

            for (;yz_non_vanishing_elements(d, grid_lower_corner[d] + z) && (z < (grid_size[d] - 1));z++);

            // we found the last one
            int zmax = z + grid_lower_corner[d];
            if (yz_non_vanishing_elements(d, grid_lower_corner[d] + z))
                 zmax += 1;

            // do brute force copy
            for (int a1 = 0; a1 < p_alpha_beta_folded.size(1); a1++) {
                memcpy(p_alpha_beta_reduced_.template at<CPU>(d, a1, offset),
                       p_alpha_beta_folded.template at<CPU>(d, a1, zmin),
                       sizeof(T) * (zmax - zmin));
            }

            // update the offset in case we have several intervals
            offset += zmax - zmin;
        }
    }
}

template <typename T> void collocate(const mdarray<T, 3, CblasRowMajor> &co,
                                     const mdarray<T, 3, CblasRowMajor> &p_alpha_beta_folded,
                                     const mdarray<int, 2, CblasRowMajor> &yz_non_vanishing_elements,
                                     const int grid_lower_corner[3],
                                     mdarray<T, 3, CblasRowMajor> &Vtmp,
                                     mdarray<T, 3, CblasRowMajor> &V)
{
    timer.start("alloc");
    mdarray<T, 3, CblasRowMajor> p_alpha_beta_reduced_;
    const int *__restrict zp_12_i = yz_non_vanishing_elements.template at<CPU>(0, 0);
    const int *__restrict yp_12_i = yz_non_vanishing_elements.template at<CPU>(1, 0);
    const int *__restrict xp_12_i = yz_non_vanishing_elements.template at<CPU>(2, 0);

    timer.stop("alloc");
    int n[3];
    int grid_size[3] = {V.size(0), V.size(1), V.size(2)};
    // computes Coef_{\alpha\gamma\beta}

    timer.start("compute_polynomials_non_zero_elems");
    compute_polynomials_non_zero_elems<T>(p_alpha_beta_folded,
                                          yz_non_vanishing_elements,
                                          grid_lower_corner,
                                          grid_size,
                                          n,
                                          p_alpha_beta_reduced_);
    timer.stop("compute_polynomials_non_zero_elems");

// return if there is nothing to do.
    if ((n[0] == 0) || (n[1] == 0) || (n[2] == 0))
        return;


    Vtmp.zero();
    timer.start("collocate_core");
    collocate_core<double>(Vtmp, co, p_alpha_beta_reduced_);
    timer.stop("collocate_core");
    // although I compute the block corresponding to the gaussians sum I need to
    // add it back to its position in the grid. The periodic boundaries
    // conditions and the fact that the grid might be distributed makes things a
    // tad complicated.
    int z_offset = 0;

    timer.start("sum_up");
    const int zsize = V.size(0);
    const int ysize = V.size(1);
    const int xsize = V.size(2);

    for (int z = 0; z < zsize; z++) {

        if (!zp_12_i[z + grid_lower_corner[0]]) {
            continue;
        }

        int zmin = z + grid_lower_corner[0];

        for (; zp_12_i[z + grid_lower_corner[0]] && (z < (zsize - 1)); z++);
        int zmax = z + grid_lower_corner[0];
        if (zp_12_i[z + grid_lower_corner[0]] && z == (zsize - 1))
            zmax += 1;

        int y_offset = 0;
        for (int y = 0; y < ysize; y++) {

            // search for the first point where the polynomial is above threhold.

            if (!yp_12_i[y + grid_lower_corner[1]])
                continue;

            int ymin = y + grid_lower_corner[1];
            for (; yp_12_i[y + grid_lower_corner[1]] && (y < ysize - 1); y++);

            int ymax = y + grid_lower_corner[1];
            if (yp_12_i[y + grid_lower_corner[1]] && y == (ysize - 1))
                ymax += 1;

            int x_offset = 0;
            for (int x = 0; x < xsize; x++) {
                if (!xp_12_i[x + grid_lower_corner[2]])
                    continue;

                int xmin = x + grid_lower_corner[2];
                for (; xp_12_i[x + grid_lower_corner[2]] && (x < xsize - 1); x++);

                int xmax = x + grid_lower_corner[2];
                if (xp_12_i[x + grid_lower_corner[2]] && x == (xsize - 1))
                    xmax += 1;

                for (int z1 = 0; z1 < (zmax - zmin); z1++) {
                    const int z2 = z_offset + z1;
                    const int z2p = z1 + zmin;
                    for (int y1 = 0; y1 < (ymax - ymin); y1++) {
                        const T*__restrict src = Vtmp.template at<CPU>(z2, y_offset + y1, x_offset);
                        T* __restrict dst = V.template at<CPU>(z2p, y1 + ymin, xmin);
                        for (int x1 = 0; x1 < (xmax - xmin); x1++) {
                            dst[x1] += src[x1];
                        }
                    }
                }
                x_offset = xmax - xmin;
            }
            y_offset += ymax - ymin;
        }
        z_offset += zmax - zmin;
    }
    timer.stop("sum_up");
}

void test_collocation_initialization()
{
    int length = 17;
    int period = 80;
    int center = 1;
    std::vector<int> non_zeroes_;
    std::vector<int> non_zeroes_test;
    std::vector<int> folded_table;
    std::vector<int> unfolded_table;
    mdarray<int, 2, CblasRowMajor> yz_non_vanishing_elements = mdarray<int, 2, CblasRowMajor>(3, period);

    non_zeroes_.resize(period);
    non_zeroes_test.resize(period);

    folded_table.resize(period);
    unfolded_table.resize(length);

    for (int i = 0; i < unfolded_table.size(); i++)
        unfolded_table[i] = 1;

    for (int center = 0; center < 30; center++) {
        for (int i = 0; i < non_zeroes_.size(); i++) {
            non_zeroes_[i] = 0;
            non_zeroes_test[i] = 0;
        }

        compute_nonzero_values(center, period, unfolded_table.size(), 0, period, 0, non_zeroes_);

        int interval_[16];
        int xmax;
        find_interval(non_zeroes_, interval_, xmax);

        for (int z = 0; z < xmax; z += 2)
            std::cout << interval_[z] << " " << interval_[z + 1] << std::endl;

        std::cout << center << " " << period << " " << length << " | test ";

        for (int i = 0; i < non_zeroes_.size(); i++)
            std::cout << non_zeroes_[i] << " ";
        std::cout << "\n";

        for (int i = 0; i < length; i++)
            non_zeroes_test[(center - (length + 1) / 2 + i + 16 * period) % period] = 1;

        std::cout << center << " " << period << " " << length << " | refs ";
        for (int i = 0; i < non_zeroes_test.size(); i++)
            std::cout << non_zeroes_test[i] << " ";
        std::cout << "\n";
    }

    std::cout << "\n\n" << std::endl;
    for (int center = 0; center < 30; center++) {
        memset(&folded_table[0], 0, sizeof(int) * folded_table.size());
        compute_folded_integrals<int>(center,
                                      unfolded_table,
                                      folded_table);

        std::cout << center << " " << period << " " << length << " | test ";
        for (int i = 0; i < folded_table.size(); i++)
            std::cout << folded_table[i];
        std::cout << "\n";

        memset(&non_zeroes_test[0], 0, sizeof(int) * folded_table.size());

        compute_folded_integrals_naive<int>(center,
                                            unfolded_table,
                                            non_zeroes_test);

        std::cout << center << " " << period << " " << length << " | refs ";
        for (int i = 0; i < non_zeroes_test.size(); i++)
            std::cout << non_zeroes_test[i];
        std::cout << "\n\n";
    }
}

int main(int argc, char **argv)
{
    // information about the gaussian
    double ra[] = {0.3779, 0.0, 0.0};
    double rb[] = {-0.3779, 0.0 , 0.0};
    double dr[] = {0.3779, 0.3779, 0.3779};
    double alpha_a = 0.5;
    double alpha_b = 0.5;
//		double alpha_a = 0.5;
//		double alpha_b = 0.5;
    int period[3] = {20, 20, 20};
    int center[3];
    int lmax[2] = {2, 2};
    int lmin[2] = {0, 0};
    double mu_ab, mu_mean, r_ab[3], kappa_ab;

    double radius = 5;

    mdarray<double, 2, CblasRowMajor> coefs;
    mdarray<double, 3, CblasRowMajor> Potential, Potential1;
    mdarray<double, 3, CblasRowMajor> p_alpha_beta_folded;
    mdarray<double, 3, CblasRowMajor> p_alpha;
    test_collocation_initialization();
    mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
    // just setup an arbitrary grid
    Potential = mdarray<double, 3, CblasRowMajor>(period[0], period[1], period[2]);
    Potential.zero();
    Potential1 = mdarray<double, 3, CblasRowMajor>(period[0], period[1], period[2]);
    Potential1.zero();

    coefs = mdarray<double, 2, CblasRowMajor>(offset_[lmax[0] + 1], offset_[lmax[1] + 1]);

    p_alpha_beta_folded =  mdarray<double, 3, CblasRowMajor>(3,
                                                             lmax[0] + lmax[1] + 1,
                                                             std::max(std::max(Potential.size(0),
                                                                               Potential.size(1)),
                                                                      Potential.size(2)));

    // non vanishing elements on this part of the grid
    mdarray<int, 2, CblasRowMajor> yz_non_vanishing_elements = mdarray<int, 2, CblasRowMajor>(3,
                                                                                              std::max(std::max(Potential.size(0),
                                                                                                                Potential.size(1)),
                                                                                                       Potential.size(2)));

    Potential.zero();

    mdarray<double, 3, CblasRowMajor> co = mdarray<double, 3, CblasRowMajor>(lmax[0] + lmax[1] + 1,
                                                                             lmax[0] + lmax[1] + 1,
                                                                             lmax[0] + lmax[1] + 1);

    mdarray<double, 3, CblasRowMajor> Vtmp = mdarray<double, 3, CblasRowMajor>(Potential.size(0),
                                                                               Potential.size(1),
                                                                               Potential.size(2));

    timer.start("total");
    for (int s = 0; s < 10000; s++) {
        if ((s % 1000) == 0)
            printf("%d\n", s);

        timer.start("one_set_of_gaussian");;
        for (int i = 0; i < coefs.size(0); i++)
            for (int j = 0; j < coefs.size(1); j++)
                coefs(i, j) = 1;

        calculate_center_of_two_gaussians(ra,
                                          rb,
                                          alpha_a,
                                          alpha_b,
                                          r_ab,
                                          &kappa_ab,
                                          &mu_ab,
                                          &mu_mean);

        center[0] = (r_ab[0] + 0.00000000001) / dr[0] + period[0] / 2;
        center[1] = (r_ab[1] + 0.00000000001) / dr[1] + period[0] / 2;
        center[2] = (r_ab[2] + 0.00000000001) / dr[2] + period[0] / 2;

#ifdef DEBUG
        printf("test : %.15lf %.15lf %.15lf\n", r_ab[0], r_ab[1], r_ab[2]);
        printf("test : %d %d %d\n", center[0], center[1], center[2]);
#endif

        // for (int i = 0; i < coefs.size(0); i++)
        //		for (int j = 0; j < coefs.size(1); j++)
        //				coefs(i, j) *= kappa_ab;
        /* This is NOT the most time consuming part. */
        int length_p[3];

        // compute (x - x12)^alpha exp(-...) for alpha = [0...l1 + l2]

        calculate_polynomials(lmax[0] + lmax[1],
                              4.179687500000000000000,
                              dr,
                              r_ab,
                              mu_mean,
                              p_alpha,
                              length_p);

        co.zero();
        timer.start("compute_compact_polynomial_coefficients");
        // apply the transformation (x-x1)(x-x2) -> sum bla bla (x-x12)
        compute_compact_polynomial_coefficients<double>(coefs, lmin, lmax, ra, rb, r_ab, co);
        timer.stop("compute_compact_polynomial_coefficients");


#ifdef DEBUG
        FILE *f1 = fopen("coefs.dat", "w+");
        fwrite(co.template at<CPU>(), sizeof(double), co.size(), f1);
        fclose(f1);
#endif
        p_alpha_beta_folded.zero();

        timer.start("polyall");
        compute_folded_integrals_all_powers<double>(center, // position of the gaussian center in int coordinates
                                                    period, // period of the grid.
                                                    length_p,
                                                    p_alpha,
                                                    p_alpha_beta_folded);
        timer.stop("polyall");

#ifdef DEBUG
        f1 = fopen("p_alpha_beta.dat","w+");
        fwrite(p_alpha_beta_folded.at<CPU>(), sizeof(double), p_alpha_beta_folded.size(), f1);
        fclose(f1);
#endif
        int corner[3] = {0, 0, 0};
        compute_nonzero_values(center[0], period[0], length_p[0], corner[0], Potential.size(0), 0, yz_non_vanishing_elements);
        compute_nonzero_values(center[1], period[1], length_p[1], corner[1], Potential.size(1), 1, yz_non_vanishing_elements);
        compute_nonzero_values(center[2], period[2], length_p[2], corner[2], Potential.size(2), 2, yz_non_vanishing_elements);

#ifdef DEBUG
        std::cout << center[0] << " "  << period[0] << " " << length_p[0] << std::endl;
        std::cout << center[1] << " "  << period[1] << " " << length_p[1] << std::endl;
        std::cout << center[2] << " "  << period[2] << " " << length_p[2] << std::endl;
        for (int i= 0; i < yz_non_vanishing_elements.size(1); i++)
            std::cout << yz_non_vanishing_elements(0, i) << " ";
        printf("\n");
        for (int i= 0; i < yz_non_vanishing_elements.size(1); i++)
            std::cout << yz_non_vanishing_elements(1, i) << " ";
        std::cout << std::endl;
#endif

        timer.start("collocate");
        collocate<double>(co,
                          p_alpha_beta_folded,
                          yz_non_vanishing_elements,
                          corner,
                          Vtmp,
                          Potential);
        timer.stop("collocate");

        // if (generic) {
        //     /* when the grid is not orthorombic, we pay the prize of computing 3
        //      * n ^ 2 exponentials and n ^ 3 multiplications. There are no ways
        //      * around it
        //     */

        //     /* r_ab should be in the basis coordinates */
        //     apply_non_orthorombic_corrections<double>(r_ab, mu_mean, basis, corner, Potential);
        // }
        timer.stop("one_set_of_gaussian");
    }
    timer.stop("total");

    // process timings
    const auto result = timer.process();

    // print default statistics
    std::cout << "Default statistic:" << std::endl;
    std::cout << result.print();

    FILE *f = fopen("test.dat", "w+");
    fwrite(Potential.at<CPU>(), sizeof(double), Potential.size(), f);
    fclose(f);
    p_alpha.clear();
    Potential.clear();
    Potential1.clear();
    coefs.clear();
    p_alpha_beta_folded.clear();
    return 0;
}
