#include <iostream>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>

template <typename T> void collocate_core_naive(const int *length_,
                                                const mdarray<T, 3, CblasRowMajor> &co,
                                                const mdarray<T, 3, CblasRowMajor> &p_alpha_beta_reduced_,
                                                mdarray<T, 3, CblasRowMajor> &Vtmp)
{
    Vtmp.zero();
    for (int alpha = 0; alpha < co.size(2); alpha++) {
        for (int gamma = 0; gamma < co.size(0); gamma++) {
            for (int beta = 0; beta < co.size(1); beta++) {
                double coef = co(alpha, gamma, beta);
                for (int z = 0; z < length_[0]; z++) {
                    double c1 = coef * p_alpha_beta_reduced_(0, gamma, z);
                    for (int y = 0; y < length_[1]; y++) {
                        double c2 = c1 * p_alpha_beta_reduced_(1, beta, y);
                        for (int x = 0; x < length_[2]; x++) {
                            Vtmp(z, y, x) += c2 * p_alpha_beta_reduced_(2, alpha, x);
                        }
                    }
                }
            }
        }
    }
}

template <typename T> bool test_collocate_core(const int i, const int j, const int k, const int lmax)
{
    mdarray<T, 3, CblasRowMajor> pol = mdarray<T, 3, CblasRowMajor>(3, lmax, std::max(std::max(i, j), k));
    mdarray<T, 3, CblasRowMajor> co = mdarray<T, 3, CblasRowMajor>(lmax, lmax, lmax);
    mdarray<T, 3, CblasRowMajor> Vgemm(i, j, k);
    mdarray<T, 3, CblasRowMajor> Vref(i, j, k);
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1.0, 1.0);
    int length[3] = {i, j, k};
    for (int s = 0; s < pol.size(); s++)
        pol[s] = distribution(generator);

    for (int s = 0; s < co.size(); s++)
        co[s] = distribution(generator);

    Vgemm.zero();
    collocate_core(length_,
                   co,
                   pol,
                   Vgemm);

    collocate_core_naive(length_,
                         co,
                         pol,
                         Vref);

    T maxi = -2.0;
    for (int l = 0; l < Vgemm.size(0); l++)
        for (int m = 0; m < Vgemm.size(1); m++)
            for (int n = 0; n < Vgemm.size(2); n++)
                maxi = std::max(std::abs(Vref(l, m, n) - Vgemm(l, m, n)), maxi);

    if (maxi > 1e-14)
        return false;

    pol.clear();
    co.clear();
    Vgemm.clear();
    Vref.clear();
    return true;
}

template <typename T> void integrate_core_naive(const int *length_,
                                                const mdarray<T, 3, CblasRowMajor> &pol_,
                                                const mdarray<T, 3, CblasRowMajor> &Vtmp,
                                                mdarray<T, 3, CblasRowMajor> &co)
{
    for (int gamma = 0; gamma < co.size(0); gamma++) {
        for (int beta = 0; beta < co.size(1); beta++) {
            for (int alpha = 0; alpha < co.size(2); alpha++) {
                T res = 0.0;
                for (int z = 0; z < length_[0]; z++) {
                    for (int y = 0; y < length_[1]; y++) {
                        const T c1 = pol_(0, gamma, z) * pol_(1, beta, y);
                        const T*__restrict vtmp = Vtmp.template at<CPU>(z, y, 0);
                        for (int x = 0; x < length_[2]; x++) {
                            res += c1 * pol_(2, alpha, x) * vtmp[x];
                        }
                    }
                }
                co(gamma, beta, alpha) = res;
            }
        }
    }
}

template <typename T> bool test_integrate_core(const int i, const int j, const int k, const int lmax)
{
    mdarray<T, 3, CblasRowMajor> pol = mdarray<T, 3, CblasRowMajor>(3,
                                                                    lmax,
                                                                    std::max(std::max(i, j), k));
    mdarray<T, 3, CblasRowMajor> co_ref = mdarray<T, 3, CblasRowMajor>(lmax, lmax, lmax);
    mdarray<T, 3, CblasRowMajor> co_gemm = mdarray<T, 3, CblasRowMajor>(lmax, lmax, lmax);
    mdarray<T, 3, CblasRowMajor> V = mdarray<T, 3, CblasRowMajor>(i, j, k);
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1.0, 1.0);
    int length[3] = {i, j, k};
    for (int s = 0; s < pol.size(); s++)
        pol[s] = distribution(generator);

    for (int s = 0; s < V.size(); s++)
        V[s] = distribution(generator);
    co_gemm.zero();
    integrate_core<T>(length,
                      pol,
                      V,
                      co_gemm);
    co_ref.zero();

    integrate_core_naive<T>(length,
                            pol,
                            V,
                            co_ref);

    T maxi = -2.0;
    for (int l = 0; l < co_gemm.size(0); l++)
        for (int m = 0; m < co_gemm.size(1); m++)
            for (int n = 0; n < co_gemm.size(2); n++) {
                maxi = std::max(std::abs(co_gemm(l, m, n) - co_ref(l, m, n)), maxi);
            }

    if (maxi > 1e-13)
        return false;

    return true;
}
