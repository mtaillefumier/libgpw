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
#include "coefficient.hpp"
#include "polynomials.hpp"

rt_graph::Timer timer;

int popcount(unsigned x)
{
		int c = 0;
		for (; x != 0; x >>= 1)
				if (x & 1)
						c++;
		return c;
}

inline void compute_indice(const int n, int &alpha, int &beta, int &gamma)
{
		gamma = (n & 0xFF0000) >> 16;
		beta = (n & 0xFF00) >> 8;
		alpha = (n & 0xFF);
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
		if (length >= period) {
				for (int  i = 0; i < (int)non_zeroes_.size(1); i++) {
						non_zeroes_(dir, i) = 1;
				}
				return;
		}

		for (int  i = 0; i < non_zeroes_.size(1); i++) {
				non_zeroes_(dir, i) = 0;
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
														const int dir, // direction x, y, z
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

/*
 * Naive implementation of applying periodic boundaries conditions on a table of
 * length > period
 */

template <typename T> void compute_folded_integrals_naive(const int center,
																													const std::vector<T> &unfolded_table,
																													std::vector<T> &folded_table)
{
		const int period = folded_table.size();
		const int length = unfolded_table.size();
		for (int i = 0; i < unfolded_table.size(); i++)
				folded_table[(center - (length + 1) / 2 + i + 16 * period) % period] += unfolded_table[i];
}


// the initial table contains a gaussian. But periodic boundaries conditions
// screw up thing a bit and we have to sum up multiple copies of the gaussian
// (the center been shifted by a period). this boils down to split the table in
// intervals of length corresponding to the period of the grid and then add them
// together elementwise

// basically we do this F(x) = \sum_{n} G_a(n L + x), where G_a is a "gaussian"
// centered around x_a and x is the grid coordinates.
//
// *Warning* : this can only be used when the grid orthorombic.

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

/* Compute the gaussian parameters resulting from the product of two
 * gaussians.
 */

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


/*  Apply periodic boundaries conditions on a product of two functions
 *  f(a)g(a), in our case two polynomials (x-x1)(x-x2).
 */

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
		 * for non orthorombic case n_ijk = n_ijk ^ortho T^z_ij T^y_ik T^x_jk
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

template <typename T> void compute_two_gaussian_coefficients(const mdarray<T, 3, CblasRowMajor> &coef,
																														 const int lmin[2], const int lmax[2],
																														 T *rab,
																														 mdarray<T, 3, CblasRowMajor> &co)
{
		timer.start("compute_two_gaussian_coefficients_init");
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
		timer.stop("compute_two_gaussian_coefficients_init");

		for (int l1 = lmin[0]; l1 <= lmax[0]; l1++) {
				for (int l2 = lmin[1]; l2 <= lmax[2]; l2++) {
						for (int alpha1 = 0; alpha1 <= l1; alpha1++) {
								for (int alpha2 = 0; alpha2 <= l2; alpha2++) {
										for (int beta1 = 0; beta1 <= (l1 - alpha1); beta1++) {
												const int gamma1 = l1 - alpha1 - beta1;
												for (int beta2 = 0; beta2 <= (l2 - alpha2); beta2++) {
														const int gamma2 = l2 - alpha2 - beta2;
														T tmp = 0.0;
														for (int k1 = 0; k1 <= alpha1; k1++) {
																const T c1 = coef(k1 + k2, k3 + k4, k5 + k6) *
																		power(2, 0, alpha1, k1);
																for (int k2 = 0; k2 <= alpha2; k2++) {
																		const T c2 = c1 * power(2, 1, alpha2, k2);
																		for (int k3 = 0; k3 <= beta1; k3++) {
																				const T c3 = c2 * power(1, 0, beta1, k3);
																				for (int k4 = 0; k4 <= beta2; k4++) {
																						const T c4 = c3 * power(1, 1, beta2, k4);
																						for (int k5 = 0; k5 <= gamma1; k5++) {
																								cosnt T c5 = c4 * power(0, 0, gamma1, k5);
																								for (int k6 = 0; k6 <= gamma2; k6++) {
																										tmp += c5 * power(0, 1, gamma2, k6);
																								}
																						}
																				}
																		}
																}
														}

														co(offset_[l1] + , offset_[l2] + ) = tmp;
												}
										}
								}
						}
				}
		}
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

template <typename T> void compute_exponential_corrections(const int x1, /* coordinate of the brillouin zone in the first direction of the plane */
																													 const int x2, /* coordinate of the brillouin zone in the second direction of the plane */
																													 const int d, /* plane choice */
																													 const int *period, /* period of the lattice in all three directions */
																													 const int *lower_corner, /* coordinates of the left point of the local part of the grid */
																													 const int *upper_corner, /* coordinates of the right point of the local part of the grid */
																													 const T *__restrict basis, /* displacement vectors of the lattice */
																													 const int *grid_center, /* indices of the origin (0,0,0) of the grid */
																													 T *__restrict dr, /* lattice spacing in the three directions */
																													 const T*__restrict *r_12, /* Coordinates of the gaussian center  */
																													 const T eta_12, /* gaussian prefactor */
																													 mdarray<T, 3, CblasRowMajor> &Exp)
{
		const int plane[3][2] = {{0, 1}, /* z, y plane */
														 {0, 2},  /* z, x plane */
														 {1, 2}}; /* y, x plane */

		/* r_12 should be in the natural basis of the lattice */
		const double x = x1 * period[plane[d][0]] * dr[plane[d][0]] - r_12[plane[d][0]];

		const double y = x2 * period[plane[d][1]] * dr[plane[d][1]] - r_12[plane[d][1]];

		// x,y
		// 2 v11 v21 + 2 v12 v22 + 2 v13 v23
		// x,z
		// 2 v11 v31 + 2 v12 v32 + 2 v13 v33
		// y z
		// 2 v21 v31 + 2 v22 v32 + 2 v23 v33

		const double coef[3] = {
				-2.0 * (basis[3] * basis[6] + basis[4] * basis[7] + basis[5] * basis[8]), // y, z
				-2.0 * (basis[0] * basis[6] + basis[1] * basis[7] + basis[2] * basis[8]), // x, z
				-2.0 * (basis[0] * basis[3] + basis[1] * basis[4] + basis[2] * basis[5])
		}; //x, y

		for(int x1 = lower_corner[plane[d][0]]; x1 < upper_corner[plane[d][0]]; x1++) {
				for(int y1 = lower_corner[plane[d][1]]; y1 < upper_corner[plane[d][1]]; y1++) {
						const double c1 = x1 * dr[plane[d][1]] + x;
						const double c2 = x2 * dr[plane[d][2]] + y;
						Exp(d, x1, y1) = exp(c1 * c2 * coef[d] * eta_12);
				}
		}
}


/* Note : collocate and integrate do not care about the zero elements. They just
 * do the matrix-matrix multiplications needed for the work to be done, nothing
 * else. It is possible to pack only the datas that are relevant for the
 * computation, but this should be done externally (and with common sense). It
 * can save a lot of time if the local size of the grid is "large" compared to
 * the typical extension of the gaussians but completely irrelevant if the
 * gaussians have extension of the order of the period or more. In that case
 * time is waisted for packing and unpacking (result) data
 */

template <typename T> void integrate_core(const int *length,
																					const mdarray<T, 3, CblasRowMajor> &polynomials,
																					const mdarray<T, 3, CblasRowMajor> &V,
																					mdarray<T, 3, CblasRowMajor> &Velems)
{
		/* the first operation is a matrix-matrix operation. we do

			 V_{ijk} P_i^\alpha = T ^_{jk}^alpha.

			 we need to transpose the second matrix

			 The code in mathematica is

			 integrate[Pot_List, Poly_List] := Module[{T1, T2, dims, deg},
						 dims = Map[Dimensions[#] &, Poly][[All, 2]];
						 Print[dims];
						 deg = Dimensions[Poly[[1]], 1][[1]];
						 Print[deg];
						 T1 = ArrayReshape[ArrayFlatten[V], {dims[[1]] dims[[2]], dims[[3]]}];
						 T2 = T1.Transpose[Poly[[3]]];
						 T1 = ArrayReshape[Flatten[T2], {dims[[1]], dims[[2]], deg}];
						 Print[T1 // Dimensions];
						 T2 = Map[Transpose[#].Transpose[Poly[[2]]] &, T1];
						 T1 = ArrayReshape[Flatten[T2], {dims[[1]], deg deg}];
						 Return[ArrayReshape[Transpose[T1].Transpose[Poly[[1]]] // Flatten, {deg, deg, deg}]];
			 ];

			 integratep[V, Poly] is equivalent to

			 Table[Sum[V[[k, j, i]] Poly[[1]][[\[Gamma], k]] Poly[[2]][[\[Beta], j]] Poly[[3]][[\[Alpha], i]],
								 {i, 1, 20}, {j, 1, 18}, {k, 14}],
								 {\[Alpha], 1, 6},
								 {\[Beta], 1, 6},
								 {\[Gamma], 1, 6}]

			 note that the indices alpha, beta, gamma are *in* this order so in fortran it will be gamma, beta, alpha

			 the c code is very simple
		*/

		mdarray<T, 3, CblasRowMajor> tmp = mdarray<T, 3, CblasRowMajor>(length[0], length[1], polynomials.size(1));
		mdarray<T, 3, CblasRowMajor> tmp1 = mdarray<T, 3, CblasRowMajor>(length[0], polynomials.size(1), polynomials.size(1));

		/* T1.Transpose[Poly[[3]]] */

		cblas_dgemm(CblasRowMajor,
								CblasNoTrans,
								CblasTrans,
								length[0] * length[1], /* (k j) */
								polynomials.size(1), /* alpha */
								length[2], /* i */
								1.0,
								V.template at<CPU>(0, 0, 0),
								V.ld(),
								polynomials.template at<CPU>(2, 0, 0), /* P_x (i) */
								polynomials.ld(),
								0.0,
								tmp.template at<CPU>(0, 0, 0), /* (k j), alpha */
								tmp.ld());

		/*
			Then the mapping

			Map[Transpose[#].Transpose[Poly[[2]]] &, T1];

		 */

		for (int k = 0; k < V.size(0); k++) {
				cblas_dgemm(CblasRowMajor,
										CblasNoTrans,
										CblasNoTrans,
										polynomials.size(1), // P[beta][j] -> beta -> row(A)
										polynomials.size(1), // T[k][j][alpha] -> alpha -> col(B)
										length[1], // sum over j = col A = row B
										1.0,
										polynomials.template at<CPU>(1, 0, 0), /* (beta, j) */
										polynomials.ld(),
										tmp.template at<CPU>(k, 0, 0), /* (k, j, alpha) k is fixed */
										tmp.ld(), // alpha
										0.0,
										tmp1.template at<CPU>(k, 0, 0), /* (k, beta, alpha) */
										tmp1.ld());
		}

		/* Finally
			 Transpose[T1].Transpose[Poly[[1]]

		*/

		cblas_dgemm(CblasRowMajor,
								CblasNoTrans,
								CblasNoTrans,
								polynomials.size(1), // gamma
								polynomials.size(1) * polynomials.size(1), // (beta, alpha)
								V.size(0), // k
								1.0,
								polynomials.template at<CPU>(0, 0, 0), // P_z(gamma, k)
								polynomials.ld(),
								tmp1.template at<CPU>(0, 0, 0), // T(k, beta, alpha)
								polynomials.size(1) * polynomials.size(1),
								0.0,
								Velems.template at<CPU>(0, 0, 0), // (gamma, beta, alpha)
								Velems.ld() * polynomials.size(1));
}

template<typename T> void collocate_core(const int *length_,
																				 const mdarray<T, 3, CblasRowMajor> &co,
																				 const mdarray<T, 3, CblasRowMajor> &p_alpha_beta_reduced_,
																				 mdarray<T, 3, CblasRowMajor> &Vtmp)
{
		timer.start("init");

		mdarray<T, 3, CblasRowMajor> C(co.size(0), co.size(1), length_[1]);
		mdarray<T, 3, CblasRowMajor> xyz_alpha_beta(co.size(0), length_[0], length_[1]);

		// C.zero();
		// xyz_alpha_beta.zero();
		timer.stop("init");

		if (co.size(0) > 1) {
				timer.start("dgemm");
// we can batch this easily
				for (int a1 = 0; a1 < co.size(0); a1++) {
						// we need to replace this with libxsmm
						cblas_dgemm(CblasRowMajor,
												CblasNoTrans,
												CblasNoTrans,
												co.size(2),
												length_[1],
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

				for (int a1 = 0; a1 < co.size(0); a1++) {
						cblas_dgemm(CblasRowMajor,
												CblasTrans,
												CblasNoTrans,
												length_[0],
												length_[1],
												co.size(2),
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
										length_[0] * length_[1],
										length_[2],
										co.size(2),
										1.0,
										xyz_alpha_beta.template at<CPU>(0, 0, 0),
										xyz_alpha_beta.size(1) * xyz_alpha_beta.ld(),
										p_alpha_beta_reduced_.template at<CPU>(2, 0, 0),
										p_alpha_beta_reduced_.ld(),
										0.0,
										Vtmp.template at<CPU>(0, 0, 0),
										Vtmp.ld());
				timer.stop("dgemm");
		} else {
				for (int z1 = 0; z1 < length_[0]; z1++) {
						const T tz = co(0, 0, 0) * p_alpha_beta_reduced_(0, 0, z1);
						for (int y1 = 0; y1 < length_[1]; y1++) {
								const T tmp = tz * p_alpha_beta_reduced_(1, 0, y1);
								const T *__restrict src = p_alpha_beta_reduced_.template at<CPU>(2, 0, 0);
								T *__restrict dst = Vtmp.template at<CPU>(z1, y1, 0);
								for (int x1 = 0; x1 < length_[2]; x1++) {
										dst[x1] = tmp * src[x1];
								}
						}
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


		/*
		 * we pack all non zero elements of a set of polynomial. If the polynomials
		 * have non zero values on a small portion of the grid then it is
		 * advantageous to copy the non zero data to a temporary variable. The
		 * problem is when is it worth to do it.

		 * For instance, if we have n m k / 8 non zero elements, then we save a lot
		 * during the matrix-matrix multiplication and it cost us (n + m + k) / 2
		 * elements to copy for each exponent.
		 *
		 * The problem with this solution is that we can not use batched dgemm and
		 * will perform badly on GPUs.
		 */

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
		collocate_core<double>(n, co, p_alpha_beta_reduced_, Vtmp);
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
		p_alpha_beta_reduced_.clear();
}

template <typename T> void collocate_non_orthorombic(const mdarray<T, 3, CblasRowMajor> &polynomials,
																										 const mdarray<T, 3, CblasRowMajor> &co,
																										 const int *lower_corner,
																										 const int *upper_corner,
																										 const int *polynomial_length, // number
																																									 // of
																																									 // values
																																									 // where
																																									 // the
																																									 // polynomials
																																									 // (time
																																									 // exponentials
																																									 // are
																																									 // evaluated)
																										 const int *gaussian_center, // compared to the (0, 0, 0) of the global grid.
																										 const int *period,
																										 const T*__restrict basis,
																										 const T*__restrict dr,
																										 const T eta_12,
																										 const T *__restrict r_12,
																										 mdarray<T, 3, CblasRowMajor> &V)
{
		int dir_min[3];
		int dir_max[3];

		mdarray<T, 3, CblasRowMajor>	polynomials_tmp(3, polynomials.size(1), std::max(std::max(V.size(0), V.size(1)), V.size(2)));
		mdarray<T, 3, CblasRowMajor>	Exp(3,
																			std::max(std::max(V.size(0), V.size(1)), V.size(2)),
																			std::max(std::max(V.size(0), V.size(1)), V.size(2)));
		mdarray<T, 3, CblasRowMajor>	Vtmp(V.size(0), V.size(1), V.size(2));

		for(int dir = 0; dir < 3; dir++) {
				dir_min[dir] = (-(polynomial_length[dir] - 1) / 2 + gaussian_center[dir]) / period[dir] - 1;
				dir_max[dir] = ((polynomial_length[dir] - 1) / 2 - gaussian_center[dir]) / period[dir] + 1;
		}

		for (int z = dir_min[0]; z <= dir_max[0]; z++) {
				polynomials_tmp.zero();

				if (!copy_poly_in_tmp(0,
															z,
															period[0],
															lower_corner[0],
															upper_corner[0],
															polynomials,
															-(polynomial_length[0] - 1) / 2 + gaussian_center[0],
															(polynomial_length[0] - 1) / 2 - gaussian_center[0],
															polynomials_tmp))
						continue;

				for (int y = dir_min[1]; y <= dir_max[1]; y++) {
						if (!copy_poly_in_tmp(1,
																	y,
																	period[1],
																	lower_corner[1],
																	upper_corner[1],
																	polynomials,
																	-(polynomial_length[1] - 1) / 2 + gaussian_center[1],
																	(polynomial_length[1] - 1) / 2 - gaussian_center[1],
																	polynomials_tmp))
								continue;

						/* compute the exponential correction */
						compute_exponential_corrections(z, y, 0, 1, period, lower_corner, upper_corner, basis, dr, eta_12, r_12, Exp);

						for (int x = dir_min[0]; x <= dir_max[2]; x++) {
								if (!copy_poly_in_tmp(2,
																			x,
																			period[2],
																			lower_corner[2],
																			upper_corner[2],
																			polynomials,
																			-(polynomial_length[2] - 1) / 2 + gaussian_center[2],
																			(polynomial_length[2] - 1) / 2 - gaussian_center[2],
																			polynomials_tmp))
										continue;

								/*  do the matrix-matrix multiplication */
								collocate_core(Vtmp, co, polynomials_tmp);

								/* can do that in parallel with collocate_core */
								/* compute the exponential correction */
								compute_exponential_corrections(z, x, 0, 2, period, lower_corner, upper_corner, basis, dr, eta_12, r_12, Exp.template at<CPU>(1, 0, 0));

								compute_exponential_corrections(y, x, 1, 2, period, lower_corner, upper_corner, basis, dr, eta_12, r_12, Exp.template at<CPU>(2, 0, 0));

								/*  need to multiply by Exp_ij Exp_jk Exp_ki */
								for (int z1 = 0; z1 < V.size(0); z1++) {
										for (int y1 = 0; y1 < V.size(1); y1++) {
												const T tmp = Exp(0, z1, y1);
												const T *__restrict src1 = Exp(1, z1, 0);
												const T *__restrict src2 = Exp(2, y1, 0);
												const T *__restrict src3 = Vtmp.template at<CPU>(z1, y1, 0);
												T*__restrict dst = V.template at<CPU>(z1, y1, 0);
												for (int x1 = 0; x1 < V.size(2); x1++) {
														dst[x1] += src3[x1] * src2[x1] * src1[x1] * tmp;
												}
										}
								}
						}
				}
		}
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
						printf("%*d %*d\n", 3, interval_[z], 3, interval_[z + 1]);

				printf("%*d%*d%*d | test ", 3, center, 3, period, 3, length);

				for (int i = 0; i < non_zeroes_.size(); i++)
						std::cout << non_zeroes_[i];
				std::cout << "\n";

				for (int i = 0; i < length; i++)
						non_zeroes_test[(center - (length + 1) / 2 + i + 16 * period) % period] = 1;

				printf("%*d%*d%*d | refs ", 3, center, 3, period, 3, length);
				for (int i = 0; i < non_zeroes_test.size(); i++)
						std::cout << non_zeroes_test[i];
				std::cout << "\n";
		}

		std::cout << "\n\n";

		for (int center = 0; center < 30; center++) {
				memset(&folded_table[0], 0, sizeof(int) * folded_table.size());
				compute_folded_integrals<int>(center,
																			unfolded_table,
																			folded_table);

				printf("%*d%*d%*d | test ", 3, center, 3, period, 3, length);
				for (int i = 0; i < folded_table.size(); i++)
						std::cout << folded_table[i];
				std::cout << "\n";

				memset(&non_zeroes_test[0], 0, sizeof(int) * folded_table.size());

				compute_folded_integrals_naive<int>(center,
																						unfolded_table,
																						non_zeroes_test);
				printf("%*d%*d%*d | refs ", 3, center, 3, period, 3, length);
				for (int i = 0; i < non_zeroes_test.size(); i++)
						std::cout << non_zeroes_test[i];
				std::cout << "\n\n";
		}
}

#include "collocate_test.cc"

template void integrate_core<double>(const int*,
																		 const mdarray<double, 3, CblasRowMajor>&,
																		 const mdarray<double, 3, CblasRowMajor>&,
																		 mdarray<double, 3, CblasRowMajor>&);
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


		timer.start("tests");
		timer.start("test_collocate_core");
		if (test_collocate_core<double>(14, 23, 17, 3))
				std::cout << "Test 1 passed\n";

		if (test_collocate_core<double>(14, 23, 17, 1))
				std::cout << "Test 2 passed\n";

		if (test_collocate_core<double>(1, 12, 29, 5))
				std::cout << "Test 3 passed\n";

		if (test_collocate_core<double>(13, 45, 29, 7))
				std::cout << "Test 4 passed\n";

		timer.stop("test_collocate_core");

		timer.start("test_integrate_core");


		if (test_integrate_core<double>(14, 23, 17, 3))
				std::cout << "Integrate core : Test 1 passed\n";

		if (test_integrate_core<double>(14, 23, 17, 3))
				std::cout << "Integrate core : Test 1 passed\n";

		if (test_integrate_core<double>(14, 23, 17, 3))
				std::cout << "Integrate core : Test 1 passed\n";

		if (test_integrate_core<double>(14, 23, 17, 3))
				std::cout << "Integrate core : Test 1 passed\n";
		timer.stop("test_integrate_core");

		timer.stop("tests");

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