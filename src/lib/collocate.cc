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
#include "utils.hpp"


rt_graph::Timer timer;


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

template <typename T> void compute_folded_polynomial_naive(const int center,
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

template <typename T> void compute_folded_polynomials(const int center,
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

template <typename T> void compute_folded_polynomials(const int center,
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

template <typename T> void compute_folded_polynomials_all_powers(const int center[3], // center of the gaussian
																																 const int period[3], // period of the grid.
																																 const int length[3], // number of points in each direction where the gaussian is evaluated.
																																 const mdarray<T, 3, CblasRowMajor> &p_alpha,
																																 const mdarray<T, 3, CblasRowMajor> &p_beta,
																																 mdarray<T, 4, CblasRowMajor> &p_alpha_beta_folded)
{
		std::vector<T> tmp(p_alpha.size(2), 0.0);
		p_alpha_beta_folded.zero();
		for (int deg = 0; deg < 3; deg++) {
				for (int a1 = 0; a1 < p_alpha.size(1) - 1; a1++) {
						const T *__restrict p_a = p_alpha.template at<CPU>(deg, a1, 0);
						for (int a2 = 0; a2 < p_beta.size(1) - 1; a2++) {
								T *__restrict p_ab = &tmp[0];
								const T *__restrict p_b = p_beta.template at<CPU>(deg, a2, 0);
								for (int s = 0; s < p_alpha.size(2); s++)
										tmp[s] = p_a[s] * p_b[s];

								compute_folded_polynomials<T>(center[deg],
																							period[deg],
																							p_alpha.size(2),
																							&tmp[0],
																							p_alpha_beta_folded.template at<CPU>(deg, a1, a2, 0));
						}
				}
		}

		tmp.clear();
}

template <typename T> void compute_folded_polynomials_all_powers(const int center[3],
																																 const int period[3],
																																 const int length[3],
																																 const mdarray<T, 3, CblasRowMajor> &p_alpha,
																																 mdarray<T, 3, CblasRowMajor> &p_alpha_beta_folded)
{

		/*
		 * the second dimensions of p_alpha_beta_folded is p_alpha.size(1) - 1,
		 * because I store the (x - x12) in it without the exponential prefactor. it
		 * is only needed when we compute the non-orthorombic case.
		 */
		p_alpha_beta_folded.zero();
		for (int deg = 0; deg < 3; deg++) {
				for (int a1 = 0; a1 < p_alpha_beta_folded.size(1); a1++) {
						const T *__restrict p_a = p_alpha.template at<CPU>(deg, a1, 0);
						compute_folded_polynomials<T>(center[deg],
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

		const int n[3][2] = {{0, 1},
												 {0, 2},
												 {1, 2}};

		/* beta gamma */
		const T c[3] = {
				-2.0 * (basis_c(1, 0) * basis_c(2, 0) + basis_c(1, 1) * basis_c(2, 1) + basis_c(1, 2) * basis_c(2, 2)),
									/* alpha gamma */
				-2.0 * (basis_c(0, 0) * basis_c(2, 0) + basis_c(0, 1) * basis_c(2, 1) + basis_c(0, 2) * basis_c(2, 2)),
				/* alpha beta */
				-2.0 * (basis_c(0, 0) * basis_c(1, 0) + basis_c(0, 1) * basis_c(1, 1) + basis_c(0, 2) * basis_c(1, 2))
		};

		tmp.zero();

		for (int dir = 0; dir < 3; dir++) {
				int d1 = n[dir][0];
				int d2 = n[dir][1];

				const T coef = c[dir];

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

template <typename T> void compute_exponential_corrections(const int d, /* plane choice */
																													 const int *lower_corner,
																													 const int *upper_corner,
																													 const mdarray<T, 2, CblasRowMajor> &xi,
																													 const mdarray<T, 2, CblasRowMajor> &basis, /* displacement vectors of the lattice */
																													 const T eta_12, /* gaussian prefactor */
																													 mdarray<T, 3, CblasRowMajor> &Exp)
{
		const int plane[3][2] = {{0, 1},  /* z, y plane */
														 {0, 2},  /* z, x plane */
														 {1, 2}}; /* y, x plane */


		// x,y
		// 2 v11 v21 + 2 v12 v22 + 2 v13 v23
		// x,z
		// 2 v11 v31 + 2 v12 v32 + 2 v13 v33
		// y z
		// 2 v21 v31 + 2 v22 v32 + 2 v23 v33

		const double coef[3] = {
				-2.0 * (basis(2, 0) * basis(1, 0) + basis(1, 1) * basis(2, 1) + basis(1, 2) * basis(2, 2)), // y, z
				-2.0 * (basis(0, 0) * basis(2, 0) + basis(0, 1) * basis(2, 1) + basis(0, 2) * basis(2, 2)), // x, z
				-2.0 * (basis(0, 0) * basis(1, 0) + basis(0, 1) * basis(1, 1) + basis(0, 2) * basis(1, 2))
		}; //x, y

		for(int x1 = 0; x1 < upper_corner[plane[d][0]] - lower_corner[plane[d][0]]; x1++) {
				for(int y1 = 0; y1 < upper_corner[plane[d][1]] - lower_corner[plane[d][1]]; y1++) {
						const double c1 = xi(plane[d][0], x1);
						const double c2 = xi(plane[d][1], y1);
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

template<typename T> void collocate_core(const int length_[3],
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


template <typename T> void integrate(const mdarray<T, 3, CblasRowMajor> &p_alpha_beta_folded,
																		 const mdarray<int, 2, CblasRowMajor> &yz_non_vanishing_elements,
																		 const int *grid_size, /* size of the grid, fix also the period */
																		 const int grid_lower_corner[3],
																		 const mdarray<T, 3, CblasRowMajor> &V, // potential
																		 mdarray<T, 3, CblasRowMajor> &co) // Matrix elements
{
		timer.start("alloc");
		mdarray<T, 3, CblasRowMajor> p_alpha_beta_reduced_;
		const int *__restrict zp_12_i = yz_non_vanishing_elements.template at<CPU>(0, 0);
		const int *__restrict yp_12_i = yz_non_vanishing_elements.template at<CPU>(1, 0);
		const int *__restrict xp_12_i = yz_non_vanishing_elements.template at<CPU>(2, 0);

		timer.stop("alloc");
		int n[3];
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

		mdarray<T, 3, CblasRowMajor> Vreduced(n[0], n[1], n[2]);
		timer.start("Extract_potential_data");
		pack_elems(grid_size, grid_lower_corner, yz_non_vanishing_elements, V, Vreduced);
		timer.stop("Extract_potential_data");

		integrate_core(n, p_alpha_beta_reduced_, Vreduced, co);
		Vreduced.clear();
}

template <typename T> void collocate(const mdarray<T, 3, CblasRowMajor> &co,
																		 const mdarray<T, 3, CblasRowMajor> &p_alpha_beta_folded,
																		 const mdarray<int, 2, CblasRowMajor> &yz_non_vanishing_elements,
																		 const int grid_size[3],
																		 const int grid_lower_corner[3],
																		 mdarray<T, 3, CblasRowMajor> &Vtmp,
																		 mdarray<T, 3, CblasRowMajor> &density)
{
		timer.start("alloc");
		mdarray<T, 3, CblasRowMajor> p_alpha_beta_reduced_;
		const int *__restrict zp_12_i = yz_non_vanishing_elements.template at<CPU>(0, 0);
		const int *__restrict yp_12_i = yz_non_vanishing_elements.template at<CPU>(1, 0);
		const int *__restrict xp_12_i = yz_non_vanishing_elements.template at<CPU>(2, 0);

		timer.stop("alloc");
		int n[3];
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
		unpack_elems_and_add<T>(grid_lower_corner,
														yz_non_vanishing_elements,
														Vtmp,
														density);
		timer.stop("sum_up");
		p_alpha_beta_reduced_.clear();
}

template <typename T> void collocate_non_orthorombic(const mdarray<T, 3, CblasRowMajor> &polynomials,
																										 const mdarray<T, 3, CblasRowMajor> &co,
																										 const int *lower_corner,
																										 const int *upper_corner,
																										 const int *polynomial_length,
																										 const int *gaussian_center, // compared to the (0, 0, 0) of the global grid.
																										 const int *period,
																										 const mdarray<T, 2, CblasRowMajor> &basis,
																										 const T eta_12,
																										 mdarray<T, 3, CblasRowMajor> &density)
{
		int dir_min[3];
		int dir_max[3];

		mdarray<T, 3, CblasRowMajor>	polynomials_tmp(3, polynomials.size(1) - 1, std::max(std::max(density.size(0), density.size(1)), density.size(2)));
		mdarray<T, 2, CblasRowMajor>	xi(3, std::max(std::max(density.size(0), density.size(1)), density.size(2)));
		mdarray<T, 3, CblasRowMajor>	Exp(3,
																			std::max(std::max(density.size(0), density.size(1)), density.size(2)),
																			std::max(std::max(density.size(0), density.size(1)), density.size(2)));
		mdarray<T, 3, CblasRowMajor>	Vtmp(density.size(0), density.size(1), density.size(2));
		const int n[3] = {density.size(0), density.size(1), density.size(2)};
		const int poly_min[3] = {-(polynomial_length[0] - 1) / 2 + gaussian_center[0],
														 -(polynomial_length[1] - 1) / 2 + gaussian_center[1],
														 -(polynomial_length[2] - 1) / 2 + gaussian_center[2]};
		const int poly_max[3] = {(polynomial_length[0] - 1) / 2 + gaussian_center[0],
														 (polynomial_length[1] - 1) / 2 + gaussian_center[1],
														 (polynomial_length[2] - 1) / 2 + gaussian_center[2]};

		for(int dir = 0; dir < 3; dir++) {
				dir_min[dir] = poly_min[dir] / period[dir] - 1;
				dir_max[dir] = poly_max[dir] + 1;
		}
		for (int z = dir_min[0]; z <= dir_max[0]; z++) {
				polynomials_tmp.zero();
				xi.zero();
				if (!copy_poly_in_tmp<T>(0,
																 z,
																 period[0],
																 lower_corner[0],
																 upper_corner[0],
																 polynomials,
																 poly_min[0],
																 poly_max[0],
																 polynomials_tmp,
																 xi))
						continue;

				for (int y = dir_min[1]; y <= dir_max[1]; y++) {
						if (!copy_poly_in_tmp<T>(1,
																		 y,
																		 period[1],
																		 lower_corner[1],
																		 upper_corner[1],
																		 polynomials,
																		 poly_min[1],
																		 poly_max[1],
																		 polynomials_tmp,
																		 xi))
								continue;


						bool compute = true;
						for (int x = dir_min[0]; x <= dir_max[2]; x++) {
								if (!copy_poly_in_tmp<T>(2,
																				 x,
																				 period[2],
																				 lower_corner[2],
																				 upper_corner[2],
																				 polynomials,
																				 poly_min[2],
																				 poly_max[2],
																				 polynomials_tmp,
																				 xi))
										continue;

								memset(xi.template at<CPU>(2, 0), 0, sizeof(T) * xi.size(1));

								if (compute) {
										/* compute the exponential correction */
										timer.start("exponentials_generic");
										compute_exponential_corrections<T>(0,
																											 lower_corner,
																											 upper_corner,
																											 xi,
																											 basis,
																											 eta_12,
																											 Exp);
										timer.stop("exponentials_generic");
										compute = false;
								}
								Vtmp.zero();
								/*  do the matrix-matrix multiplication */
								collocate_core(n, co, polynomials_tmp, Vtmp);
								/* can do that in parallel with collocate_core */
								/* compute the exponential correction */
								timer.start("exponentials_generic");
								compute_exponential_corrections<T>(1,
																									 lower_corner,
																									 upper_corner,
																									 xi,
																									 basis,
																									 eta_12,
																									 Exp);
								timer.stop("exponentials_generic");
								timer.start("exponentials_generic");
								compute_exponential_corrections<T>(2,
																									 lower_corner,
																									 upper_corner,
																									 xi,
																									 basis,
																									 eta_12,
																									 Exp);
								timer.stop("exponentials_generic");

								/*  need to multiply by Exp_ij Exp_jk Exp_ki */
								for (int z1 = 0; z1 < density.size(0); z1++) {
										for (int y1 = 0; y1 < density.size(1); y1++) {
												const T tmp = Exp(0, z1, y1);
												const T *__restrict src1 = Exp.template at<CPU>(1, z1, 0);
												const T *__restrict src2 = Exp.template at<CPU>(2, y1, 0);
												const T *__restrict src3 = Vtmp.template at<CPU>(z1, y1, 0);
												T*__restrict dst = density.template at<CPU>(z1, y1, 0);
												for (int x1 = 0; x1 < density.size(2); x1++) {
														//    dst[x1] += src3[x1] * src2[x1] * src1[x1] * tmp;
														dst[x1] += src3[x1];
												}
										}
								}
								printf("%.10lf\n", density(10, 10, 10));
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
				int xmax = 0;
				find_interval(non_zeroes_, interval_, xmax);

				for (int z = 0; z < xmax; z ++)
						printf("%*d %*d\n", 3, interval_[2 * z], 3, interval_[2 * z + 1]);

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
				compute_folded_polynomials<int>(center,
																				unfolded_table,
																				folded_table);

				printf("%*d%*d%*d | test ", 3, center, 3, period, 3, length);
				for (int i = 0; i < folded_table.size(); i++)
						std::cout << folded_table[i];
				std::cout << "\n";

				memset(&non_zeroes_test[0], 0, sizeof(int) * folded_table.size());

				compute_folded_polynomial_naive<int>(center,
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

// void grid_collocate_internal(const bool use_ortho,
//														 const int func,
//														 const int la_max,
//														 const int la_min,
//														 const int lb_max,
//														 const int lb_min,
//														 const double zeta,
//														 const double zetb,
//														 const double rscale,
//														 const double *dh,
//														 const double *dh_inv,
//														 const double *ra,
//														 const double *rab,
//														 const int *npts,
//														 const int *ngrid,
//														 const int *lb_grid,
//														 const bool *periodic,
//														 const double radius,
//														 const int o1,
//														 const int o2,
//														 const int n1,
//														 const int n2,
//														 const double *pab,
//														 double *grid)
// {
//		/* better naming convention */
//		/* offset for extracting the gaussian weights */
//		const int coef_offset[2] = {o1, o2};

//		/* gaussian weights for the density matrix */
//		const mdarray<double, 2, CblasRowMajor> pab_bis(pab, n2, n1);

//		/* well, can be the density operator, or anything really. I just call that
//		 * density but it contains the result of the collocation */
//		mdarray<double, 3, CblasRowMajor> density(grid, ngrid[2], ngrid[1], ngrid[0]);

//		/* contains the subblock of the gaussian weights that are needed for the
//		 * collocation */
//		mdarray<double, 2, CblasRowMajor> coefs = mdarray<double, 2, CblasRowMajor>(return_offset_l(lb_max + 1), return_offset_l(la_max + 1));

//		/* contains the local part of the polynomials after applying periodic
//		 * boundaries conditions */
//		darray<double, 3, CblasRowMajor> p_alpha_beta_folded;

//		mdarray<int, 2, CblasRowMajor> yz_non_vanishing_elements = mdarray<int, 2, CblasRowMajor>(3,
//																																															std::max(std::max(density.size(0),
//																																																								density.size(1)),
//																																																			 density.size(2)));

//		/* Full polynomials computed independently of the grid size */
//		mdarray<double, 3, CblasRowMajor> p_alpha;

// /* basis displacement vectors */
//		mdarray<double, 2, CblasRowMajor> basis(dh_inv, 3, 3);

//		/*
//		 * depending on the operation we want to do, we may have to compute
//		 * polynomials of higher degree
//		 */

//		int lmin_diff[2], lmax_diff[2];
//		grid_prepare_get_ldiffs(func,	lmin_diff, lmax_diff);

//		/* we have the windows for which we want to compute the polynomials */
//		const int lmin[2] = {max(la_min + lmin_diff[0], 0), max(lb_min + lmin_diff[1], 0)};
//		const int lmax[2] = {la_max + lmax_diff[0], lb_max + lmax_diff[1]};

//		/* Compute the position of the second atom starting from the position of the
//		 * first atom and the center of the gaussian obtained from the product of
//		 * two gaussians centered around the two atoms. */

//		const double zetp = zeta + zetb;
//		const double f = zetb / zetp;
//		const double rab2 = rab[0] * rab[0] + rab[1] * rab[1] + rab[2] * rab[2];
//		const double prefactor = rscale * exp(-zeta * f * rab2);
//		double rp[3], rb[3];
//		for (int i=0; i<3; i++) {
//				rp[i] = ra[i] + f * rab[i];
//				rb[i] = ra[i] + rab[i];
//		}

//		/*
//			Extracted weights for the collocate method. It is a tensor of rank 3 of
//			the form co(alpha1, alpha2, beta1, beta2, gamma1, gamma2) where the alpha,
//			beta, gamma indices are put together
//		 */

//		mdarray<double, 2, CblasRowMajor> coefs = mdarray<double, 2, CblasRowMajor>(return_offset_l(la_max + 1), return_offset_l(lb_max + 1));
//		{
//				const int llmin[2] = {la_min, lb_min};
//				cosnt int llmax[2] = {la_max, lb_max};
//				extract_polynomial_coefficients(coef,
//																				coef_offset_,
//																				llmin[2], llmax[2],
//																				coefs);
//		}


//		/* contains the transformed weights to do the collocate */
//		darray<double, 3, CblasRowMajor> co = mdarray<double, 3, CblasRowMajor>(lmax[0] + lmax[1],
//																																						lmax[0] + lmax[1],
//																																						lmax[0] + lmax[1]);

//		mdarray<double, 3, CblasRowMajor> Vtmp = mdarray<double, 3, CblasRowMajor>(density.size(0),
//																																							 density.size(1),
//																																							 density.size(2));

//		/* rescale the coefficients */
//		for (int i = 0; i < coefs.size(0); i++)
//				for (int j = 0; j < coefs.size(1); j++)
//						coefs(i, j) *= prefactor;

//		// compute (x - x12)^alpha exp(-...) for alpha = [0...l1 + l2]

//		calculate_polynomials(lmax[0] + lmax[1],
//													radius,
//													dr,
//													r_ab,
//													mu_mean,
//													p_alpha,
//													length_p);

//		co.zero();

//		// compute the coefficients for the collocate method depending on the type
//		// of calculation

//		compute_compact_polynomial_coefficients<double>(func, coefs, coef_offset, lmin, lmax, ra, rb, r_ab, co);

//		p_alpha_beta_folded.zero();

//		int length_p[3];
//		compute_folded_polynomials_all_powers<double>(center, // position of the gaussian center in grid coordinates
//																									period, // period of the grid.
//																									length_p, //
//																									p_alpha,
//																									p_alpha_beta_folded);


//		compute_nonzero_values(center[0], period[0], length_p[0], lower_corner[0], Potential.size(0), 0, yz_non_vanishing_elements);
//		compute_nonzero_values(center[1], period[1], length_p[1], lower_corner[1], Potential.size(1), 1, yz_non_vanishing_elements);
//		compute_nonzero_values(center[2], period[2], length_p[2], lower_corner[2], Potential.size(2), 2, yz_non_vanishing_elements);

//		collocate<double>(co,
//											p_alpha_beta_folded,
//											yz_non_vanishing_elements,
//											period,
//											lower_corner,
//											Vtmp,
//											density);
// }


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
		int lower_corner[3] = {0, 0, 0};
		int upper_corner[3] = {20, 20, 20};
		int center[3];
		int lmax[2] = {0, 0};
		int lmin[2] = {0, 0};
		double mu_ab, mu_mean, r_ab[3], kappa_ab;

		double radius = 5;

		mdarray<double, 2, CblasRowMajor> coefs;
		mdarray<double, 3, CblasRowMajor> Potential, Potential1;
		mdarray<double, 3, CblasRowMajor> p_alpha_beta_folded;
		mdarray<double, 3, CblasRowMajor> p_alpha;
		mdarray<double, 2, CblasRowMajor> basis(3, 3);
		const int coef_offsets_[2] = {0, 0};
		const int grid_center[3] = {10, 10, 10};

		center[0] = (r_ab[0] + 0.00000000001) / dr[0] + grid_center[0];
		center[1] = (r_ab[1] + 0.00000000001) / dr[1] + grid_center[1];
		center[2] = (r_ab[2] + 0.00000000001) / dr[2] + grid_center[2];
		mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);

		// just setup an arbitrary grid
		Potential = mdarray<double, 3, CblasRowMajor>(period[0], period[1], period[2]);
		Potential.zero();
		Potential1 = mdarray<double, 3, CblasRowMajor>(period[0], period[1], period[2]);
		Potential1.zero();

		coefs = mdarray<double, 2, CblasRowMajor>(return_offset_l(lmax[0] + 1), return_offset_l(lmax[1] + 1));

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

		basis.zero();
		basis(0, 0) = 1.0;
		basis(1, 1) = 1.0;
		basis(2, 2) = 1.0;
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

		if (test_integrate_core<double>(14, 23, 17, 1))
				std::cout << "Integrate core : Test 2 passed\n";

		if (test_integrate_core<double>(1, 12, 29, 5))
				std::cout << "Integrate core : Test 3 passed\n";

		if (test_integrate_core<double>(13, 45, 29, 7))
				std::cout << "Integrate core : Test 4 passed\n";
		timer.stop("test_integrate_core");

		timer.stop("tests");

		timer.start("total");
		for (int s = 0; s < 10000; s++) {
				if ((s % 1000) == 0)
						printf("%d\n", s);

				timer.start("one_set_of_gaussian");;

				calculate_center_of_two_gaussians(ra,
																					rb,
																					alpha_a,
																					alpha_b,
																					r_ab,
																					&kappa_ab,
																					&mu_ab,
																					&mu_mean);

				for (int i = 0; i < coefs.size(0); i++)
						for (int j = 0; j < coefs.size(1); j++)
								coefs(i, j) = kappa_ab;


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
//				timer.start("compute_compact_polynomial_coefficients");
				// apply the transformation (x-x1)(x-x2) -> sum bla bla (x-x12)
				compute_compact_polynomial_coefficients<double>(coefs, coef_offsets_, lmin, lmax, ra, rb, r_ab, co);
//				timer.stop("compute_compact_polynomial_coefficients");

				p_alpha_beta_folded.zero();

//				timer.start("polyall");
				compute_folded_polynomials_all_powers<double>(center, // position of the gaussian center in grid coordinates
																											period, // period of the grid.
																											length_p, //
																											p_alpha,
																											p_alpha_beta_folded);
//				timer.stop("polyall");

				compute_nonzero_values(center[0], period[0], length_p[0], lower_corner[0], Potential.size(0), 0, yz_non_vanishing_elements);
				compute_nonzero_values(center[1], period[1], length_p[1], lower_corner[1], Potential.size(1), 1, yz_non_vanishing_elements);
				compute_nonzero_values(center[2], period[2], length_p[2], lower_corner[2], Potential.size(2), 2, yz_non_vanishing_elements);

//				timer.start("collocate");
				collocate<double>(co,
													p_alpha_beta_folded,
													yz_non_vanishing_elements,
													period,
													lower_corner,
													Vtmp,
													Potential);

//				timer.stop("collocate");

				printf("%5lf\n", co(0, 0, 0));
				// timer.start("collocate_non_orthorombic");
				// collocate_non_orthorombic<double>(p_alpha,
				//                                   co,
				//                                   lower_corner,
				//                                   upper_corner,
				//                                   length_p,
				//                                   center, // compared to the (0, 0, 0) of the global grid.
				//                                   period,
				//                                   basis,
				//                                   mu_mean,
				//                                   Potential1);
				// timer.stop("collocate_non_orthorombic");
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
		f = fopen("test1.dat", "w+");
		fwrite(Potential1.at<CPU>(), sizeof(double), Potential1.size(), f);
		fclose(f);
		p_alpha.clear();
		Potential.clear();
		Potential1.clear();
		coefs.clear();
		p_alpha_beta_folded.clear();
		return 0;
}
