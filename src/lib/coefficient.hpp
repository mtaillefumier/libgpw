#ifndef COEFFICIENTS_HPP
#define COEFFICIENTS_HPP

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

inline int return_linear_index_from_exponents(const int alpha, const int beta, const int gamma)
{
		const int l = alpha + beta + gamma;
		return (l - alpha) * (l - alpha - 1) / 2 + gamma;
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
																		co(a11 * (lmax[1] + 2) + a22,
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

#endif
