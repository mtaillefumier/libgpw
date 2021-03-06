/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2020  CP2K developers group                         *
 *****************************************************************************/

#include <cstdio>
#include <cmath>
#include <cassert>
#include <climits>

//#include "grid_base_ref_c.h"
#include "rt_graph.hpp"
#include "mdarray.hpp"

extern rt_graph::Timer timer;


static const int ncoset[] = {1,  // l=0
														 4,  // l=1
														 10, // l=2 ...
														 20, 35, 56, 84, 120, 165, 220, 286, 364,
														 455, 560, 680, 816, 969, 1140, 1330};

static const double fac[] = {
				0.10000000000000000000E+01, 0.10000000000000000000E+01, 0.20000000000000000000E+01,
				0.60000000000000000000E+01, 0.24000000000000000000E+02, 0.12000000000000000000E+03,
				0.72000000000000000000E+03, 0.50400000000000000000E+04, 0.40320000000000000000E+05,
				0.36288000000000000000E+06, 0.36288000000000000000E+07, 0.39916800000000000000E+08,
				0.47900160000000000000E+09, 0.62270208000000000000E+10, 0.87178291200000000000E+11,
				0.13076743680000000000E+13, 0.20922789888000000000E+14, 0.35568742809600000000E+15,
				0.64023737057280000000E+16, 0.12164510040883200000E+18, 0.24329020081766400000E+19,
				0.51090942171709440000E+20, 0.11240007277776076800E+22, 0.25852016738884976640E+23,
				0.62044840173323943936E+24, 0.15511210043330985984E+26, 0.40329146112660563558E+27,
				0.10888869450418352161E+29, 0.30488834461171386050E+30, 0.88417619937397019545E+31,
				0.26525285981219105864E+33 };

// *****************************************************************************
// Returns zero based indices.
static int coset(int lx, int ly, int lz) {
		const int l = lx + ly + lz;
		if (l==0) {
				return 0;
		} else {
				return ncoset[l-1] + ((l-lx) * (l-lx+1)) /2 + lz;
		}
}

// *****************************************************************************
static int min(int x, int y) {
		return (x < y) ? x : y;
}

// *****************************************************************************
static int max(int x, int y) {
		return (x > y) ? x : y;
}

// *****************************************************************************
static int mod(int a, int m)
{
		return (a%m + m) % m;
}

// *****************************************************************************
static void grid_prepare_pab_tau(const int o1,
																 const int o2,
																 const int la_max,
																 const int la_min,
																 const int lb_max,
																 const int lb_min,
																 const int maxco,
																 const double zeta,
																 const double zetb,
																 const mdarray<double, 2, CblasRowMajor> &pab,
																 mdarray<double, 2, CblasRowMajor> &pab_tau) {

		// create a new pab_tau so that mapping pab_tau with pgf_a pgf_b
		// is equivalent to mapping pab with 0.5 * (nabla pgf_a) . (nabla pgf_b)
		// (ddx pgf_a ) (ddx pgf_b) = (lax pgf_{a-1x} - 2*zeta*pgf_{a+1x})*(lbx pgf_{b-1x} - 2*zetb*pgf_{b+1x})

		const int nla = ncoset[la_max+1];
		const int nlb = ncoset[lb_max+1];

		// Initialize with zeros.
		pab_tau.zero();

		for (int lxa=0; lxa<=la_max; lxa++) {
				for (int lxb=0; lxb<=lb_max; lxb++) {
						for (int lya=0; lya<=la_max-lxa; lya++) {
								for (int lyb=0; lyb<=lb_max-lxb; lyb++) {
										for (int lza=max(la_min-lxa-lya, 0); lza<=la_max-lxa-lya; lza++) {
												for (int lzb=max(lb_min-lxb-lyb, 0); lzb<=lb_max-lxb-lyb; lzb++) {
														const int ico = coset(lxa, lya, lza);
														const int jco = coset(lxb, lyb, lzb);

														int ico_l, jco_l;
														// x  (all safe if lxa = 0, as the spurious added terms have zero prefactor)

														ico_l = coset(max(lxa-1, 0), lya, lza);
														jco_l = coset(max(lxb-1, 0), lyb, lzb);
														pab_tau(jco_l, ico_l) += lxa * lxb * pab(o2+jco, o1+ico);
														ico_l = coset(max(lxa-1, 0), lya, lza);
														jco_l = coset((lxb+1), lyb, lzb);
														pab_tau(jco_l, ico_l) += -2.0 * lxa * zetb * pab(o2+jco, o1+ico);
														ico_l = coset((lxa+1), lya, lza);
														jco_l = coset(max(lxb-1, 0), lyb, lzb);
														pab_tau(jco_l, ico_l) += -2.0 * zeta * lxb * pab(o2+jco, o1+ico);
														ico_l = coset((lxa+1), lya, lza);
														jco_l = coset((lxb+1), lyb, lzb);
														pab_tau(jco_l, ico_l) += 4.0 * zeta * zetb * pab(o2+jco, o1+ico);

														// y

														ico_l = coset(lxa, max(lya-1, 0), lza);
														jco_l = coset(lxb, max(lyb-1, 0), lzb);
														pab_tau(jco_l, ico_l) += lya * lyb * pab(o2+jco, o1+ico);
														ico_l = coset(lxa, max(lya-1, 0), lza);
														jco_l = coset(lxb, (lyb+1), lzb);
														pab_tau(jco_l, ico_l) += -2.0 * lya * zetb * pab(o2+jco, o1+ico);
														ico_l = coset(lxa, (lya+1), lza);
														jco_l = coset(lxb, max(lyb-1, 0), lzb);
														pab_tau(jco_l, ico_l) += -2.0 * zeta * lyb * pab(o2+jco, o1+ico);
														ico_l = coset(lxa, (lya+1), lza);
														jco_l = coset(lxb, (lyb+1), lzb);
														pab_tau(jco_l, ico_l) += 4.0 * zeta * zetb * pab(o2+jco, o1+ico);

														// z

														ico_l = coset(lxa, lya, max(lza-1, 0));
														jco_l = coset(lxb, lyb, max(lzb-1, 0));
														pab_tau(jco_l, ico_l) += lza * lzb * pab(o2+jco, o1+ico);
														ico_l = coset(lxa, lya, max(lza-1, 0));
														jco_l = coset(lxb, lyb, (lzb+1));
														pab_tau(jco_l, ico_l) += -2.0 * lza * zetb * pab(o2+jco, o1+ico);
														ico_l = coset(lxa, lya, (lza+1));
														jco_l = coset(lxb, lyb, max(lzb-1, 0));
														pab_tau(jco_l, ico_l) += -2.0 * zeta * lzb * pab(o2+jco, o1+ico);
														ico_l = coset(lxa, lya, (lza+1));
														jco_l = coset(lxb, lyb, (lzb+1));
														pab_tau(jco_l, ico_l) += 4.0 * zeta * zetb * pab(o2+jco, o1+ico);
												}
										}
								}
						}
				}
		}

		// Divide by two.
		//TODO Maybe divide all prefactors above instead.
		for (int ico=0; ico<nla; ico++) {
				for (int jco=0; jco<nlb; jco++) {
					 pab_tau(jco, ico) *= 0.5;
				}
		}
}

// *****************************************************************************
static void grid_prepare_pab_rho(const int o1,
																 const int o2,
																 const int la_max,
																 const int la_min,
																 const int lb_max,
																 const int lb_min,
																 const int maxco,
																 const mdarray<double, 2, CblasRowMajor> &pab,
																 mdarray<double, 2, CblasRowMajor> &pab_rho) {

		const int nla = ncoset[la_max];
		const int nlb = ncoset[lb_max];

		// Initialize with zeros.
		for (int ico=0; ico<nla; ico++) {
				for (int jco=0; jco<nlb; jco++) {
						pab_rho(jco, ico) = 0.0;
				}
		}
		for (int lxa=0; lxa<=la_max; lxa++) {
				for (int lxb=0; lxb<=lb_max; lxb++) {
						for (int lya=0; lya<=la_max-lxa; lya++) {
								for (int lyb=0; lyb<=lb_max-lxb; lyb++) {
										for (int lza=max(la_min-lxa-lya, 0); lza<=la_max-lxa-lya; lza++) {
												for (int lzb=max(lb_min-lxb-lyb, 0); lzb<=lb_max-lxb-lyb; lzb++) {
														const int ico = coset(lxa, lya, lza);
														const int jco = coset(lxb, lyb, lzb);
														pab_rho(jco, ico) = pab(o2+jco, o1+ico);
												}
										}
								}
						}
				}
		}
}

// *****************************************************************************
static void grid_prepare_alpha(const double ra[3],
															 const double rb[3],
															 const double rp[3],
															 const int la_max,
															 const int lb_max,
															 const int lmax,
															 mdarray<double, 4, CblasRowMajor> &alpha) {

		// Initialize with zeros.
		alpha.zero();

		//
		//   compute polynomial expansion coefs -> (x-a)**lxa (x-b)**lxb -> sum_{ls} alpha(ls,lxa,lxb,1)*(x-p)**ls
		//

		for (int iaxis=0; iaxis<3; iaxis++) {
				const double drpa = rp[iaxis] - ra[iaxis];
				const double drpb = rp[iaxis] - rb[iaxis];
				for (int lxa=0; lxa<=la_max; lxa++) {
						for (int lxb=0; lxb<=lb_max; lxb++) {
								double binomial_k_lxa = 1.0;
								double a = 1.0;
								for (int k=0; k<=lxa; k++) {
										double binomial_l_lxb = 1.0;
										double b = 1.0;
										for (int l=0; l<=lxb; l++) {
												alpha(iaxis, lxb, lxa, lxa - l + lxb - k) += binomial_k_lxa * binomial_l_lxb * a * b;
												binomial_l_lxb *= ((double)(lxb - l)) / ((double)(l + 1));
												b *= drpb;
										}
										binomial_k_lxa *= ((double)(lxa - k)) / ((double)(k + 1));
										a *= drpa;
								}
						}
				}
		}
}

// *****************************************************************************
static void grid_prepare_coef(const int la_max,
															const int la_min,
															const int lb_max,
															const int lb_min,
															const int lmax,
															const double prefactor,
															const mdarray<double, 4, CblasRowMajor> &alpha,
															const mdarray<double, 2, CblasRowMajor> &pab,
															std::vector<double> &coef_xyz) {  // TODO: add size of coef_xyz

		const int lp = la_max + lb_max;

		const int n = ((lmax*2+1)*(lmax*2+2))/2;
		double coef_xyt[n];
		double coef_xtt[lmax*2 + 1];
		int lxyz = 0;
		for (int lzp = 0; lzp<=lp; lzp++) {
				for (int lyp = 0; lyp<=lp-lzp; lyp++) {
						for (int lxp = 0; lxp<=lp-lzp-lyp; lxp++) {
								coef_xyz[lxyz] = 0.0;
								lxyz++;
						}
				}
		}

		for (int lzb = 0; lzb<=lb_max; lzb++) {
				for (int lza = 0; lza<=la_max; lza++) {
						int lxy = 0;
						for (int lyp = 0; lyp<=lp-lza-lzb; lyp++) {
								for (int lxp = 0; lxp<=lp-lza-lzb-lyp; lxp++) {
										coef_xyt[lxy] = 0.0;
										lxy++;
								}
								lxy = lxy + lza + lzb;
						}
						for (int lyb = 0; lyb<=lb_max-lzb; lyb++) {
								for (int lya = 0; lya<=la_max-lza; lya++) {
										const int lxpm = (lb_max-lzb-lyb) + (la_max-lza-lya);
										for (int i=0; i<=lxpm; i++) {
												coef_xtt[i] = 0.0;
										}
										for (int lxb = max(lb_min-lzb-lyb, 0); lxb<=lb_max-lzb-lyb; lxb++) {
												for (int lxa = max(la_min-lza-lya, 0); lxa<=la_max-lza-lya; lxa++) {
														const int ico = coset(lxa, lya, lza);
														const int jco = coset(lxb, lyb, lzb);
														const double p_ele = prefactor * pab(jco, ico);
														for (int lxp = 0; lxp<=lxa+lxb; lxp++) {
																coef_xtt[lxp] += p_ele * alpha(0, lxb, lxa, lxp);
														}
												}
										}
										int lxy = 0;
										for (int lyp = 0; lyp<=lya+lyb; lyp++) {
												for (int lxp = 0; lxp<=lp-lza-lzb-lya-lyb; lxp++) {
														coef_xyt[lxy] += alpha(1, lyb, lya, lyp) * coef_xtt[lxp];
														lxy++;
												}
												lxy += lza + lzb + lya + lyb - lyp;
										}
								}
						}
						lxyz = 0;
						for (int lzp = 0; lzp<=lza+lzb; lzp++) {
								int lxy = 0;
								for (int lyp = 0; lyp<=lp-lza-lzb; lyp++) {
										for (int lxp = 0; lxp<=lp-lza-lzb-lyp; lxp++) {
												coef_xyz[lxyz] += alpha(2, lzb, lza, lzp) * coef_xyt[lxy];
												lxy++;
												lxyz++;
										}
										lxy += lza + lzb;
										lxyz += lza + lzb - lzp;
								}
								for (int lyp = lp-lza-lzb+1; lyp<=lp-lzp; lyp++) {
										for (int lxp = 0; lxp<=lp-lyp-lzp; lxp++) {
												lxyz++;
										}
								}
						}
				}
		}
}

// *****************************************************************************
static void grid_fill_map(const bool periodic,
													const int lb_cube,
													const int ub_cube,
													const int cubecenter,
													const int lb_grid,
													const int npts,
													const int ngrid,
													const int cmax,
													int *map) {

		if (periodic) {
				 int start = lb_cube;
				 while (true) {
						const int offset = mod(cubecenter + start, npts)  + 1 - start;
						const int length = min(ub_cube, npts - offset) - start;
						for (int ig=start; ig<=start+length; ig++) {
							 map[ig + cmax] = ig + offset;
						}
						if (start + length >= ub_cube){
								break;
						}
						start += length + 1;
				 }
		} else {
				 // this takes partial grid + border regions into account
				 const int offset = mod(cubecenter + lb_cube + lb_grid, npts) + 1 - lb_cube;
				 // check for out of bounds
				 assert(ub_cube + offset <= ngrid);
				 assert(lb_cube + offset >= 1);
				 for (int ig=lb_cube; ig <= ub_cube; ig++) {
						map[ig + cmax] = ig + offset;
				 }
		}
}


// *****************************************************************************
static void grid_fill_pol(const double dr,
													const double roffset,
													const int lb_cube,
													const int lp,
													const int cmax,
													const double zetp,
													mdarray<double, 3, CblasRowMajor> &pol) {
//
//   compute the values of all (x-xp)**lp*exp(..)
//
//  still requires the old trick:
//  new trick to avoid to many exps (reuse the result from the previous gridpoint):
//  exp( -a*(x+d)**2)=exp(-a*x**2)*exp(-2*a*x*d)*exp(-a*d**2)
//  exp(-2*a*(x+d)*d)=exp(-2*a*x*d)*exp(-2*a*d**2)
//
			const double t_exp_1 = exp(-zetp * pow(dr, 2));
			const double t_exp_2 = pow(t_exp_1, 2);

			double t_exp_min_1 = exp(-zetp * pow(+dr - roffset, 2));
			double t_exp_min_2 = exp(-2 * zetp * (+dr - roffset) * (-dr));
			for (int ig=0; ig >= lb_cube; ig--) {
					const double rpg = ig * dr - roffset;
					t_exp_min_1 *= t_exp_min_2 * t_exp_1;
					t_exp_min_2 *= t_exp_2;
					double pg = t_exp_min_1;
					// pg  = EXP(-zetp*rpg**2)
					for (int icoef=0; icoef<=lp; icoef++) {
							pol(ig+cmax, icoef, 0) = pg;
							pg *= rpg;
					}
			}

			double t_exp_plus_1 = exp(-zetp * pow(-roffset,2));
			double t_exp_plus_2 = exp(-2 * zetp * (-roffset) * (+dr));
			for (int ig=0; ig >= lb_cube; ig--) {
					const double rpg = (1-ig) * dr - roffset;
					t_exp_plus_1 *= t_exp_plus_2 * t_exp_1;
					t_exp_plus_2 *= t_exp_2;
					double pg = t_exp_plus_1;
					// pg  = EXP(-zetp*rpg**2)
					for (int icoef=0; icoef<=lp; icoef++) {
							pol(ig+cmax, icoef, 1) = pg;
							pg *= rpg;
					}
			}
}

// *****************************************************************************
static void grid_collocate_core(const int lp,
																const int cmax,
																const std::vector<double> &coef_xyz,
																const mdarray<double, 4, CblasRowMajor> &pol,
																const mdarray<int, 2, CblasRowMajor> &map,
																const int nspheres,
																const std::vector<int> &sphere_bounds,
																const int ngrid[3],
																mdarray<double, 3, CblasRowMajor> &grid) {

		int sci = 0;
		size_t ops = 0;
		const int kgmin = sphere_bounds[sci++];

		timer.start("outer_loop");
		for (int kg=kgmin; kg <= 0; kg++) {
				const int kg2 = 1 - kg;
				const int k = map(2, kg + cmax);
				const int k2 = map(2, kg2 + cmax);

				// initialize coef_xy
				const int n_coef_xy = (lp+1)*(lp+2)/2;
				double coef_xy[n_coef_xy][2];
				for (int i=0; i < n_coef_xy; i++) {
						coef_xy[i][0] = 0.0;
						coef_xy[i][1] = 0.0;
				}

				int lxyz = 0;
				for (int lzp=0; lzp <= lp; lzp++) {
						int lxy = 0;
						for (int lyp=0; lyp <= lp-lzp; lyp++) {
								for (int lxp=0; lxp <= lp-lzp-lyp; lxp++) {
										coef_xy[lxy][0] += coef_xyz[lxyz] * pol(2, kg+cmax, lzp, 0);
										coef_xy[lxy][1] += coef_xyz[lxyz] * pol(2, kg+cmax, lzp, 1);
										lxyz++;
										lxy++;
								}
								lxy += lzp;
						}
				}

				const int jgmin = sphere_bounds[sci++];
				for (int jg=jgmin; jg <= 0; jg++) {
						const int jg2 = 1 - jg;
						const int j = map(1, jg + cmax);
						const int j2 = map(1, jg2 + cmax);

						// initialize coef_x
						double coef_x[lp+1][4];
						for (int i=0; i < lp+1; i++) {
								for (int j=0; j < 4; j++) {
										coef_x[i][j] = 0.0;
								}
						}

						int lxy = 0;
						for (int lyp=0; lyp <= lp; lyp++) {
								for (int lxp=0; lxp <= lp-lyp; lxp++) {
										coef_x[lxp][0] += coef_xy[lxy][0] * pol(1, jg+cmax, lyp, 0);
										coef_x[lxp][1] += coef_xy[lxy][1] * pol(1, jg+cmax, lyp, 0);
										coef_x[lxp][2] += coef_xy[lxy][0] * pol(1, jg+cmax, lyp, 1);
										coef_x[lxp][3] += coef_xy[lxy][1] * pol(1, jg+cmax, lyp, 1);
										lxy++;
								}
						}

						const int igmin = sphere_bounds[sci++];
						for (int ig=igmin; ig<=0; ig++) {
								const int ig2 = 1 - ig;
								const int i = map(0, ig + cmax);
								const int i2 = map(0, ig2 + cmax);

								double s01 = 0.0;
								double s02 = 0.0;
								double s03 = 0.0;
								double s04 = 0.0;
								double s05 = 0.0;
								double s06 = 0.0;
								double s07 = 0.0;
								double s08 = 0.0;

								for (int lxp=0; lxp <= lp; lxp++) {
										s01 += coef_x[lxp][0] * pol(0, ig+cmax, lxp, 0);
										s02 += coef_x[lxp][1] * pol(0, ig+cmax, lxp, 0);
										s03 += coef_x[lxp][2] * pol(0, ig+cmax, lxp, 0);
										s04 += coef_x[lxp][3] * pol(0, ig+cmax, lxp, 0);
										s05 += coef_x[lxp][0] * pol(0, ig+cmax, lxp, 1);
										s06 += coef_x[lxp][1] * pol(0, ig+cmax, lxp, 1);
										s07 += coef_x[lxp][2] * pol(0, ig+cmax, lxp, 1);
										s08 += coef_x[lxp][3] * pol(0, ig+cmax, lxp, 1);
								}

								grid(k-1, j-1, i-1) += s01;
								grid(k2-1, j-1, i-1) += s02;
								grid(k-1, j2-1, i-1) += s03;
								grid(k2-1, j2-1, i-1) += s04;
								grid(k-1, j-1, i2-1) += s05;
								grid(k2-1, j-1, i2-1) += s06;
								grid(k-1, j2-1, i2-1) += s07;
								grid(k2-1, j2-1, i2-1) += s08;
						}
				}
		}
		timer.stop("outer_loop");
}

// *****************************************************************************
static void grid_collocate_ortho(const int lp,
																 const double zetp,
																 const std::vector<double> &coef_xyz,
																 const double dh[3][3],
																 const double dh_inv[3][3],
																 const double rp[3],
																 const int npts[3],
																 const int lb_grid[3],
																 const bool periodic[3],
																 const int lb_cube[3],
																 const int ub_cube[3],
																 const int nspheres,
																 const std::vector<int> &sphere_bounds,
																 const int ngrid[3],
																 mdarray<double, 3, CblasRowMajor> &grid) {

	 // *** position of the gaussian product
	 //
	 // this is the actual definition of the position on the grid
	 // i.e. a point rp(:) gets here grid coordinates
	 // MODULO(rp(:)/dr(:),npts(:))+1
	 // hence (0.0,0.0,0.0) in real space is rsgrid%lb on the rsgrid ((1,1,1) on grid)

		// cubecenter(:) = FLOOR(MATMUL(dh_inv, rp))
		int cubecenter[3];
		for (int i=0; i<3; i++) {
				double dh_inv_rp = 0.0;
				for (int j=0; j<3; j++) {
						dh_inv_rp += dh_inv[j][i] * rp[j];
				}
				cubecenter[i] = floor(dh_inv_rp);
		}

		double roffset[3];
		for (int i=0; i<3; i++) {
				roffset[i] = rp[i] - ((double) cubecenter[i]) * dh[i][i];
		}

		//cmax = MAXVAL(ub_cube)
		int cmax = INT_MIN;
		for (int i=0; i<3; i++) {
				cmax = max(cmax, ub_cube[i]);
		}

		// a mapping so that the ig corresponds to the right grid point
		mdarray<int, 2, CblasRowMajor> map =
				mdarray<int, 2, CblasRowMajor>(3, 2 * cmax + 1);

		for (int i=0; i<3; i++) {
				grid_fill_map(periodic[i],
											lb_cube[i],
											ub_cube[i],
											cubecenter[i],
											lb_grid[i],
											npts[i],
											ngrid[i],
											cmax,
											map.template at<CPU>(i, 0));
		}

		mdarray<double, 4, CblasRowMajor> pol = mdarray<double, 4, CblasRowMajor>(3, cmax + 1, lp + 1, 2);
		for (int i=0; i<3; i++) {
				mdarray<double, 3, CblasRowMajor> tmp = mdarray<double, 3, CblasRowMajor>(pol.at<CPU>(i, 0, 0, 0), cmax + 1, lp + 1, 2);
				grid_fill_pol(dh[i][i], roffset[i], lb_cube[i], lp, cmax, zetp, tmp);
		}

		timer.start("collocate_core");
		grid_collocate_core(lp,
												cmax,
												coef_xyz,
												pol,
												map,
												nspheres,
												sphere_bounds,
												ngrid,
												grid);
		timer.stop("collocate_core");

}


// *****************************************************************************
static void grid_collocate_general(const int lp,
																	 const double zetp,
																	 const std::vector<double> &coef_xyz,
																	 const double dh[3][3],
																	 const double dh_inv[3][3],
																	 const double rp[3],
																	 const int npts[3],
																	 const int lb_grid[3],
																	 const bool periodic[3],
																	 const int lmax,
																	 const double radius,
																	 const int ngrid[3],
																	 mdarray<double, 3, CblasRowMajor> &grid)
{

//
// transform P_{lxp,lyp,lzp} into a P_{lip,ljp,lkp} such that
// sum_{lxp,lyp,lzp} P_{lxp,lyp,lzp} (x-x_p)**lxp (y-y_p)**lyp (z-z_p)**lzp =
// sum_{lip,ljp,lkp} P_{lip,ljp,lkp} (i-i_p)**lip (j-j_p)**ljp (k-k_p)**lkp
//

		// aux mapping array to simplify life
		mdarray<int, 3, CblasRowMajor> coef_map = mdarray<int, 3, CblasRowMajor>(lp + 1, lp + 1, lp + 1);

		//TODO really needed?
		//coef_map = HUGE(coef_map)
		for (int lzp=0; lzp<=lp; lzp++) {
				for (int lyp=0; lyp<=lp; lyp++) {
						for (int lxp=0; lxp<=lp; lxp++) {
								coef_map(lzp, lyp, lxp) = INT_MAX;
						}
				}
		}

		int lxyz = 0;
		for (int lzp=0; lzp<=lp; lzp++) {
				for (int lyp=0; lyp<=lp-lzp; lyp++) {
						for (int lxp=0; lxp<=lp-lzp-lyp; lxp++) {
								coef_map(lzp, lyp, lxp) = ++lxyz;
						}
				}
		}

		// center in grid coords
		// gp = MATMUL(dh_inv, rp)
		double gp[3];
		for (int i=0; i<3; i++) {
				gp[i] = 0.0;
				for (int j=0; j<3; j++) {
						gp[i] += dh_inv[j][i] * rp[j];
				}
		}

		// transform using multinomials
		mdarray<double,3 , CblasRowMajor> hmatgridp = mdarray<double,3 , CblasRowMajor>(lp+1, 3, 3);
		for (int i=0; i<3; i++) {
				for (int j=0; j<3; j++) {
						hmatgridp(0, j, i) = 1.0;
						for (int k=1; k<=lp; k++) {
								hmatgridp(k, j, i) = hmatgridp(k-1, j, i) * dh[j][i];
						}
				}
		}

		// zero coef_ijk
		const int n_coef_ijk = ((lp+1)*(lp+2)*(lp+3))/6;
		double coef_ijk[n_coef_ijk];
		for (int i=0; i<n_coef_ijk; i++) {
				coef_ijk[i] = 0.0;
		}

		const int lpx = lp;
		for (int klx=0; klx<=lpx; klx++) {
				for (int jlx=0; jlx<=lpx-klx; jlx++) {
						for (int ilx=0; ilx<=lpx-klx-jlx; ilx++) {
								const int lx = ilx + jlx + klx;
								const int lpy = lp - lx;
								for (int kly=0; kly<=lpy; kly++) {
										for (int jly=0; jly<=lpy-kly; jly++) {
												for (int ily=0; ily<=lpy-kly-jly; ily++) {
														const int ly = ily + jly + kly;
														const int lpz = lp - lx - ly;
														for (int klz=0; klz<=lpz; klz++) {
																for (int jlz=0; jlz<=lpz-klz; jlz++) {
																		for (int ilz=0; ilz<=lpz-klz-jlz; ilz++) {
																				const int lz = ilz + jlz + klz;
																				const int il = ilx + ily + ilz;
																				const int jl = jlx + jly + jlz;
																				const int kl = klx + kly + klz;
																				const int lijk= coef_map(kl, jl, il);
																				const int lxyz = coef_map(lz, ly, lx);
																				coef_ijk[lijk-1] += coef_xyz[lxyz-1] *
																						hmatgridp(ilx, 0, 0) * hmatgridp(jlx, 1, 0) * hmatgridp(klx, 2, 0) *
																						hmatgridp(ily, 0, 1) * hmatgridp(jly, 1, 1) * hmatgridp(kly, 2, 1) *
																						hmatgridp(ilz, 0, 2) * hmatgridp(jlz, 1, 2) * hmatgridp(klz, 2, 2) *
																						fac[lx] * fac[ly] * fac[lz] /
																						(fac[ilx] * fac[ily] * fac[ilz] * fac[jlx] * fac[jly] * fac[jlz] * fac[klx] * fac[kly] * fac[klz]);
																		}
																}
														}
												}
										}
								}
						}
				}
		}

		// get the min max indices that contain at least the cube that contains a sphere around rp of radius radius
		// if the cell is very non-orthogonal this implies that many useless points are included
		// this estimate can be improved (i.e. not box but sphere should be used)
		int index_min[3], index_max[3];
		for (int idir=0; idir<3; idir++) {
				index_min[idir] = INT_MAX;
				index_max[idir] = INT_MIN;
		}
		for (int i=-1; i<=1; i++) {
				for (int j=-1; j<=1; j++) {
						for (int k=-1; k<=1; k++) {
								const double x = rp[0] + i * radius;
								const double y = rp[1] + j * radius;
								const double z = rp[2] + k * radius;
								for (int idir=0; idir<3; idir++) {
										const double resc = dh_inv[0][idir] * x + dh_inv[1][idir] * y + dh_inv[2][idir] * z;
										index_min[idir] = min(index_min[idir], floor(resc));
										index_max[idir] = max(index_max[idir], ceil(resc));
								}
						}
				}
		}

		int offset[3];
		for (int idir=0; idir<3; idir++) {
				offset[idir] = mod(index_min[idir] + lb_grid[idir], npts[idir]) + 1;
		}

		// go over the grid, but cycle if the point is not within the radius
		for (int k=index_min[2]; k<=index_max[2]; k++) {
				const double dk = k - gp[2];
				int k_index;
				if (periodic[2]) {
						k_index = mod(k, npts[2]) + 1;
				} else {
						k_index = k - index_min[2] + offset[2];
				}

				// zero coef_xyt
				const int n_coef_xyt = ((lmax*2+1)*(lmax*2+2))/2;
				double coef_xyt[n_coef_xyt];
				for (int i=0; i<n_coef_xyt; i++) {
						coef_xyt[i] = 0.0;
				}

				int lxyz = 0;
				double dkp = 1.0;
				for (int kl=0; kl<=lp; kl++) {
						int lxy = 0;
						for (int jl=0; jl<=lp-kl; jl++) {
								for (int il=0; il<=lp-kl-jl; il++) {
										coef_xyt[lxy++] += coef_ijk[lxyz++] * dkp;
								}
								lxy += kl;
						}
						dkp *= dk;
				}


				for (int j=index_min[1]; j<=index_max[1]; j++) {
						const double dj = j - gp[1];
						int j_index;
						if (periodic[1]) {
								j_index = mod(j, npts[1]) + 1;
						} else {
								j_index = j - index_min[1] + offset[1];
						}

						double coef_xtt[lmax*2 +1];
						for (int i=0; i<=lmax*2; i++) {
								coef_xtt[i] = 0.0;
						}
						int lxy = 0;
						double djp = 1.0;
						for (int jl=0; jl<=lp; jl++) {
								for (int il=0; il<=lp-jl; il++) {
										coef_xtt[il] += coef_xyt[lxy++] * djp;
								}
								djp *= dj;
						}

						// find bounds for the inner loop
						// based on a quadratic equation in i
						// a*i**2+b*i+c=radius**2

						// v = pointj-gp(1)*hmatgrid(:, 1)
						// a = DOT_PRODUCT(hmatgrid(:, 1), hmatgrid(:, 1))
						// b = 2*DOT_PRODUCT(v, hmatgrid(:, 1))
						// c = DOT_PRODUCT(v, v)
						// d = b*b-4*a*(c-radius**2)
						double a=0.0, b=0.0, c=0.0;
						for (int i=0; i<3; i++) {
								const double pointk = dh[2][i] * dk;
								const double pointj = pointk + dh[1][i] * dj;
								const double v = pointj - gp[0] * dh[0][i];
								a += dh[0][i] * dh[0][i];
								b += 2.0 * v * dh[0][i];
								c += v * v;
						}
						double d = b * b -4 * a * (c - radius * radius);
						if (d < 0.0) {
								continue;
						}

						// prepare for computing -zetp*rsq
						d = sqrt(d);
						const int ismin = ceill((-b - d) / (2.0 * a));
						const int ismax = floor((-b + d) / (2.0 * a));
						a *= -zetp;
						b *= -zetp;
						c *= -zetp;
						const int i = ismin - 1;

						// the recursion relation might have to be done
						// from the center of the gaussian (in both directions)
						// instead as the current implementation from an edge
						double exp2i = exp((a * i + b) * i + c);
						double exp1i = exp(2.0 * a * i + a + b);
						const double exp0i = exp(2.0 * a);

						for (int i=ismin; i<=ismax; i++) {
								const double di = i - gp[0];

								// polynomial terms
								double res = 0.0;
								double dip = 1.0;
								for (int il=0; il<=lp; il++) {
										res += coef_xtt[il] * dip;
										dip *= di;
								}

								// the exponential recursion
								exp2i *= exp1i;
								exp1i *= exp0i;
								res *= exp2i;

								int i_index;
								if (periodic[0]) {
										i_index = mod(i, npts[0]) + 1;
								} else {
										i_index = i - index_min[0] + offset[0];
								}
								grid(k_index-1, j_index-1, i_index-1) += res;
						}
				}
		}
}

// *****************************************************************************
static void grid_collocate_internal(const bool compute_tau,
																		const bool use_ortho,
																		const int la_max,
																		const int la_min,
																		const int lb_max,
																		const int lb_min,
																		const double zeta,
																		const double zetb,
																		const double rscale,
																		const double dh[3][3],
																		const double dh_inv[3][3],
																		const double ra[3],
																		const double rab[3],
																		const int npts[3],
																		const int ngrid[3],
																		const int lb_grid[3],
																		const bool periodic[3],
																		const int lmax,
																		const double radius,
																		const int lb_cube[3],
																		const int ub_cube[3],
																		const int nspheres,
																		const std::vector<int> &sphere_bounds,
																		const int maxco,
																		const int o1,
																		const int o2,
																		const mdarray<double, 2, CblasRowMajor> &pab,
																		mdarray<double, 3, CblasRowMajor> &grid){

		const double zetp = zeta + zetb;
		const double f = zetb / zetp;
		const double rab2 = rab[0] * rab[0] + rab[1] * rab[1] + rab[2] * rab[2];
		const double prefactor = rscale * exp(-zeta * f * rab2);
		double rp[3], rb[3];
		for (int i=0; i<3; i++) {
				rp[i] = ra[i] + f * rab[i];
				rb[i] = ra[i] + rab[i];
		}

		//TODO: maybe move grid_prepare_alpha and grid_prepare_coef into if/else blocks.
		int la_max_local, la_min_local, lb_max_local, lb_min_local;
		if (compute_tau) {
				la_max_local = la_max + 1;
				la_min_local = max(la_min - 1, 0);
				lb_max_local = lb_max + 1;
				lb_min_local = max(lb_min - 1, 0);
		} else {
				la_max_local = la_max;
				la_min_local = la_min;
				lb_max_local = lb_max;
				lb_min_local = lb_min;
		}


		mdarray<double, 2, CblasRowMajor> pab_local = mdarray<double, 2, CblasRowMajor>(ncoset[lb_max_local], ncoset[la_max_local]);

		if (compute_tau) {
				timer.start("pab_tau");
				grid_prepare_pab_tau(o1,
														 o2,
														 la_max,
														 la_min,
														 lb_max,
														 lb_min,
														 maxco,
														 zeta,
														 zetb,
														 pab,
														 pab_local);
				timer.stop("pab_tau");
		} else {
				timer.start("pab_rho");
				grid_prepare_pab_rho(o1,
														 o2,
														 la_max,
														 la_min,
														 lb_max,
														 lb_min,
														 maxco,
														 pab,
														 pab_local);
				timer.stop("pab_rho");
		}

		//   *** initialise the coefficient matrix, we transform the sum
		//
		// sum_{lxa,lya,lza,lxb,lyb,lzb} P_{lxa,lya,lza,lxb,lyb,lzb} *
		//         (x-a_x)**lxa (y-a_y)**lya (z-a_z)**lza (x-b_x)**lxb (y-a_y)**lya (z-a_z)**lza
		//
		// into
		//
		// sum_{lxp,lyp,lzp} P_{lxp,lyp,lzp} (x-p_x)**lxp (y-p_y)**lyp (z-p_z)**lzp
		//
		// where p is center of the product gaussian, and lp = la_max + lb_max
		// (current implementation is l**7)
		//

		mdarray<double, 4, CblasRowMajor> alpha = mdarray<double, 4, CblasRowMajor>(3, lmax + 1 , lmax + 1, 2 * lmax + 1);
		grid_prepare_alpha(ra,
											 rb,
											 rp,
											 la_max_local,
											 lb_max_local,
											 lmax,
											 alpha);

		//
		//   compute P_{lxp,lyp,lzp} given P_{lxa,lya,lza,lxb,lyb,lzb} and alpha(ls,lxa,lxb,1)
		//   use a three step procedure
		//   we don't store zeros, so counting is done using lxyz,lxy in order to have
		//   contiguous memory access in collocate_fast.F
		//

		//TODO: original used 2*lmax_global instead of lp, but that was probably
		//      because in Fortran stack variable can not be defined mid-routine and
		//      lp depends on compute_tau.

		const int lp = la_max_local + lb_max_local;
		std::vector<double> coef_xyz((lp + 1) * (lp + 2) * (lp + 3) / 6, 0.0);

		timer.start("grid_prepare_coef");
		grid_prepare_coef(la_max_local,
											la_min_local,
											lb_max_local,
											lb_min_local,
											lmax,
											prefactor,
											alpha,
											pab_local,
											coef_xyz);
		timer.stop("grid_prepare_coef");

		if (use_ortho) {
				timer.start("grid_collocate_ortho");
				grid_collocate_ortho(lp,
														 zetp,
														 coef_xyz,
														 dh,
														 dh_inv,
														 rp,
														 npts,
														 lb_grid,
														 periodic,
														 lb_cube,
														 ub_cube,
														 nspheres,
														 sphere_bounds,
														 ngrid,
														 grid);
				timer.stop("grid_collocate_ortho");
		} else {
				grid_collocate_general(lp,
															 zetp,
															 coef_xyz,
															 dh,
															 dh_inv,
															 rp,
															 npts,
															 lb_grid,
															 periodic,
															 lmax,
															 radius,
															 ngrid,
															 grid);
		}
}


// *****************************************************************************
void grid_collocate_pgf_product_rspace(const bool compute_tau,
																			 const bool use_ortho,
																			 const int la_max,
																			 const int la_min,
																			 const int lb_max,
																			 const int lb_min,
																			 const double zeta,
																			 const double zetb,
																			 const double rscale,
																			 const double dh[3][3],
																			 const double dh_inv[3][3],
																			 const double ra[3],
																			 const double rab[3],
																			 const int npts[3],
																			 const int ngrid[3],
																			 const int lb_grid[3],
																			 const bool periodic[3],
																			 const int lmax,
																			 const double radius,
																			 const int lb_cube[3],
																			 const int ub_cube[3],
																			 const int nspheres,
																			 const std::vector<int> &sphere_bounds,
																			 const int maxco,
																			 const int o1,
																			 const int o2,
																			 const mdarray<double, 2, CblasRowMajor> &pab,
																			 mdarray<double, 3, CblasRowMajor> &grid){

// Uncomment this to dump all tasks to file.
//#define __GRID_DUMP_TASKS

#ifdef __GRID_DUMP_TASKS
		mdarray<double, 3, CblasRowMajor> grid_before = mdarray<double, 3, CblasRowMajor>(grid.size(0), grid.size(1), grid.size(2));

		memcpy(grid_before.template at<CPU>(), grid.template at<CPU>(), sizeof(double) * grid.size());
		grid.zero();

		// double grid_before[ngrid[2]][ngrid[1]][ngrid[0]];
		// for (int i=0; i<ngrid[2]; i++) {
		// for (int j=0; j<ngrid[1]; j++) {
		// for (int k=0; j<ngrid[0]; j++) {
		//     grid_before[i][j][k] = grid[i][j][k];
		//     grid[i][j][k] = 0.0;
		// }
		// }
		// }
#endif

		grid_collocate_internal(compute_tau,
														use_ortho,
														la_max,
														la_min,
														lb_max,
														lb_min,
														zeta,
														zetb,
														rscale,
														dh,
														dh_inv,
														ra,
														rab,
														npts,
														ngrid,
														lb_grid,
														periodic,
														lmax,
														radius,
														lb_cube,
														ub_cube,
														nspheres,
														sphere_bounds,
														maxco,
														o1,
														o2,
														pab,
														grid);

#ifdef __GRID_DUMP_TASKS

		static int counter = 0;
		counter++;
		char filename[100];
		snprintf(filename, sizeof(filename), "grid_collocate_%05i.task", counter);

		const int D = DECIMAL_DIG;  // In C11 we could use DBL_DECIMAL_DIG.
		FILE *fp = fopen(filename, "w+");
		fprintf(fp, "#Grid collocate task v1\n");
		fprintf(fp, "compute_tau %i\n", compute_tau);
		fprintf(fp, "use_ortho %i\n", use_ortho);
		fprintf(fp, "la_max %i\n", la_max);
		fprintf(fp, "la_min %i\n", la_min);
		fprintf(fp, "lb_max %i\n", lb_max);
		fprintf(fp, "lb_min %i\n", lb_min);
		fprintf(fp, "zeta %.*e\n", D, zeta);
		fprintf(fp, "zetb %.*e\n", D, zetb);
		fprintf(fp, "rscale %.*e\n", D, rscale);
		for (int i=0; i<3; i++)
				fprintf(fp, "dh %i %.*e %.*e %.*e\n", i, D, dh[i][0], D, dh[i][1], D, dh[i][2]);
		for (int i=0; i<3; i++)
				fprintf(fp, "dh_inv %i %.*e %.*e %.*e\n", i, D, dh_inv[i][0], D, dh_inv[i][1], D, dh_inv[i][2]);
		fprintf(fp, "ra %.*e %.*e %.*e\n", D, ra[0], D, ra[1], D, ra[2]);
		fprintf(fp, "rab %.*e %.*e %.*e\n", D, rab[0], D, rab[1], D, rab[2]);
		fprintf(fp, "npts %i %i %i\n", npts[0], npts[1], npts[2]);
		fprintf(fp, "ngrid %i %i %i\n", ngrid[0], ngrid[1], ngrid[2]);
		fprintf(fp, "lb_grid %i %i %i\n", lb_grid[0], lb_grid[1], lb_grid[2]);
		fprintf(fp, "periodic %i %i %i\n", periodic[0], periodic[1], periodic[2]);
		fprintf(fp, "lmax %i\n", lmax);
		fprintf(fp, "radius %.*e\n", D, radius);
		if (use_ortho) {
				fprintf(fp, "lb_cube %i %i %i\n", lb_cube[0], lb_cube[1], lb_cube[2]);
				fprintf(fp, "ub_cube %i %i %i\n", ub_cube[0], ub_cube[1], ub_cube[2]);
		}
		fprintf(fp, "nspheres %i\n", nspheres);

		int nspheres_nonzero = 0;
		for (int i=0; i<nspheres; i++) {
				if (sphere_bounds[i] != 0) {
						nspheres_nonzero++;
				}
		}
		fprintf(fp, "nspheres_nonzero %i\n", nspheres_nonzero);

		for (int i=0; i<nspheres; i++) {
				if (sphere_bounds[i] != 0) {
						fprintf(fp, "sphere_bounds %i %i\n", i, sphere_bounds[i]);
				}
		}

		fprintf(fp, "maxco %i\n", maxco);
		fprintf(fp, "o1 %i\n", o1);
		fprintf(fp, "o2 %i\n", o2);

		for (int i=0; i<maxco; i++) {
				for (int j=0; j<maxco; j++) {
						fprintf(fp, "pab %i %i %.*e\n", i, j, D, pab(i, j));
				}
		}

		int ngrid_nonzero = 0;
		for (int i=0; i<ngrid[2]; i++) {
				for (int j=0; j<ngrid[1]; j++) {
						for (int k=0; j<ngrid[0]; j++) {
								if (grid(i, j, k) != 0.0) {
										ngrid_nonzero++;
								}
						}
				}
		}
		fprintf(fp, "ngrid_nonzero %i\n", ngrid_nonzero);

		for (int i=0; i<ngrid[2]; i++) {
				for (int j=0; j<ngrid[1]; j++) {
						for (int k=0; j<ngrid[0]; j++) {
								if (grid(i, j, k) != 0.0) {
										fprintf(fp, "grid %i %i %i %.*e\n", i, j, k, D, grid(i, j, k));
								}
								grid(i, j, k) += grid_before(i, j, k);
						}
				}
		}
		fprintf(fp, "#THE_END\n");
		fclose(fp);
		printf("Wrote %s\n", filename);

#endif

}

//EOF
