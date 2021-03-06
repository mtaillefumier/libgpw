/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2020  CP2K developers group                         *
 *****************************************************************************/

#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#include "grid_base_ref_replay.h"
#include "grid_base_ref_c.h"

#include "rt_graph.hpp"
extern rt_graph::Timer timer;

double grid_collocate_replay(const char* filename, const int cycles)
{
		printf("Reading: '%s'\n", filename);
		FILE *fp = fopen(filename, "r");
		assert(fp != NULL);

		char line[100], key[100];

		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(strcmp(line, "#Grid collocate task v1\n") == 0);

		int compute_tau_i;
		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(sscanf(line, "%s %i", key, &compute_tau_i) == 2);
		assert(strcmp(key, "compute_tau") == 0);
		bool compute_tau = compute_tau_i;

		int use_ortho_i;
		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(sscanf(line, "%s %i", key, &use_ortho_i) == 2);
		assert(strcmp(key, "use_ortho") == 0);
		bool use_ortho = use_ortho_i;

		int la_max;
		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(sscanf(line, "%s %i", key, &la_max) == 2);
		assert(strcmp(key, "la_max") == 0);

		int la_min;
		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(sscanf(line, "%s %i", key, &la_min) == 2);
		assert(strcmp(key, "la_min") == 0);

		int lb_max;
		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(sscanf(line, "%s %i", key, &lb_max) == 2);
		assert(strcmp(key, "lb_max") == 0);

		int lb_min;
		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(sscanf(line, "%s %i", key, &lb_min) == 2);
		assert(strcmp(key, "lb_min") == 0);

		double zeta;
		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(sscanf(line, "%s %le", key, &zeta) == 2);
		assert(strcmp(key, "zeta") == 0);

		double zetb;
		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(sscanf(line, "%s %le", key, &zetb) == 2);
		assert(strcmp(key, "zetb") == 0);

		double rscale;
		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(sscanf(line, "%s %le", key, &rscale) == 2);
		assert(strcmp(key, "rscale") == 0);

		double dh[3][3];
		for (int i=0; i<3; i++) {
				int j;
				assert(fgets(line, sizeof(line), fp) != NULL);
				assert(sscanf(line, "%s %i %le %le %le", key, &j, &dh[i][0], &dh[i][1], &dh[i][2]) == 5);
				assert(strcmp(key, "dh") == 0 && i == j);
		}

		double dh_inv[3][3];
		for (int i=0; i<3; i++) {
				int j;
				assert(fgets(line, sizeof(line), fp) != NULL);
				assert(sscanf(line, "%s %i %le %le %le", key, &j, &dh_inv[i][0], &dh_inv[i][1], &dh_inv[i][2]) == 5);
				assert(strcmp(key, "dh_inv") == 0 && i == j);
		}

		double ra[3];
		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(sscanf(line, "%s %le %le %le", key, &ra[0], &ra[1], &ra[2]) == 4);
		assert(strcmp(key, "ra") == 0);

		double rab[3];
		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(sscanf(line, "%s %le %le %le", key, &rab[0], &rab[1], &rab[2]) == 4);
		assert(strcmp(key, "rab") == 0);

		int npts[3];
		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(sscanf(line, "%s %i %i %i", key, &npts[0], &npts[1], &npts[2]) == 4);
		assert(strcmp(key, "npts") == 0);

		int ngrid[3];
		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(sscanf(line, "%s %i %i %i", key, &ngrid[0], &ngrid[1], &ngrid[2]) == 4);
		assert(strcmp(key, "ngrid") == 0);

		int lb_grid[3];
		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(sscanf(line, "%s %i %i %i", key, &lb_grid[0], &lb_grid[1], &lb_grid[2]) == 4);
		assert(strcmp(key, "lb_grid") == 0);

		int periodic_i[3];
		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(sscanf(line, "%s %i %i %i", key, &periodic_i[0], &periodic_i[1], &periodic_i[2]) == 4);
		assert(strcmp(key, "periodic") == 0);
		bool periodic[3] = {periodic_i[0], periodic_i[1], periodic_i[2]};

		int lmax;
		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(sscanf(line, "%s %i", key, &lmax) == 2);
		assert(strcmp(key, "lmax") == 0);

		double radius;
		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(sscanf(line, "%s %le", key, &radius) == 2);
		assert(strcmp(key, "radius") == 0);

		int *lb_cube, *ub_cube;
		int lb_cube_arr[3], ub_cube_arr[3];
		if (use_ortho) {
				lb_cube = lb_cube_arr;
				assert(fgets(line, sizeof(line), fp) != NULL);
				assert(sscanf(line, "%s %i %i %i", key, &lb_cube[0], &lb_cube[1], &lb_cube[2]) == 4);
				assert(strcmp(key, "lb_cube") == 0);

				ub_cube = ub_cube_arr;
				assert(fgets(line, sizeof(line), fp) != NULL);
				assert(sscanf(line, "%s %i %i %i", key, &ub_cube[0], &ub_cube[1], &ub_cube[2]) == 4);
				assert(strcmp(key, "ub_cube") == 0);
		} else {
				lb_cube = NULL;
				ub_cube = NULL;
		}

		int nspheres;
		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(sscanf(line, "%s %i", key, &nspheres) == 2);
		assert(strcmp(key, "nspheres") == 0);

		std::vector<int> sphere_bounds(nspheres, 0);

		int nspheres_nonzero;
		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(sscanf(line, "%s %i", key, &nspheres_nonzero) == 2);
		assert(strcmp(key, "nspheres_nonzero") == 0);

		for (int i=0; i < nspheres_nonzero; i++) {
				int j, value;
				assert(fgets(line, sizeof(line), fp) != NULL);
				assert(sscanf(line, "%s %i %i", key, &j, &value) == 3);
				assert(strcmp(key, "sphere_bounds") == 0);
				sphere_bounds[j] = value;
		}

		int maxco;
		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(sscanf(line, "%s %i", key, &maxco) == 2);
		assert(strcmp(key, "maxco") == 0);

		int o1;
		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(sscanf(line, "%s %i", key, &o1) == 2);
		assert(strcmp(key, "o1") == 0);

		int o2;
		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(sscanf(line, "%s %i", key, &o2) == 2);
		assert(strcmp(key, "o2") == 0);

		mdarray<double, 2, CblasRowMajor> pab = mdarray<double, 2, CblasRowMajor>(maxco, maxco);
		for (int i=0; i<maxco; i++) {
				for (int j=0; j<maxco; j++) {
						int i2, j2;
						double value;
						assert(fgets(line, sizeof(line), fp) != NULL);
						assert(sscanf(line, "%s %i %i %le", key, &i2, &j2, &value) == 4);
						assert(strcmp(key, "pab") == 0 && i == i2 && j==j2);
						pab(i, j) = value;
				}
		}

		int ngrid_nonzero;
		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(sscanf(line, "%s %i", key, &ngrid_nonzero) == 2);
		assert(strcmp(key, "ngrid_nonzero") == 0);

		mdarray<double, 3, CblasRowMajor> grid_ref = mdarray<double, 3, CblasRowMajor>(ngrid[2], ngrid[1], ngrid[0]);
		grid_ref.zero();

		for (int n=0; n < ngrid_nonzero; n++) {
				int i, j, k;
				double value;
				assert(fgets(line, sizeof(line), fp) != NULL);
				assert(sscanf(line, "%s %i %i %i %le", key, &i, &j, &k, &value) == 5);
				assert(strcmp(key, "grid") == 0);
				grid_ref(i, j, k) = value;
		}

		assert(fgets(line, sizeof(line), fp) != NULL);
		assert(strcmp(line, "#THE_END\n") == 0);

		mdarray<double, 3, CblasRowMajor> grid_test = mdarray<double, 3, CblasRowMajor>(ngrid[2], ngrid[1], ngrid[0]);
		grid_test.zero();

		printf("grid size : %d %d %d\n", ngrid[0], ngrid[1], ngrid[2]);
		printf("radius : %.5lf\n", radius);
		printf("npts : %d %d %d\n", npts[0], npts[1], npts[2]);
		printf("lmin : %d %d\n", la_min, lb_min);
		printf("lmax : %d %d\n", la_max, lb_max);
		printf("nspheres : %d\n", nspheres);
		printf("ub_cube : %d %d %d\n", ub_cube[0], ub_cube[1], ub_cube[2]);
		for (int i=0; i < cycles ; i++) {
				timer.start("grid_collocate_pgf_product_rspace");
				grid_collocate_pgf_product_rspace(compute_tau,
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
																					grid_test);
				timer.stop("grid_collocate_pgf_product_rspace");
		}

		double max_diff = 0.0;

		for (int i=0; i<ngrid[2]; i++) {
				for (int j=0; j<ngrid[1]; j++) {
						for (int k=0; j<ngrid[0]; j++) {
								const double diff = fabs(grid_test(i, j, k) - cycles * grid_ref(i, j, k));
								max_diff = fmax(max_diff, diff);
						}
				}
		}

		printf("Max diff: %le\n", max_diff);
		return max_diff;
}

//EOF
