#include "overlap.hpp"

void overlap::periodic_boundaries_conditions(const double *rb, const double *periods)
{
    double dist = 100000000;

    int shift[3];
    for (int z = -1; z <= 1; z++) {
        double dz = z * periods[0];
        for (int y = -1; y <= 1; y++) {
            double dy = y * periods[1];
            for (int x = -1; x <= 1; x++) {
                double dx = x * periods[2];
                double rabx = ra_[2] + dx - rb[2];
                double raby = ra_[1] + dx - rb[1];
                double rabz = ra_[0] + dx - rb[0];
                if (dist > rabx * rabx + raby * raby + rabz * rabz) {
                    dist = rabx * rabx + raby * raby + rabz * rabz;
                    shift[0] = z;
                    shift[1] = y;
                    shift[2] = x;
                }
            }
        }
    }

    rb_[0] = rb[0] - shift[0] * periods[0];
    rb_[1] = rb[1] - shift[1] * periods[1];
    rb_[2] = rb[2] - shift[2] * periods[2];
 }

overlap::overlap(double *ra, double *rb,
                 double *exponents_a, double *exponents_b,
                 double *coefs_a, double *coefs_b,
                 int la, int lb,
                 int na_, int nb_,
                 mdarray<double, 2, CblasRowMajor> &dr, // spacing of the differnt grids
                 double *periods)
{
    l1_ = la;
    l2_ = lb;

    exponents_a_ = std::vector<double>(na_);
    exponents_b_ = std::vector<double>(nb_);
    coefs_ = mdarray<double, 2, CblasRowMajor>(na_, nb_);
    r_ab_ = mdarray<double, 2, CblasRowMajor>(na_, nb_, 3);
    mu_mean_ = mdarray<double, 2, CblasRowMajor>(na_, nb_);
    mu_geo_ = mdarray<double, 2, CblasRowMajor>(na_, nb_);
    kappa_ab_ = mdarray<double, 2, CblasRowMajor>(na_, nb_);

    v_ab_ = mdarray<double, 2, CblasRowMajor>(length_[l1_], length_[l2_]);

    ra_[0] = ra[0];
    ra_[1] = ra[1];
    ra_[2] = ra[2];

    // set up rb such that pbc are taken into account

    periodic_boundaries_conditions(rb, periods);

    for (int l = 0; l < exponents_a_.size(); l++)
        exponents_a_[l] = exponents_a[l];

    for (int l = 0; l < exponents_b_.size(); l++)
        exponents_b_[l] = exponents_b[l];

    for (int m1 = 0; m1 < exponents_b_.size(); m1++) {
        for (int m2 = 0; m2 < exponents_b_.size(); m2++) {
            mu_mean(m1, m2) = exponents_a_[m1] + exponents_b_[m2];
            mu_geo_(m1, m2) = exponents_a_[m1] * exponents_b_[m2] / mu_mean(m1, m2);
        }
    }

    for (int m1 = 0; m1 < exponents_a_.size(); m1++) {
        for (int m2 = 0; m2 < exponents_b_.size(); m2++) {
            r_ab_(m1, m2, 0) = (exponents_a_[m1] * ra[0] + exponents_b_[m2] * rb[0]) / mu_mean(m1, m2);
            r_ab_(m1, m2, 1) = (exponents_a_[m1] * ra[1] + exponents_b_[m2] * rb[1]) / mu_mean(m1, m2);
            r_ab_(m1, m2, 2) = (exponents_a_[m1] * ra[2] + exponents_b_[m2] * rb[2]) / mu_mean(m1, m2);
        }
    }

    const double scal = (rb[0] - ra[0]) * (rb[0] - ra[0]);
    scal += (rb[1] - ra[1]) * (rb[1] - ra[1]);
    scal += (rb[2] - ra[2]) * (rb[2] - ra[2]);

    for (int m1 = 0; m1 < exponents_a_.size(); m1++) {
        for (int m2 = 0; m2 < exponents_b_.size(); m2++) {
            kappa_ab_(m1, m2) = exp(-mu_geo_(m1, m2) * scal);
        }
    }

    // now the radius can be estimated from the integration of x^10 exp (-x ^ 2).
    // if we replace exp (- alpha x ^ 2) x ^ n with y = sqrt(alpha) x we have

    // exp (-alpha x ^ 2) x ^ n = alpha ^ (-n / 2) y^n exp (- y ^ 2)

    // now y^n exp (- y ^ 2) converges very fast. Typically an interval y \in
    // [-5..5] in enough to have it converged for discretation \delta y \in [0.01..0.5]

    // so the radius in the original coordinates will be [-5..5] / sqrt(alpha)
    // with any discretization

    // to be safe we can simply take [-6..6] / sqrt(alpha)

    for (int m1 = 0; m1 < exponents_a_.size(); m1++) {
        for (int m2 = 0; m2 < exponents_b_.size(); m2++) {
             radius_(m1, m2) = 7.0 / sqrt(mu_mean(m1, m2));
        }
    }

    // need to add the computation of the best grid for computations

    for (int m1 = 0; m1 < exponents_a_.size(); m1++) {
        for (int m2 = 0; m2 < exponents_b_.size(); m2++) {
            // is the radius larger than the period.
            rewind_(m1, m2) = (radius > periods[0]) * 4 + (radius > periods[1]) * 2 + (radius > periods[2]);
        }
    }
}

~overlap::overlap()
{
    exponents_a_.clear();
    exponents_b_.clear();
}

void overlap::compute_polynomial_prefactors(double *dr, double mu_mean_, double radius, int *radius_int)
{
    for (int s = 0; s < 3; s++)
        radius_int[s] = radius_(mu1, mu2) / dr[s];

    rad_max = std::max(std::max(rad_int_[0], rad_int_[1]), rad_int_[2]);
    p_alpha = mdarray<double, 3, CblasRowMajor>(l1_ + 1, 3, rad_max);
    p_beta = mdarray<double, 3, CblasRowMajor>(l2_ + 1, 3, rad_max);

    p_alpha_.zero();
    p_beta_.zero();

    for (int s = 0; s < 3; s++) {
        for (int x = - radius_int[s]; x <= radius_int[s]; x++) {
            p_alpha_(0, s, x + radius_int[s]) = exp(-(x * dr[s] - rab_[s]) * (x * dr[s] - rab_[s]) * mu_mean_(m1, m2));
            p_beta_(0, s, x + radius_int[s]) = 1;
        }
        for (int m1 = 1; m1 < l1_; m1++) {
            for (int x = - radius_int[s]; x <= radius_int[s]; x++) {
                p_alpha_(m1, s, x) = (s * dr[s] - ra) * p_alpha(m1, s - 1, x);
            }
        }

        for (int m2 = 1; m2 < l2_; m2++) {
            for (int x = - radius_int[s]; x <= radius_int[s]; x++) {
                p_beta_(m2, s, x) = (s * dr[s] - rb) * p_beta_(m2, s - 1, x);
            }
        }
    }
}

void overlap::fold_table(const int deg, const int mu1, const int mu2, double dr, const int a1, const int a2, std::vector<double> &x_12)
{
    const double *__restrict pb = p_beta.at<CPU>(2, gamma1, p_beta_.size(2) / 2);
    const double *__restrict pa = p_alpha.at<CPU>(2, gamma2, p_beta_.size(2) / 2);
    const int num_rep = radius_(m1, m2) / (x.size() * dr);

    for (int x = 0; x < x_12.size(); x++) {
        x_[x] = pa[x] * pb[x];
        for (int n = 1; n < num_rep; n++) {
            x_12[x] += pa[x + n * x_12.size()] * pb[x +  n * x_12.size()];
            x_12[x] += pa[x - n * x_12.size()] * pb[x -  n * x_12.size()];
        }
    }
}

void overlap::calculate_integral_sphere_variant3(const double *__restrict dr, mdarray<double, 3, CblasRowMajor> &V)
{
    // I assume for now that the potential is centered around 0
    // first compute the radius in integer coordinates
    int radius_int[3];
    std::vector<double> tmp(2 * radius_int[2] + 1, 0.0);
    std::vector<double> x_12(2 * radius_int[2] + 1, 0.0);
    std::vector<double> y_12(2 * radius_int[1] + 1, 0.0);
    std::vector<double> z_12(2 * radius_int[0] + 1, 0.0);

    mdarray<double, 3, CblasRowMajor> V_reduced;


    for (int mu1 = 0; mu1 < mu_mean.size(0); mu1++) {
        for (int mu2 = 0; mu2 < mu_mean.size(1); mu2++) {

            compute_polynomial_prefactors(dr, mu1, mu2, radius_int);
            V_reduced = mdarray<double, 3, CblasRowMajor>(std::min(V.size(0), radius_int[0]),
                                                          std::min(V.size(1), radius_int[1]),
                                                          std::min(V.size(2), radius_int[2]));

            // v_reduced and the polynomials are all centered around the same
            // discrete point
            compute_sliding_window(dr, mu1, mu2, V, V_reduced);

            for (int m1 = 0; m1 <  length_[l1]; m1++) {
                int gamma1, beta1, alpha1;
                compute_indice(exponents[offset_[l1] + m1], alpha1, beta1, gamma1);
                for (int m2 = 0; m2 <  length_[l2]; m2++) {
                    int gamma2, beta2, alpha2;
                    compute_indice(exponents[offset_[l2] + m2], alpha2, beta2, gamma2);

                    fold_table(0, mu1, mu2, dr[0], alpha1, alpha2, x_12);
                    fold_table(1, mu1, mu2, dr[1], alpha1, alpha2, y_12);
                    fold_table(2, mu1, mu2, dr[2], alpha1, alpha2, z_12);

                    for (int z = -radius_int[0]; z <= radius_int[0]; z++) {
                        // compute the radius of the disc at fixed z
                        double radius_y = sqrt(radius * radius - z * dr[0] * z * dr[0]);
                        int y_r = lrint(radius_y / dr[1]);

                        for (int y = -y_r; y <= y_r; y++) {
                            // compute the segment length for the x direction
                            int x_r = lrint(sqrt(radius_y * radius_y - y * dr[1] * y * dr[1]) / dr[2]);
                            const double pre_factor2 = z_12[z + radius_int[0]] * y_12[y + radius_int[1]];

                            // this is where things get a tad ugly.

                            // first I compute the starting index correponding to the element -r_x (modulo the period)
                            if (r_x < V_reduced.size(2)) {
                                // it is very simple
                                const Pot1 = V_reduced.at<CPU>((z + radius_int[0] + V_reduced.size(0) / 2) % V_reduced.size(0),
                                                               (y + radius_int[1] + V_reduced.size(1) / 2) % V_reduced.size(1),
                                                               V_reduced.size(2) / 2 - x_r);

                                for (int x = 0; x < 2 * x_r + 1; x++)
                                    tp[x] += pre_factor2 * pot1[x];
                            } else {
                                const int starting_index = (-x_r + radius_int[2] + V.size(2) / 2)%V.size(2);
                                const double *__restrict const pot1 = V_reduced.at<CPU>((z + radius_int[0] + V_reduced.size(0) / 2) % V_reduced.size(0),
                                                                                        (y + radius_int[1] + V_reduced.size(1) / 2) % V_reduced.size(1),
                                                                                        0);

                                double *__restrict tp = &tmp[-xr + radius_int[0]];
                                // x = -x_r.. -x_r + V.size(2)
                                for (int x = starting_index; x < V_reduced.size(2); x++)
                                    tp[x - starting_index] += pre_factor2 * pot1[x];

                                const int num_rep = (2 * x_r + 1 - starting_index) / V.size(2);
                                const int reminder = (2 * x_r + 1 - starting_index) % V.size(2);
                                for (int rep = 0; rep < num_rep; rep++) {
                                    for (int x = 0; x < V.size(2); x++)
                                        tp[x] += pre_factor2 * pot1[x];
                                }
                                for (int x = 0; x < reminder; x++)
                                    tp[x] += pre_factor2 * pot1[x];
                            }
                        }
                    }

                    // compute the final reduction that gives the integral
                    double res1 = 0.0;
#pragma ivdep
                    for (int s = 0; s < tmp.size(); s++)
                        res1 += tmp[s]  * x12[x];

                    // do the contraction
                    res(m1, m2) += coefs_(mu1, mu2) * res1;
                    std::memset(&tmp[0], 0, sizeof(double) * tmp.size());
                }
            }
        }
    }
}

void overlap::compute_sliding_window(const double *rab,
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

    std::cout << ind[0] << " " << ind[1] << " " << ind[2] << std::endl;

    std::vector<double> tmp (3 * V.size(2));
    // now I want to go from ind - L/2 to ind + L/2 in each direction
    for (int z = -V_reduced.size(0) / 2; z < V_reduced.size(0) / 2; z++) {
        int z1 = (ind[0] - z + V.size(0)) % V.size(0);
        for (int y = -V_reduced.size(1) / 2; y < V_reduced.size(1) / 2; y++) {
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

            memcpy(V_reduced.at<CPU>(z + V_reduced.size(0) / 2, y + V_reduced.size(1) / 2, 0),
                   &tmp[ind[2] - V_reduced.size(2) / 2,
                   sizeof(double) * V_reduced.size(2));
        }
    }
}
