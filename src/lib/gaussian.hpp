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



class two_gaussians {
private :
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


    mdarray<double, 3, CblasRowMajor> p_alpha_;
    mdarray<double, 3, CblasRowMajor> p_beta_;
    mdarray<double, 2, CblasRowMajor> coefs_;
    int l1_, l2_;
    double dr_[3];
    double ra_[3], rb_[3];
    double rab_[3];
    double radius_;
    double mu_mean_;
    doubla kappa_;
    double mu_ab_;
    mdarray<double, 3, CblasRowMajor> Pot_;
    std::vector<double> x_12, y_12, z_12;
    std::vector<double> yp_12_i, zp_12_i;
    int p_alpha_max_[3];
    int p_beta_max_[3];


    two_gaussians(const double *dr, const int *Potential_dims_) {
        pot_ = Potential_;
        potential_dimensions_[0] = dims_[0];
        potential_dimensions_[1] = dims_[1];
        potential_dimensions_[2] = dims_[2];
        dr_[0] = dr[0];
        dr_[1] = dr[1];
        dr_[2] = dr[2];
        p_alpha_ = mdarray<double, 3, CblasRowMajor>(3, 10, 128);
        p_beta_ = mdarray<double, 3, CblasRowMajor>(3, 10, 128);
        Pot_ = mdarray<double, 3, CblasRowMajor>(dims_[0], dims_[1], dims_[2]);
        x_12(128, 0.0);
        y_12(128, 0.0);
        z_12(128, 0.0);

        xp_12(dims_[2], 0.0);
        yp_12(dims_[1], 0.0);
        zp_12(dims_[0], 0.0);

        yp_12_i(dims_[1], 0);
        zp_12_i(dims_[0], 0);
    }

    ~two_gaussians()
        {
            p_alpha_.deallocate();
            p_beta_.deallocate();
            x_12.clear();
            y_12.clear();
            z_12.clear();
            xp_12.clear();
            yp_12.clear();
            zp_12.clear();
        }

    void calculate_polynomials()
        {
            // l is the angular momentum. This gives the degree of the polynomial. For
            // the forces we need polynomials of degree l + 1

            l1_ = l1;
            l2_ = l2;
            // first compute the max of n_alpha
            p_alpha_max_[0] = radius_ / dr_[0];
            p_alpha_max_[1] = radius_ / dr_[1];
            p_alpha_max_[2] = radius_ / dr_[1];

            p_alpha_.zero();
            p_beta_.zero();
            for (int j = 0; j < 3; j++) {
                for (int i = -p_alpha_max_[j]; i <= p_alpha_max_[j]; i++) {
                    p_alpha_(j, 0, i + n_max) = exp(-mu_mean * (r_ab[j] - i * dr[j]) * (r_ab[j] - i * dr[j]));
                }

                for (int i = -p_alpha_max_[j]; i <= p_alpha_max_[j]; i++) {
                    p_beta_(j, 0, i + n_max) = p_alpha_(j, 0, i + n_max);
                }

                // we add the extra exponent for the forces
                // [0..n_max-1] positive r_alpha from 0..n_max * dr_alpha
                // [n_max,.. 2 n_max + 1] negative r_alpha
                // fft convention for the frequency domain
                for (int m = 1; m < l + 2; m++) {
                    for (int i = -n_max; i <= n_max; i++) {
                        p_alpha(j, m, i + n_max) = (r_a[j] - i * dr[j]) * p_alpha(j, m - 1, i + n_max);
                    }
                    for (int i = -n_max; i <= n_max; i++) {
                        p_beta(j, m, i + n_max) = (r_b[j] - i * dr[j]) * p_beta(j, m - 1, i + n_max);
                    }
                }
            }
        }

    void new_set_of_gaussians(const double *r_a,
                              const double *r_b,
                              const double *r_ab,
                              const double sigma_a,
                              const double sigma_b,
                              const double radius,
                              const double *coeffs) {
        ra_[0] = r_a[0];
        ra_[1] = r_a[1];
        ra_[2] = r_a[2];

        rb_[0] = r_b[0];
        rb_[1] = r_b[1];
        rb_[2] = r_b[2];

        mu_mean_ = (sigma_a + sigma_b);
        for (int i = 0; i < 3; i++)
            rab[i] = (sigma_a * r_a[i] + sigma_b * r_b[i])/(*mu_mean);

        mu_ab_ = sigma_a * sigma_b /(*mu_mean);
        double tmp = (r_a[0] - r_b[0]) * (r_a[0] - r_b[0]) +
            (r_a[1] - r_b[1]) * (r_a[1] - r_b[1]) +
            (r_a[2] - r_b[2]) * (r_a[2] - r_b[2]);
        kappa_ = exp(- tmp * mu_ab);

        radius_ = radius;

        calculate_polynomials();

        center_[0] = r_ab[0] / dr_[0];
        center_[1] = r_ab[1] / dr_[1];
        center_[2] = r_ab[2] / dr_[2];

        nonzero_values(center_[0], dims_[0], p_alpha_max_[0], *zp_i[0]);
        nonzero_values(center_[1], dims_[1], p_alpha_max_[1], *yp_i[0]);

        calculate_polynomials();

        coefs = mdarray<double, 2, CblasRowMajor> (coeffs, length_[l1], length_[l2]);
    }

    inline void compute_indice(const int n, int &alpha, int &beta, int &gamma) {
        gamma = (n & 0xFF0000) >> 16;
        beta = (n & 0xFF00) >> 8;
        alpha = (n & 0xFF);
    }

    void compute_nonzero_values(const int center, const int period, const int length, int *non_zeroes_) {
        for (int  i = 0; i < period; i++) {
            non_zeroes_[i] = 0;
        }

        if ((period - center) >= (length + 1) / 2) {
            for (int i = center; i < center + (length + 1) / 2; i++) {
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
                                  double *unfolded_table,
                                  double *folded_table) {
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

    void collocate() {
        for (int m1 = 0; m1 <  length_[l1]; m1++) {
            int gamma1, beta1, alpha1;
            compute_indice(exponents[offset_[l1] + m1], alpha1, beta1, gamma1);
            for (int m2 = 0; m2 <  length_[l2]; m2++) {
                int gamma2, beta2, alpha2;
                compute_indice(exponents[offset_[l2] + m2], alpha2, beta2, gamma2);

                // multiply by the prefactor
                {
                    const double *__restrict pb = p_beta.template at<CPU>(0, alpha1, 0);
                    const double *__restrict pa = p_alpha.template at<CPU>(0, alpha2, 0);
                    for (int x = 0; x < x_12.size(); x++) {
                        x_12[x] = coef(m1, m2) * pa[x] * pb[x];
                    }
                }

                {
                    const double *__restrict pb = p_beta.template at<CPU>(1, beta1, 0);
                    const double *__restrict pa = p_alpha.template at<CPU>(1, beta2, 0);
                    for (int x = 0; x < y_12.size(); x++) {
                        y_12[x] = pa[x] * pb[x];
                    }
                }

                {
                    const double *__restrict pb = p_beta.template at<CPU>(2, gamma1, 0);
                    const double *__restrict pa = p_alpha.template at<CPU>(2, gamma2, 0);
                    for (int x = 0; x < z_12.size(); x++) {
                        z_12[x] = pa[x] * pb[x];
                    }
                }

                compute_folded_integrals(ctr[2], dims_[2], p_alpha_max_[2], x_12, xp_12);
                compute_folded_integrals(ctr[1], dims_[1], p_alpha_max_[1], y_12, yp_12);
                compute_folded_integrals(ctr[0], dims_[0], p_alpha_max_[0], z_12, zp_12);

                for (int z = 0; z < V.size(0); z++) {
                    if (zp_12_i[z]) {
                        for (int y = 0; y < V.size(1); y++) {
                            if (yp_12_i[y]) {
                                const T scal = zp_12[z] * yp_12[y];
                                T *__restrict v = Pot_.template at<CPU>(z, y, 0);
                                const T *__restrict xx = &xp_12[0];
                                for (int x = 0; x < V.size(2); x++) {
                                    v[x] += scal * xx[x];
                                }
                            }
                        }
                    }
                }

                memset(&xp_12[0], 0, sizeof(double) * xp_12.size());
                memset(&yp_12[0], 0, sizeof(double) * yp_12.size());
                memset(&zp_12[0], 0, sizeof(double) * zp_12.size());
            }
        }
    }
}
