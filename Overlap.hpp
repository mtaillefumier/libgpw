#ifndef _OVERLAP_HPP_
#define _OVERLAP_HPP_

class overlap {
private:
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

    double ra_[3], rb_[3];
    mdarray<double, 2, CblasRowMajor> exponents_;
    mdarray<double, 2, CblasRowMajor> coefs_;
    mdarray<double, 2, CblasRowMajor> radius_;
    mdarray<double, 3, CblasRowMajor> r_ab_;
    mdarray<double, 2, CblasRowMajor> mu_mean_;
    mdarray<double, 2, CblasRowMajor> mu_geo_;
    mdarray<double, 2, CblasRowMajor> kappa_ab_;
    int l1_;
    int l2_;
    mdarray<double, 4, CblasRowMajor> p_alpha_;
    mdarray<double, 4, CblasRowMajor> p_beta_;
    mdarray<double, 4, CblasRowMajor> exp_ab_;
    mdarray<double, 2, CblasRowMajor> v_ab_;

public:
    overlap(double *ra, double *rb, double *exponents_a, double *exponents_b, double *coefs_a, double *coefs_b, int la, int lb, int na_, int nb_);
    ~overlap();
    void compute_polynomial_prefactors(double *dr, double radius);
    void compute_exponential_prefactors(double *dr);
    void calculate_integral_sphere_variant3(const double radius, const double *__restrict dr, mdarray<double, 3, CblasRowMajor> &V, const int center[3]);
    void compute_sliding_window(const double *rab, const int *radius_int,
                                const double *dr, mdarray<double, 3, CblasRowMajor> &V,
                                mdarray<double, 3, CblasRowMajor> &V_reduced);
}
