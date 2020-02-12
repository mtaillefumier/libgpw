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

/*
 * Evaluate the polynomials (x - x_i) ^ \alpha on an interval [-radius, radius]
 * with regularly spaced points
 */

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
