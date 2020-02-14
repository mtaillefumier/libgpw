#ifndef POLYNOMIALS_HPP
#define POLYNOMIALS_HPP


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
						for (int a1 = 0; a1 < (int)p_alpha_beta_folded.size(1); a1++) {
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

		p_alpha = mdarray<T, 3, CblasRowMajor>(3, l + 2, 2 * n_max + 1);
		p_alpha.zero();

		// std::cout << r_a[0] << " " << r_a[1] << " " << r_a[2] << "\n";
		// std::cout << std::fmod(r_a[0], dr[0]) << " " << std::fmod(r_a[1], dr[1]) << " " << std::fmod(r_a[2], dr[2]) << "\n";

		for (int d = 0; d < 3; d++) {
				double rab = std::fmod(r_a[d], dr[d]);
				/*
				 *	Store the (x - x_i) values as well. It is only used when we consider
				 * the non-orthorombic case
				 */
				T *__restrict dst = p_alpha.template at<CPU>(d, l + 1, 0) + length__[d];
				for (int i = - length__[0]; i <= length__[0]; i++) {
						dst[i] = ((double)i) * dr[d] + rab;
				}

				dst = p_alpha.template at<CPU>(d, 0, 0);
				const T *__restrict src = p_alpha.template at<CPU>(d, l + 1, 0);
				for (int i = 0; i < (2 * length__[0] + 1); i++) {
						dst[i] = exp(-mu_mean * src[i] * src[i]);
				}


				rab = std::fmod(r_a[d], dr[d]);
				for (int m = 1; m < l + 1; m++) {
						const T *__restrict src1 = p_alpha.template at<CPU>(d, m - 1, 0);
						dst = p_alpha.template at<CPU>(d, m, 0);
						for (int i = 0; i < (2 * length__[d] + 1); i++) {
								dst[i] = src1[i] * src[i];
						}
				}
		}

		length__[0] = 2 * length__[0] + 1;
		length__[1] = 2 * length__[1] + 1;
		length__[2] = 2 * length__[2] + 1;
}

template <typename T> bool copy_poly_in_tmp(const int dir,
																						const int pos,
																						const int period,
																						const int lower_corner,
																						const int upper_corner,
																						const mdarray<T, 3, CblasRowMajor> &polynomials,
																						const int poly_min, /* position of the starting point of the gaussian in grid coordinates */
																						const int poly_max, /* position of the ending point of the gaussian in grid coordonates (non periodic boundaries conditions) */
																						mdarray<T, 3, CblasRowMajor> &polynomials_tmp,
																						mdarray<T, 2, CblasRowMajor> &xi)
{
		// check if we have no zero polynomials elements

		// check if the window at position pos [pos * period, (pos + 1) * period] intersects [poly_min .. poly_max]

		// no intersection
		if ((pos + 1) * period < poly_min)
				return false;

		// no intersection
		if (pos * period > poly_max)
				return false;

		// [pos * period, (pos + 1) * period) intersects [poly_min .. poly_max]

		// what are the boundaries of the intersection

		const int i_min = std::max(pos * period, poly_min);
		const int i_max = std::min((pos + 1) * period, poly_max);

		// we have the interval

		// now check if the window [lower_corner..upper_corner) is there
		// std::cout << "ZB : " << pos * period << " " << (pos + 1) * period << "\n";
		// std::cout << poly_min << " " << poly_max << "\n";
		// std::cout << i_min << " " << i_max << "\n";

		int j_min =std::max(pos * period + lower_corner, i_min);
		int j_max =std::min(pos * period + upper_corner, i_max);

		if (j_min > pos * period + upper_corner)
				return false;

		if (j_max < pos * period + lower_corner)
				return false;

		j_min = (j_min + 32 * period) % period - lower_corner;
		j_max = (j_max + 32 * period - 1) % period - lower_corner;
		j_max++;

		// the cell is inside the window

		const int offset = std::max(poly_min, pos * period + lower_corner) - poly_min;

		/* z1 and z2 are within the local window */
		for (int alpha = 0; alpha < polynomials.size(1) - 1; alpha++) {
				memcpy(polynomials_tmp.template at<CPU>(dir, alpha, j_min),
							 polynomials.template at<CPU>(dir, alpha, offset),
							 (j_max - j_min) * sizeof(T));
		}

		/* xi contains the x - xi that are used for computing[] */
		memcpy(xi.template at<CPU>(dir, j_min),
					 polynomials.template at<CPU>(dir, polynomials.size(1) - 1, offset),
					 (j_max - j_min) * sizeof(T));

		return true;
}

void find_interval(const std::vector<int> &folded_table_index, int *__restrict interval_, int &kmax_) {

		for (int i = 0; i < 16; i++)
				interval_[i] = 0;

		int k = 0;
		kmax_ = 0;
		for (int i = 0; i < (int)folded_table_index.size(); i++) {
				if (!folded_table_index[i])
						continue;

				const int imin = i;
				for (;folded_table_index[i] && (i < ((int)folded_table_index.size() - 1));i++);

				int imax = i;

				interval_[k] = imin;
				k++;

				if ((folded_table_index[i] != 0) && (i == ((int)folded_table_index.size() - 1)))
						imax++;
				interval_[k] = imax;
				k++;
				kmax_++;
		}
}

void find_interval(const int *folded_table_index, const int length, int *__restrict interval_, int &kmax_) {

		for (int i = 0; i < 16; i++)
				interval_[i] = 0;

		int k = 0;
		kmax_ = 0;
		for (int i = 0; i < length; i++) {
				if (!folded_table_index[i])
						continue;

				const int imin = i;
				for (;folded_table_index[i] && (i < (length - 1));i++);

				int imax = i;

				interval_[k] = imin;
				k++;

				if ((folded_table_index[i] != 0) && (i == (length - 1)))
						imax++;
				interval_[k] = imax;
				k++;
				kmax_++;
		}
}


#endif
