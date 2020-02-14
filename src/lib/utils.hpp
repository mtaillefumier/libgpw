#ifndef UTILS_HPP_
#define UTILS_HPP_

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

template <typename T> void pack_elems(const int lower_corner[3], /* coordinates of the (0, 0, 0) element of the local part of the grid */
																			const mdarray<int, 2, CblasRowMajor> &yz_non_vanishing_elements, /* non vanishing elements of the polynomials in all three directions. Should be of length period[d] */
																			const mdarray<T, 3, CblasRowMajor> &src, /* local part of the grid */
																			mdarray<T, 3, CblasRowMajor> &dst /* packed data */
		)
{

		const int *__restrict z_int = yz_non_vanishing_elements.template at<CPU>(0, lower_corner[0]);
		const int *__restrict y_int = yz_non_vanishing_elements.template at<CPU>(1, lower_corner[1]);
		const int *__restrict x_int = yz_non_vanishing_elements.template at<CPU>(2, lower_corner[2]);

		int n[3] = {0, 0, 0};
		for (int d = 0; d < 3; d++) {
				for (int s = lower_corner[d]; s < lower_corner[d] + src.size(d); s++)
						n[d] += yz_non_vanishing_elements(d, s);
		}

		if ((n[0] == 0) || (n[1] == 0) || (n[2] == 0)) {
				n[0] = 0;
				n[1] = 0;
				n[2] = 0;
				return;
		}

		dst = mdarray<T, 3, CblasRowMajor>(n[0], n[1], n[2]);

		int z_offset = 0;
		for (int z = 0; z < src.size(0); z++) {

				if (z_int[z] == 0)
						continue;

				const int zmin = z;

				/* continue itterate */
				for (;z_int[z] && (z < src.size(0) - 1); z++);

				const int zmax = z + z_int[z];

				int y_offset = 0;
				for (int y = 0; y < src.size(0); y++) {

						if (y_int[y] == 0)
								continue;

						const int ymin = y;

						/* continue itterate */
						for (;y_int[y] && (y < src.size(1) - 1); y++);

						const int ymax = y + y_int[y];

						int x_offset = 0;
						for (int x = 0; x < src.size(2); x++) {

								if (x_int[x] == 0)
										continue;

								const int xmin = x;

								/* continue itterat*/
								for (;x_int[x] && (x < src.size(1) - 1); x++);

								const int xmax = x + x_int[x];
								for (int z1 = zmin; z1 < zmax; z1++) {
										const int pos_z = z_offset + z1 - zmin;
										for (int y1 = ymin; y1 < ymax; y1++) {
												const int ypos = y_offset + y1 - ymin;
												memcpy(dst.at<CPU>(pos_z, ypos, x_offset),
															 src.template at<CPU>(z1, y1, xmin),
															 sizeof(T) * (xmax - xmin));
										}
								}
								x_offset += (xmax - xmin);
						}
						y_offset += (ymax - ymin);
				}
				z_offset += (zmax - zmin);
		}
}

/* unpack elements and add them to the destinatiuon table */
template <typename T> void unpack_elems_and_add(const int lower_corner[3], /* coordinates of the (0, 0, 0) element of the local part of the grid */
																								const mdarray<int, 2, CblasRowMajor> &yz_non_vanishing_elements, /* non vanishing elements of the polynomials in all three directions. Should be of length period[d] */
																								const mdarray<T, 3, CblasRowMajor> &src, /* packed data */
																								mdarray<T, 3, CblasRowMajor> &dst /* local part of the grid */
		)
{

		const int *__restrict z_int = yz_non_vanishing_elements.template at<CPU>(0, lower_corner[0]);
		const int *__restrict y_int = yz_non_vanishing_elements.template at<CPU>(1, lower_corner[1]);
		const int *__restrict x_int = yz_non_vanishing_elements.template at<CPU>(2, lower_corner[2]);

		int n[3] = {0, 0, 0};
		for (int d = 0; d < 3; d++) {
				for (int s = lower_corner[d]; s < lower_corner[d] + dst.size(d); s++)
						n[d] += (yz_non_vanishing_elements(d, s)  != 0);
		}

		if ((n[0] == 0) || (n[1] == 0) || (n[2] == 0)) {
				n[0] = 0;
				n[1] = 0;
				n[2] = 0;
				return;
		}

		int z_offset = 0;
		for (int z = 0; z < src.size(0); z++) {

				if (z_int[z] == 0)
						continue;

				const int zmin = z;

				/* continue itterate */
				for (;z_int[z] && (z < src.size(0) - 1); z++);

				const int zmax = z + z_int[z];

				int y_offset = 0;
				for (int y = 0; y < src.size(0); y++) {

						if (y_int[y] == 0)
								continue;

						const int ymin = y;

						/* continue itterate */
						for (;y_int[y] && (y < src.size(1) - 1); y++);

						const int ymax = y + y_int[y];

						int x_offset = 0;
						for (int x = 0; x < src.size(2); x++) {

								if (x_int[x] == 0)
										continue;

								const int xmin = x;

								/* continue itterat*/
								for (;x_int[x] && (x < src.size(1) - 1); x++);

								const int xmax = x + x_int[x];
								for (int z1 = zmin; z1 < zmax; z1++) {
										const int pos_z = z_offset + z1 - zmin;
										for (int y1 = ymin; y1 < ymax; y1++) {
												const int ypos = y_offset + y1 - ymin;
												const T*__restrict src1 = src.template at<CPU>(pos_z, ypos, x_offset);
												T*__restrict dst1 = dst.template at<CPU>(z1, y1, xmin);
												for (int x1 = 0; x1 < (xmax - xmin); x1++) {
														dst1[x1] += src1[x1];
												}
										}
								}
								x_offset += (xmax - xmin);
						}
						y_offset += (ymax - ymin);
				}
				z_offset += (zmax - zmin);
		}
}

/* unpack elements and copy them to the destinatiuon table */
template <typename T> void unpack_elems(const int period[3], /* size of the global grid */
																								const int lower_corner[3], /* coordinates of the (0, 0, 0) element of the local part of the grid */
																								const mdarray<int, 2, CblasRowMajor> &yz_non_vanishing_elements, /* non vanishing elements of the polynomials in all three directions. Should be of length period[d] */
																								const mdarray<T, 3, CblasRowMajor> &src, /* packed data */
																								mdarray<T, 3, CblasRowMajor> &dst /* local part of the grid */
		)
{

		const int *__restrict z_int = yz_non_vanishing_elements.template at<CPU>(0, lower_corner[0]);
		const int *__restrict y_int = yz_non_vanishing_elements.template at<CPU>(1, lower_corner[1]);
		const int *__restrict x_int = yz_non_vanishing_elements.template at<CPU>(2, lower_corner[2]);

		int n[3] = {0, 0, 0};
		for (int d = 0; d < 3; d++) {
				for (int s = lower_corner[d]; s < lower_corner[d] + dst.size(d); s++)
						n[d] += yz_non_vanishing_elements(d, s);
		}

		if ((n[0] == 0) || (n[1] == 0) || (n[2] == 0)) {
				n[0] = 0;
				n[1] = 0;
				n[2] = 0;
				return;
		}

		int z_offset = 0;
		for (int z = 0; z < src.size(0); z++) {

				if (z_int[z] == 0)
						continue;

				const int zmin = z;

				/* continue itterate */
				for (;z_int[z] && (z < src.size(0) - 1); z++);

				const int zmax = z + z_int[z];

				int y_offset = 0;
				for (int y = 0; y < src.size(0); y++) {

						if (y_int[y] == 0)
								continue;

						const int ymin = y;

						/* continue itterate */
						for (;y_int[y] && (y < src.size(1) - 1); y++);

						const int ymax = y + y_int[y];

						int x_offset = 0;
						for (int x = 0; x < src.size(2); x++) {

								if (x_int[x] == 0)
										continue;

								const int xmin = x;

								/* continue itterat*/
								for (;x_int[x] && (x < src.size(1) - 1); x++);

								const int xmax = x + x_int[x];
								for (int z1 = zmin; z1 < zmax; z1++) {
										const int pos_z = z_offset + z1 - zmin;
										for (int y1 = ymin; y1 < ymax; y1++) {
												const int ypos = y_offset + y1 - ymin;
												const T*__restrict src1 = src.template at<CPU>(pos_z, ypos, x_offset);
												T*__restrict dst1 = dst.template at<CPU>(z1, y1, xmin);
												memcpy(dst1, src1, sizeof(T) * (xmax - xmin));
										}
								}
								x_offset += (xmax - xmin);
						}
						y_offset += (ymax - ymin);
				}
				z_offset += (zmax - zmin);
		}
}


#endif
