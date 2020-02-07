#include <cuda.h>

template<typename T> void copy_data_to_device(const std::vector<T> &src, T* dst, const int device)
{
		cudaSetDevice(device);
		if (src.empty()) {
				sdt::cout << "Error : there is no data to copy\n";
		}

		if (dst != nullptr) {
				cudaFree(dst);
		}

		cudaMalloc(&dst, sizeof(T) * src.size());
		cudaMemcpy(dst, &src[0], sizeof(T) * src.size(), cudaMemcpyHostToDevice);
}

typedef struct gaussian_product_ {
		std::vector<double> coef_;
		std::vector<int> l1_l2;
		std::vector<double> xi_xj;
		std::vector<double> mu_i_mu_j;
		std::vector<double> weights_;
		std::vector<unsigned int> offsets_;
		dim3 block, thread;
		double *coef_dev;
		int *l1_l2_dev;
		double *xi_xj_dev;
		double *mu_ij_dev;
		double *weight_dev;
		unsigned int *offsets_dev;
		double3 dr;
		dim3 grid_size_;
		double *density_dev;
		double *density_derivatives_dev;

		add_gaussian(const int l1, const int l2,
								 const double *xi, const double *xj,
								 cosnt double mui, const double muj,
								 const double weights, cosnt double *coef)
				{
						l1_l2.push_back((l1 << 16) + l2);
						xi_xj.push_back(xi[0]);
						xi_xj.push_back(xi[1]);
						xi_xj.push_back(xi[2]);
						xi_xj.push_back(xj[0]);
						xi_xj.push_back(xj[1]);
						xi_xj.push_back(xj[2]);
						mu_i_mu_j.push_back(mui);
						mu_i_mu_j.push_back(muj);
						weights_.push_back(weights);
						offsets_.push_back(offsets_.back() + length_[l2] * length[l1]);
						for (int s = 0; s < length_[l2] * length[l1]; s++) {
								coefs_.push_back(coef[s]);
						}
				}

		void initialize_calculations(int grid_dim[3])
				{
						copy_data_to_device<double>(coef_, coef_dev, 0);
						copy_data_to_device<double>(mu_i_mu_j, mu_ij_dev, 0);
						copy_data_to_device<double>(weights_, weights_dev, 0);
						copy_data_to_device<double>(xi_xj_, xi_xj_dev, 0);
						copy_data_to_device<int>(l1_l2_, l1_l2_dev, 0);
						copy_data_to_device<unsigned int>(offsets_, offsets_dev, 0);
						thread.x = 32;
						thread.y = 4;
						thread.z = 4;
						block.x = grid_dim[0] / 32 + ((grid_dim[0] % 32) != 0);
						block.y = grid_dim[1] / 4 + ((grid_dim[1] % 4) != 0);
						block.z = grid_dim[2] / 4 + ((grid_dim[2] % 4) != 0);
				}

		void setup_grid_info(const dim3 grid_size, const double *dr, double *density, double *density_derivatives)
				{
						if (density_dev) {
								cudaFree(density_dev);
						}

						if ((density_derivatives) && (density_derivatives_dev)) {
								cudaFree(density_derivatives_dev);
						}

						if (density_derivatives)
								cudaMalloc(&density_derivatives_dev, 3 * sizeof(double)
													 * grid_size.x
													 * grid_size.y
													 * grid_size.z);

						cudaMalloc(&density_dev, sizeof(double)
											 * grid_size.x
											 * grid_size.y
											 * grid_size.z);
				}
		void collocate(double *density, double *density_derivatives, dim3 grid_size) {
				collocate_gpu(l1_l2_dev,
											offsets_dev,
											coef_dev,
											weight_dev,
											mu_ij_dev,
											xi_xj_dev,
											l1_l2.size(),
											grid_size,
											dr,
											center_of_cube,
											density,
											density_derivatives);
		}
} gp;


/*
	l1_l2 : array containing the values of the angular momentums. They are encoded
	as l1_l2[i] = (l1 << 16) + l2. It is more than enough since l1, l2 < 20

	offsets_ : offsets_ giving the position of the initial coefficient for a given
	product of gaussian.

	coefs : array containing the coefficients of given product of gaussian

	mu_ : parameter of the gaussians.
				- mu_[2 * i] -> parameter of the first gaussian,
				- mu_[2 * i + 1] parameter of the second gaussian

	number_of_products : Number of gaussian pair treated by each block (if we use
	multiple copies of the grid to maximize the througput if it fixed by
	threadIdx.z + gridDim.z * blockDim.z)
*/

__inline__ __device__ double3 compute_cartesian_coordinates(const int center, const int pos_x, const double dr)
{
		return (pos_x - center) * dr;
}

__inline__ __device__ double gaussian_prefactor(const double3 &ra, const double3 &rb, const double mu1, const double m2)
{
		return exp(- ((ra.x - rb.x) * (ra.x - rb.x) +
									(ra.y - rb.y) * (ra.y - rb.y) +
									(ra.z - rb.z) * (ra.z - rb.z)) * m1 * m2 / (m1 + m2));
}

__inline__ __device__ double compute_polynomial_product(const int radius_int,
																												const double dr,
																												const double x,
																												const double xa,
																												const double xb,
																												const double xab,
																												const double mu_a,
																												const double mu_b,
																												const double mu_ab,
																												const int alpha1,
																												const int alpha2,
																												double &poly_derivative)
{
		/* this routine does this for each direction

			 \sum_{i >= - radius / L} ^ {i <= radius / L} (x L + x - xa) ^ alpha (x L + x - xb) ^ beta \exp\left(-mu1 (x L + x - x_ab) ^ 2\right)

		*/
		double res = 0.0;
		poly_derivative = 0.0;

		for (int i = -radius_int.x; i <= radius_int.x; i++) {
				double x1 = i * width + x - xab;
				const double prefac = exp(-mu_ab * x1 * x1);

				x1 = = i * width[0] + x - xa;
				for (int s = 0; s < alpha1; s++)
						prefac *= x1;

				double rs = alpha1 / x1 - 2 * mu_a * x1;

				x1 = i * width[0] + x - xb;
				for (int s = 0; s < alpha2; s++)
						prefac *= x1;
				res += prefac;

				// derivative along x direction. remember we have a product of two
				// gaussians centered at different points.

				rs += alpha2 / x1 - 2 * mu_b * x1;

				poly_derivative += rs * prefac;
		}

		return res;
}

__inline__ __device__ double compute_polynomial_product_derivative(const int radius_int,
																																	 const double dr,
																																	 const double x,
																																	 const double xa,
																																	 const double xb,
																																	 const double xab,
																																	 const double mu_a,
																																	 const double mu_b,
																																	 const double mu_ab,
																																	 const int alpha1,
																																	 const int alpha2)
{
		/* this routine does this for each direction

			 \sum_{i >= - radius / L} ^ {i <= radius / L} (x L + x - xa) ^ alpha (x L + x - xb) ^ beta \exp\left(-mu1 (x L + x - x_ab) ^ 2\right)

		*/
		double res = 0.0;
		for (int i = -radius_int.x; i <= radius_int.x; i++) {
				double x1 = i * width + x - xab;
				double prefac = exp(-mu_ab * x1 * x1);

				x1 = = i * width[0] + x - xa;
				for (int s = 0; s < alpha1; s++)
						prefac *= x1;

				double rs = alpha1 / x1 - 2 * mu_a * x1;

				x1 = = i * width[0] + x - xb;
				for (int s = 0; s < alpha2; s++)
						prefac *= x1;

				rs += alpha2 / x1 - 2 * mu_b * x1;
				res += rs * prefac;
		}
		return res;
}


__device__ __inline__ void compute_indice(const int n, int3 &exponent)
{
		exponent.z = (n & 0xFF0000) >> 16;
		exponent.y = (n & 0xFF00) >> 8;
		exponent.x = (n & 0xFF);
}

__derive__ __inline__ double gaussian_derivative(double x, double xa, int alpha, double beta)
{
		return (alpha / (x - xa) - 2 * beta (x - xa));
}

__global__ void collocate_gpu(const int *l1_l2,
															const double *offsets_,
															const double *__restrict__ coefs,
															const double *__restrict__ weight,
															const double *__restrict__ mu_ij,
															const double *__restrict__ pos_a_b,
															const int *number_of_products,
															const int3 grid_size,
															const double3 dr,
															const int3 center_of_cube,
															double *__restrict__ density,
															double *__restrict__ density_derivatives)
{
		// we can relax that constraint afterwards

		if(threadIdx.x + blockIdx.x * blockDim.x >= grid_size[0])
				return;
		if(threadIdx.y + blockIdx.y * blockDim.y >= grid_size[1])
				return;
		if(threadIdx.z + blockIdx.z * blockDim.z >= grid_size[2])
				return;

		// we need to change this in case of multiple copies of the grid
		const int partial_num_product = number_of_products[0];
		const double3 r = {.x = compute_cartesian_coordinates(center_of_cube[0], threadIdx.x + blockIdx.x * blockDim.x, dr[0]),
											 .y = compute_cartesian_coordinates(center_of_cube[1], threadIdx.y + blockIdx.y * blockDim.y, dr[1]),
											 .z = compute_cartesian_coordinates(center_of_cube[2], threadIdx.z + blockIdx.z * blockDim.z, dr[2])};

		// position in the grid
		for (int lk = 0; lk < partial_num_product; lk++) {
				const double weight_ = weight[lk];
				const double *coef_ = coefs + offsets_[lk];

				const int l1 = (l1_l2[lk] & 0xff00) >> 8;
				const int l2 = (l1_l2[lk] & 0xff);

				const double mu1 = mu_ij[2 * lk];
				const double mu2 = mu_ij[2 * lk + 1];
				double3 ra = {.x = pos_a_b[6 * lk],
											.y = pos_a_b[6 * lk + 1],
											.z = pos_a_b[6 * lk + 2]};

				double3 rb = {.x = pos_a_b[6 * lk + 3],
											.y = pos_a_b[6 * lk + 4],
											.z = pos_a_b[6 * lk + 5]};

				double3 rab;
				const double m_ab = mu1 * mu2 / (mu1 + mu2);


				// needs to compute the radius for the gaussian cutoff.
				const double radius = sqrt(15.0 * log (10) / mu_ab);

				const int3 radius_int = {.x = (radius / (grid_size.x * dr.x) + 1),
																 .y = (radius / (grid_size.y * dr.y) + 1),
																 .z = (radius / (grid_size.z * dr.z) + 1)};

				weight_ *= gaussian_product_prefactor(ra, rb, rab, mu1, mu2);

				// compute the polynomials. Note for the derivatives, we need
				// polynomials of degree l1 + 1, l2 + 1.

				for (int m1 = 0; m1 < length_[l1]; m1++) {
						int3 exponent_poly1;
						compute_indice(exponents[offset_[l1] + m1], exponent_poly1);
						for (int m2 = 0; m2 < length_[m2]; m2++) {
								const double w_c = weight_ * coef_[length_[m2] * m1 + m2];
								int3 exponent_poly2;
								compute_indice(exponents[offset_[l2] + m2], exponent_poly2);
								double3 poly_prod;
								double3 poly_prod_derivative;
								poly_prod.x = compute_polynomial_product(radius_int.x,
																												 dr[0],
																												 r.x,
																												 ra.x,
																												 rb.x,
																												 rab.x,
																												 mu_a,
																												 mu_b,
																												 mu_ab,
																												 exponent_poly1.x,
																												 exponent_poly2.x,
																												 poly_prod_derivative.x);

								poly_prod.y = compute_polynomial_product(radius_int.y,
																												 dr[1],
																												 r.y,
																												 ra.y,
																												 rb.y,
																												 rab.y,
																												 mu_a,
																												 mu_b,
																												 mu_ab,
																												 exponent_poly1.y,
																												 exponent_poly2.y,
																												 poly_prod_derivative.y);

								poly_prod.z = compute_polynomial_product(radius_int.z,
																												 dr[2],
																												 r.z,
																												 ra.z,
																												 rb.z,
																												 rab.z,
																												 mu_a,
																												 mu_b,
																												 mu_ab,
																												 exponent_poly1.z,
																												 exponent_poly2.z,
																												 poly_prod_derivative.z);

								dens.w += w_c * poly_prod.x * poly_prod.y * poly_prod.z;

								if (density_derivatives_) {
										dens.x += poly_prod_derivative.x * poly_prod.y * poly_prod.z;
										dens.y += poly_prod.y * poly_prod_derivative.y * poly_prod.z;
										dens.z += poly_prod.x * poly_prod.y * poly_prod_derivative.z;
								}
								/* add the code for the derivatives. note that we can also
								 * obtain the derivatives compared to the atom position the same
								 * way. up to a minus sign */
						}
				}
		}

		int Id = threadIdx.z + blockDim.z * blockIdx.z;
		Id *= blockDim.y * gridDim.y;
		Id += threadIdx.y + blockDim.y * blockIdx.y;
		Id *= blockDim.x * gridDim.x;
		Id += threadIdx.x + blockDim.x * blockIdx.x;

		density[Id] = dens.w;
		density_derivative[Id].x = dens.x;
		density_derivative[Id].y = dens.y;
		density_derivative[Id].z = dens.z;
}