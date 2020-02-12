/*
 * The logic on gpu is identical to the CPU version, with one twist, i.e. we use
 * batched matrix-matrix multiplications. It means that we need to compute all
 * coefficients for each pair of gaussian and we need to take all pairs of
 * gaussian of same l1 + l2.
 */

template <typename T> void collocate_core_gpu(cublasHandle_t handle,
																							const int lmax,
																							const int *grid_size,
																							const std::vector<T> &alpha,
																							const mdarray<T, 4, CblasRowMajor> &co,
																							const mdarray<T, 4, CblasRowMajor> &poly_,
																							const mdarray<T, 3, CblasRowMajor> &C_,
																							const mdarray<T, 4, CblasRowMajor> &xyz_alpha_beta,
																							const mdarray<T, 4, CblasRowMajor> &V)
{
		std::vector<T*> A, B, C;

		std::vector<T> beta;

		A.clear();
		B.clear();
		C.clear();

		A.resize(co.size(0));
		B.resize(co.size(0));
		C.resize(co.size(0));

		/*
		 * the matrices are stored in row major order but cuda only accept column
		 * major, so I need to invert the order of the matrices everywhere.
		 */

		for (int a1 = 0; a1 < co.size(1); a1++) {
				for (int gt = 0; gt < co.size(); gt++) {
						// A is P_y[beta][j]
						A[gt] = poly_.template at<GPU>(gt, 1, 0, 0);
						// B is Coef[alpha][gamma][beta] alpha is fixed
						B[gt] = co.template at<GPU>(gt, 0, 0, 0);
						// C is Tmp_alpha[gamma][j] = sum_beta Coef[alpha][gamma][beta] P_y[beta][j]
						C[gt] = C_.template at<GPU>(gt, 0, 0);
				}

				/*
				 * fortran sees all tables as A[j][beta], B[beta][gamma][alpha] and
				 * C[j][gamma][alpha]. Its means that
				 *
				 * \sum C[alpha][gamma][beta] A[beta][j] in C becomes
				 *
				 * A[j][beta]  C[beta][gamma][alpha]
				 *
				 * I have to inverse the matrices order in the gemm
				 */

				cublasDgemmBatched(handle,
													 CUBLAS_OP_N,
													 CUBLAS_OP_N,
													 grid_size[1], /* y first */
													 lmax,
													 lmax,
													 &alpha[0],
													 &A[0],
													 poly_.ld(),
													 &B[0],
													 lmax,
													 &beta[0],
													 &C[0],
													 C.ld(),
													 co.size(0));

				for (int gt = 0; gt < co.size(); gt++) {
						// A is in row major C[gamma][j]
						A[gt] = C_.template at<GPU>(gt, 0, 0);
						// B is in row major B[gamma][k]
						B[gt] = poly_.template at<GPU>(gt, 0, 0, 0);

						// C is row major and is xyz[alpha][k][j]
						C[gt] = xyz_alpha_beta.template at<GPU>(gt, a1, 0, 0);
				}


				cublasDgemmBatched(handle,
													 CUBLAS_OP_T,
													 CUBLAS_OP_N,
													 grid_size[1],
													 grid_size[0],
													 lmax,
													 &alpha[0],
													 &A[0],
													 C_.ld(),
													 &B[0],
													 lmax,
													 &beta[0],
													 &C[0],
													 xyz_alpha_beta.ld(),
													 co.size());
		}

		for (int gt = 0; gt < co.size(); gt++) {
				// xyz[alpha][k][j]
				B[gt] = xyz_alpha_beta.template at<GPU>(gt, 0, 0, 0);
				// P_x [alpha][i]
				A[gt] = poly_.template at<GPU>(gt, 2, 0, 0);

				// xyz[k][j][alpha] P_x[alpha][i] = V[k][j][i]
				C[gt] = V.template at<GPU>(gt, 0, 0, 0);
		}

		cublasDgemmBatched(handle,
											 CUBLAS_OP_N,
											 CUBLAS_OP_T,
											 grid_size[2],
											 grid_size[0] * grid_size[1],
											 co.size(1),
											 &alpha[0],
											 &A[0],
											 poly_.ld(),
											 &xyz_alpha_beta.ld(),
											 &beta[0],
											 &C[0],
											 V.ld(),
											 co.size());
}
