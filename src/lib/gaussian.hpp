template <typename T> class gaussian {
		//

		int batch_size_{1000};

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
		mdarray<T, 3, CblasRowMajor> coefs_;
		mdarray<T, 3, CblasRowMajor> transformed_coefs_;

		// contain the polynomials computed along x,y,z with periodic boundaries
		// conditions
		mdarray<T, 3, CblasRowMajor> p_alpha_folded_;

		// polynomials computed from -radius..radius
		mdarray<T, 3, CblasRowMajor> p_alpha_;

		// non vanishing elements of the polynomials (only the position)
		mdarray<int, 2, CblasRowMajor> yz_non_vanishing_elements_;

		gaussian(const int grid_size[3], // local size of the grid
						 const int lower_bound[3], // coordinates where the local grid should be located in the distributed grid
						 const int upper_bound[3], //
						 const int period[3], // size of the grid and period (if PBC apply)
						 const T dr[3], // linear spacing in the lattice basis
						 const T *basis, // displacement vectors of the lattice
						 const int lmax, // Maximum angular momentum
						 const bool ortho=true, // the grid is orthorombic or not
						 const bool GPU_, // do the computation on GPU
						 const int device,
						 const int batch_size=1000) // device_id
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

						p_alpha_folded_ = mdarray<T, 3, CblasRowMajor>(3, lmax_ + 1, std::max(std::max(period[0],
																																													 period[1]),
																																									period[2]));
						p_alpha_ = mdarray<T, 3, CblasRowMajor>(3,
																										lmax_ + 1,
																										128);

						p_alpha_reduced_ = mdarray<T, 3, CblasRowMajor>(3,
																														lmax_ + 1,
																														std::max(std::max(grid_size[0],
																																							grid_size[1]),
																																		 grid_size[2]));
// p_alpha_ depends on external parameters we do not know to begin
						// with
				}

		~gaussian()
				{
						p_alpha_folded_.clear();
						p_alpha_reduced.clear();
						Vtmp.clear();
						Exp.clear();
				}

		void add_gaussian(const T *ra, const T *rb, const int *lmax, const int *lmin, const T *coefs) {
				if (number_of_gaussian_ > batch_size_) {
						printf("You should call the compute method\n");
						return;
				}

				memcpy()
				number_of_gaussians_++;


		}
};
