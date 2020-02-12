#ifndef COLLOCATE_HPP
#define COLLOCATE_HPP

void template<typename T> void collocate_core(const int *length_,
																							const mdarray<T, 3, CblasRowMajor> &co,
																							const mdarray<T, 3, CblasRowMajor> &p_alpha_beta_reduced_,
																							mdarray<T, 3, CblasRowMajor> &Vtmp);


#endif
