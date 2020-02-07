#include <cuda.h>

void __device__ shift(double x1, double x2, double L)
{
    // L : size the of the block along the direction
    // x1, x2 the coordinates
    // 1) compute x1 - x2.
    // 2) if -L/2 < x1 - x2 < L/2 then do not shift the potential box
}

__inline__ __device__ double compute_polynomial(double r, double ra, const int alpha)
{
    double sum = 1.0;

    for (int s = 1; s <= alpha; s++) {
        sum *= (r - ra);
    }

    return sum;
}

__global__ integrate(const double *__restrict__ pot,
                     const int3 grid_dim,
                     const double drx,
                     const double dry,
                     const double drz,
                     const double *__restrict__ ra,
                     const double *__restrict__ rb,
                     const int *__restrict__ alpha_beta,
                     const double *__restrict__ exponents,
                     const double *__restrict__ coefs,
                     const double *__restrict__ overlaps)
{
    const dim3 Id = {threadIdx.x + blockIdx.x * blockDim.x,
                     threadIdx.y + threadIdx.y * blockDim.y,
                     threadIdx.z + threadIdx.z * blockDim.z};

    const int Idx = (Id.z + Id.y * gridDim.y) * gridDim.x + Id.x;

    __shared__ double *pot_block = (double *)array;
    const double alpha1 = exponents[2 * Idx];
    const double alpha2 = exponents[2 * Idx + 1];

    const double rax = ra[3 * Idx];
    const double ray = ra[3 * Idx + 1];
    const double raz = ra[3 * Idx + 2];

    const double rbx = rb[3 * Idx];
    const double rby = rb[3 * Idx + 1];
    const double rbz = rb[3 * Idx + 2];

    // need to compute p = (a x1 + b x2)/ (a + b), with pcb
    const double mu_mean = alpha1 + alpha2;
    const double mu_geo = alpha1 * alpha2 / mu_mean;

    const double radius = 7.0 / mu_mean;

    int3 rd_min;
    int3 rd_max;

    // the reference point is at grid_dim.x / 2.
    rd_min.x = (rab.x - radius) / drx;
    // the indice is compared to grid_dim.x / 2. Now I want to know how many
    // images I need to go on the left.


    // I add grid_dim.x / 2 so I have the indice compared to the index 0 of the table
    rd_min.x += grid_dim.x / 2;
    rd_min.x = (rd_mix.x < 0) * (rd_min.x / grid_dim.x + 1) + (rd_mix.x >= 0) * rd_mix.x;

    const double kappa_ab = exp(- mu_geo * scal);

    // I integrate over a cube not a sphere.
    // Integration over a sphere would lead to too many diverging threads.

    for (int z = 0; z < grid_dim.z; z++) {
        for (int y = 0; y < grid_dim.y; y++) {
            load_to_shared_memory(pot_block, pot + (z * grid_dim.y + y) * grid_dim.x, grid_dim.x);
            __syncthreads();

            // the potential is centered around 0 so the coordinate (Lx/2, Ly/2, Lz/2) = (0,0,0)
            double pre_z = 0;

            // since we potentially sum over multiple images of the potential, I compute the weight
            for (int rdz = rd_min.z; rdz <= rd_max.z; rdz++) {
                double z1 = (z - grid_dim.z / 2) * drz + rdz * period.z;
                pre_z += compute_polynomial(z1, raz, alpha) *
                    compute_polynomial(z1, rbz, beta) *
                    exp(- (z1 - rabz) * (z1 - rabz) * mu_mean);
            }

            double pre_y = 0.0;

            for (int rdy = - rd_min.y; rdy <= rd_max.y; rdy++) {
                double y1 = (y - grid_dim.y / 2) * dry + rdy * period.y;
                pre_y += compute_polynomial(y1, ray, alpha) *
                    compute_polynomial(y1, rby, beta) *
                    exp(- (y1 - raby) * (y1 - raby) * mu_mean);
            }

            for (int x = 0; x < grid_dim.x; x++) {
                double pre_factor_x = 0.0;
                for (int rdx = - rd_min.x; rdx <= rd_max.x; rdx++) {
                    double x1 = (x - grid_dim.x / 2) * drx + rdx * period.x;
                    pre_factor_x += compute_polynomial(x1, rax, alpha) *
                        compute_polynomial(x1, rbx, beta) *
                        exp(- (x1 - rabx) * (x1 - rabx) * mu_mean);
                }
                sum += pre_factor_x * pot_block[x] * pre_z * pre_y;
            }
        }
    }

    overlaps[Idx] = sum * kappa_ab * coefs;
}
