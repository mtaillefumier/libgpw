template <typename T> class mat {
    int row_{0};
    int col_{0};
    int leading_dimension_{0};
    size_t raw_size_{0};
    T *raw_ptr{nullptr};
    bool internal_alloc_{false};

public:
    mat(const int n, const int m)
        {
            row_ = n;
            col_ = m;
            const size_t raw_size_{0};
            // starting point of each row is aligned to multiple of 8
            leading_dimension_ = ((col_ / 8) + 1) * 8;
            raw_size_ = leading_dimension_ * n;
            posix_memalign(&raw_ptr, 64, sizeof(T) * raw_size_);
            internal_alloc_ = true;
        }

    mat(T *data, const int n, const int m, const int lda)
        {
            row_ = n;
            col_ = m;
            // starting point of each row is aligned to multiple of 8
            leading_dimension_ = lda;
            raw_ptr = data;
        }

    ~mat()
        {
            if(internal_alloc_)
                free(raw_ptr);
        }

    T *data() {
        return raw_ptr;
    }

    const int ld() {
        return leading_dimension_;
    }
    const int row() {
        return row_;
    }

    const int col() {
        return col_;
    }

    const int row() const {
        return row_;
    }

    const int col() const {
        return col_;
    }

    void resize(const int n, const int m) {
        const int lda = ((m / 8) + 1) * 8;
        row_ = n;
        col_ = m;
        leading_dimension_ = lda;
        if (raw_size_ < (lda * n))
        {
            realloc(raw_ptr, sizeof(T) * lda * n);
            raw_size_ = lda * n;
        }
    }
};

void gemm(char op1, char op2, const double alpha, mat<double> &a, mat<double> &b, const double beta, mat<double> &c)
{
    if ((op1 = 'N') && (op2 = 'N'))
        cblas_dgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    a.row(),
                    b.col(),
                    b.row(),
                    alpha,
                    a.data(),
                    a.ld(),
                    b.data(),
                    b.ld(),
                    beta,
                    c.data(),
                    c.ld());

    if ((op1 = 'T') && (op2 = 'N'))
        cblas_dgemm(CblasRowMajor,
                    CblasTrans,
                    CblasNoTrans,
                    a.col(),
                    b.col(),
                    b.row(),
                    alpha,
                    a.data(),
                    a.ld(),
                    b.data(),
                    b.ld(),
                    beta,
                    c.data(),
                    c.ld());

    if ((op1 = 'N') && (op2 = 'T'))
        cblas_dgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    a.row(),
                    b.row(),
                    b.col(),
                    alpha,
                    a.data(),
                    a.ld(),
                    b.data(),
                    b.ld(),
                    beta,
                    c.data(),
                    c.ld());

    if ((op1 = 'T') && (op2 = 'T'))
        cblas_dgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    a.col(),
                    b.row(),
                    b.col(),
                    alpha,
                    a.data(),
                    a.ld(),
                    b.data(),
                    b.ld(),
                    beta,
                    c.data(),
                    c.ld());
}
