//
// Created by junior on 19-5-28.
//

#ifndef CPPML_MATRIX_H
#define CPPML_MATRIX_H

#include <cstdio>
#include "mkl.h"

namespace CppML {
    const size_t alignment = 64;

    // 参考: https://www.zhihu.com/question/265023816 中第三个回答
    void error_handle(size_t m, size_t n) {
        fprintf(stderr, "\n ERROR: Can't allocate memory for matrices(%zu x %zu). Aborting... \n", m, n);
        fflush(stderr);
        abort(); // 出现malloc错误直接崩溃,不做异常处理,打印错误信息.
    }

    template<typename DataType, size_t m, size_t n>
    class Matrix {
        friend class Algorithm;

    private:
        DataType *data;

    private:
        // 下面这个接口过于危险,所以对其private,仅在内部使用(按照自己的内存分配处理)
        explicit Matrix(DataType *data) : data(data) {
            static_assert(m >= 1 && n >= 1);
        }

    public:
        Matrix() {
            static_assert(m >= 1 && n >= 1);
            data = reinterpret_cast<DataType *>(mkl_malloc(m * n * sizeof(DataType), alignment));
            if (data == nullptr) {
                error_handle(m, n);
            }
        }

        // TODO: 添加边界检查(在debug模式下)
        // 这里不做空检查,因为data必定非空
        void set(size_t i, size_t j, const DataType &d) {
            data[j + i * n] = d;
        }

        // TODO: 添加边界检查(在debug模式下)
        // 这里不做空检查,因为data必定非空
        DataType at(size_t i, size_t j) {
            return data[j + i * n];
        }

        ~Matrix() {
            if (data != nullptr) {
                mkl_free(data);
                data = nullptr;
            }
        }
    };

    class Algorithm {
    public:
        template<typename DataType__, size_t m_, size_t k_, size_t n_>
        static Matrix<DataType__, m_, n_> mm(const Matrix<DataType__, m_, k_> &matrix_a,
                                             const Matrix<DataType__, k_, n_> &matrix_b,
                                             double alpha = 1.0, double beta = 0.0) {
            auto result = reinterpret_cast<DataType__ *>(mkl_malloc(m_ * n_ * sizeof(DataType__), alignment));
            if (result == nullptr) {
                error_handle(m_, n_);
            }
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m_, n_, k_, alpha, matrix_a.data, k_, matrix_b.data, n_, beta, result, n_);
            return Matrix<DataType__, m_, n_>(result);
        }
    };

}
#endif //CPPML_MATRIX_H
