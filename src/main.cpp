#include <algorithm>
#include "../include/CppML.h"

int main() {
    // std::cout << sizeof(MKL_INT) << "\n";

    int i, j, data;

    printf("\n This example computes real matrix C=alpha*A*B+beta*C using \n"
           " Intel(R) MKL function dgemm, where A, B, and  C are matrices and \n"
           " alpha and beta are double precision scalars\n\n");

    constexpr int m = 5000, k = 5000, n = 5000;

    printf(" Initializing data for matrix multiplication C=A*B for matrix \n"
           " A(%ix%i) and matrix B(%ix%i)\n\n", m, k, k, n);

    printf(" Intializing matrix data \n\n");

    using namespace CppML;
    Matrix<double, m, k> A;
    Matrix<double, k, n> B;
    data = 0;
    for (i = 0; i < m; i++) {
        for (j = 0; j < k; j++) {
            A.set(i, j, (data++ + 1));
        }
    }
    data = 0;
    for (i = 0; i < k; i++) {
        for (j = 0; j < n; j++) {
            B.set(i, j, (-data++ - 1));
        }
    }

    printf(" Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n\n");
    // m x k  and k x n
    auto C = Algorithm::mm(A, B);
    printf("\n Computations completed.\n\n");

    printf(" Top left corner of matrix A: \n");
    for (i = 0; i < std::min(m, 6); i++) {
        for (j = 0; j < std::min(k, 6); j++) {
            printf("%12.0f", A.at(i, j));
        }
        printf("\n");
    }

    printf("\n Top left corner of matrix B: \n");
    for (i = 0; i < std::min(k, 6); i++) {
        for (j = 0; j < std::min(n, 6); j++) {
            printf("%12.0f", B.at(i, j));
        }
        printf("\n");
    }

    printf("\n Top left corner of matrix C: \n");
    for (i = 0; i < std::min(m, 6); i++) {
        for (j = 0; j < std::min(n, 6); j++) {
            printf("%12.5G", C.at(i, j));
        }
        printf("\n");
    }

    printf(" Example completed. \n\n");
    return 0;
}