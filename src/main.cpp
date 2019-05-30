#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2

#include <algorithm>
#include "../include/CppML.h"

#include <iostream>
#include "../third_party/eigen-git-mirror/Eigen/Dense"
#include <ctime>
#include <chrono>

void test_eigen() {
    using namespace Eigen;
    using namespace std;
    using namespace chrono;

    MatrixXd m1 = MatrixXd::Random(1000, 1000);
    MatrixXd m2 = MatrixXd::Random(1000, 1000);

    auto start = dsecnd();

    MatrixXd p = m1 * m2;

    std::cout << "eigen time:" << (dsecnd() - start) * 1000 << "ms" << std::endl;
}

void test_mkl() {
    auto max_threads = mkl_get_max_threads();
    mkl_set_num_threads(max_threads);

    int i, j, data;

    constexpr int m = 1000, k = 1000, n = 1000;

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

    auto C = Algorithm::mm(A, B);
}

int main() {
    for (int i = 0; i < 20; i++) {
        test_mkl();
        test_eigen();
    }
}