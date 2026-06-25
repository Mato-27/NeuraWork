/**
* @file MSELoss.cpp
* @brief File containing MSELoss class implementation
* @author Magdi.V
* @date June 12, 2026
*/

#include "MSELoss.h"

/* Computes the global scalar loss value between predictions and targets */
double MSELoss::forward(const Matrix& A, const Matrix& Y) const {
    double L = 0.0;
    int M = A.getColumns();
    int B = A.getRows();
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < M; j++) {
            L += std::pow(A(i,j) - Y(i,j), 2);
        }
    }
    return L / (B * M);
}


/* Computes the gradient of the loss function with respect to the input activations */
void MSELoss::backward(const Matrix& A, const Matrix& Y, Matrix& dL_dA) {
    int M = A.getColumns();
    int B = A.getRows();
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < M; j++) {
            dL_dA(i, j) = 2.0/(B*M) * (A(i, j) - Y(i, j));
        }
    }
}