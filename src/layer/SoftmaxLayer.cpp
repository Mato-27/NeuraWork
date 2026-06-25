/**
* @file SoftmaxLayer.cpp
* @brief File containing SoftmaxLayer class implementation
* @author Magdi.V
* @date June 26, 2026
*/

#include "SoftmaxLayer.h"
#include "../optimizer/Optimizer.h"

SoftmaxLayer::SoftmaxLayer() : 
    A(),
    dL_dX() {

}

const Matrix& SoftmaxLayer::forward(const Matrix& X) {
    int rows = X.getRows();
    int cols = X.getColumns();

    // Dynamically align output matrix allocation with the incoming tensor shape
    if (A.getRows() != rows || A.getColumns() != cols) A.reshape(rows, cols);

    // Loop through each sample in the mini-batch
    for (int i = 0; i < rows; i++) {
        int rowOffset = i * cols;

        // --- Pass 1: Find maximum value for numerical stability ---
        // Initialize max_val with the very first element of the current row
        double maxVal = X[rowOffset + 0];

        // Scan the rest of the columns for the current row to find the maximum affine scalar
        for (int j = 1; j < cols; j++) {
            if (X[rowOffset + j] > maxVal) maxVal = X[rowOffset + j];
        }

        // --- Pass 2: Compute shifted exponentials and cumulative sum ---
        double sumExp = 0.0;
        for (int j = 0; j < cols; j++) {
            int index = rowOffset + j;
            // Write shifted exponential directly to output buffer to save memory bandwidth
            A[index] = std::exp(X[index] - maxVal);
            sumExp += A[index];
        }

        // --- Pass 3: Normalize components into probability distributions ---
        for (int j = 0; j < cols; j++) A[rowOffset + j] /= sumExp;
    }

    return A;
}

const Matrix& SoftmaxLayer::backward(const Matrix& dL_dA) {
    int rows = dL_dA.getRows();
    int cols = dL_dA.getColumns();

    if (dL_dX.getRows() != rows || dL_dX.getColumns() != cols) dL_dX.reshape(dL_dA.getRows(), dL_dA.getColumns());

    // Process each sample in the mini-batch independently (Row-wise gradient reduction)
    for (int i = 0; i < rows; i++) {
        int rowOffset = i * cols;

        // --- Pass 1: Compute the internal dot product scalar sum(dL_dA * A) ---
        double sumGradDotAct = 0.0;
        for (int j = 0; j < cols; j++) {
            int index = rowOffset + j;
            sumGradDotAct += dL_dA[index] * A[index];
        }

        // --- Pass 2: Compute localized input error derivatives via simplified chain rule ---
        for (int j = 0; j < cols; j++) {
            int index = rowOffset + j;
            dL_dX[index] = A[index] * (dL_dA[index] - sumGradDotAct);
        }
    }

    return dL_dX;
}

void SoftmaxLayer::update(Optimizer& optimizer, int layerIndex) {
    // Intentional no-op: Non parametrix layers contain no trainable parameters to optimize
}

const Matrix& SoftmaxLayer::getA() const {
    return A;
}