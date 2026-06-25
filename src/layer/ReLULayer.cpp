/**
* @file ReLULayer.cpp
* @brief File containing ReLULayer class implementation
* @author Magdi.V
* @date June 25, 2026
*/

#include "ReLULayer.h"
#include "../optimizer/Optimizer.h"

ReLULayer::ReLULayer() : 
    A(),
    dL_dX() {

}

const Matrix& ReLULayer::forward(const Matrix& X) {
    int rows = X.getRows();
    int cols = X.getColumns();

    // Dynamically align output matrix allocation with the incoming tensor shape
    if (A.getRows() != rows || A.getColumns() != cols) A.reshape(rows, cols);

    // Execute single-pass element-wise rectification mapping (Thresholding at 0.0)
    int totalElements = rows * cols;
    for (int i = 0; i < totalElements; i++) {
        A[i] = (X[i] > 0.0) ? X[i] : 0.0;
    }

    return A;
}

const Matrix& ReLULayer::backward(const Matrix& dL_dA) {
    int rows = dL_dA.getRows();
    int cols = dL_dA.getColumns();

    if (dL_dX.getRows() != rows || dL_dX.getColumns() != cols) dL_dX.reshape(dL_dA.getRows(), dL_dA.getColumns());

    int totalElements = rows * cols;
    for (int i = 0; i < totalElements; i++) {
        dL_dX[i] = (A[i] > 0.0) ? dL_dA[i] : 0.0;
    }

    return dL_dX;
}

void ReLULayer::update(Optimizer& optimizer, int layerIndex) {
    // Intentional no-op: Non parametrix layers contain no trainable parameters to optimize
}

const Matrix& ReLULayer::getA() const {
    return A;
}