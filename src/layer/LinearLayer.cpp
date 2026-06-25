/**
* @file LinearLayer.cpp
* @brief File containing LinearLayer class implementation
* @author Magdi.V
* @date June 25, 2026
*/

#include "LinearLayer.h"
#include "../optimizer/Optimizer.h"

LinearLayer::LinearLayer(int inputSize, int outputSize) : 
    inputSize(inputSize > 1 ? inputSize : 1), 
    outputSize(outputSize > 1 ? outputSize : 1),
    X_old(), 
    W(this->inputSize, this->outputSize), 
    dW(this->inputSize, this->outputSize),
    B(1, this->outputSize), 
    dB(1, this->outputSize), 
    A(),
    dL_dX() {

    }

/* Executes the forward pass of the neural network Layer */
const Matrix& LinearLayer::forward(const Matrix& inputMatrix) {
    int currentBatchSize = inputMatrix.getRows();

    // Dynamically adjust operational caches using lazy allocation strategy
    if (X_old.getRows() != currentBatchSize) {
        X_old.reshape(currentBatchSize, inputSize);
        A.reshape(currentBatchSize, outputSize);
    }

    // Cache historical input state and compute raw linear transformation
    X_old.copyFrom(inputMatrix);
    A.dot(inputMatrix, W);

    // Apply row-wise bias broadcasting across the current mini-batch coordinates
    for (int i = 0; i < currentBatchSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            A(i, j) = A(i, j) + B(0, j);
        }
    }

    return A;
}

const Matrix& LinearLayer::backward(const Matrix& dL_dA) {
    int currentBatchSize = dL_dA.getRows();

    // Validate and resize the input error return buffer downstream
    if (dL_dX.getRows() != currentBatchSize) dL_dX.reshape(currentBatchSize, inputSize);

    // Calculate parameter weight gradients dW = (X_old)^T * dL_dA
    Matrix::d_mult(X_old, true, dL_dA, false, dW);

    // Reset bias gradient accumulation memory to prevent cross-batch pollution
    dB.fill(0.0);

    // Execute column-wise mathematical reduction to accumulate total bias error
    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < currentBatchSize; j++) {
            dB(0, i) = dB(0, i) + dL_dA(j, i);
        }
    }

    // Compute backpropagated error for preceding architecture layers dL_dX = dL_dA * W^T
    Matrix::d_mult(dL_dA, false, W, true, dL_dX);

    return dL_dX;
}

void LinearLayer::update(Optimizer& optimizer, int layerIndex) {
    optimizer.updateLayer(layerIndex, W, B, dW, dB);
}

const Matrix& LinearLayer::getA() const {
    return A;
}

const Matrix& LinearLayer::getW() const {
    return W;
}

const Matrix& LinearLayer::getdW() const {
    return dW;
}

const Matrix& LinearLayer::getB() const {
    return B;
}

const Matrix& LinearLayer::getdB() const {
    return dB;
}

/* Weight Matrix setter with constant reference tracking */
void LinearLayer::setW (const Matrix& w) {
    // Perform a deep copy of weight parameters into the persistent layer attribut
    W = w;
}

/* Bias Matrix setter with constant reference tracking */
void LinearLayer::setB (const Matrix& b) {
    // Perform a deep copy of bias parameters into the persistent layer attribute
    B = b;
}