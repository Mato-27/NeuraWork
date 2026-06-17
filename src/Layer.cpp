/**
* @file Layer.cpp
* @brief File containing Layer class implementation
* @author Magdi.V
* @date June 8, 2026
*/
#include "Optimizer.h"

/* Layer class constructor */
Layer::Layer(int inputSize, int outputSize) : 
    inputSize(inputSize > 0 ? inputSize : 0), 
    outputSize(outputSize > 0 ? outputSize : 0),
    X(1, inputSize), 
    dX(1, inputSize), 
    W(inputSize, outputSize), 
    dW(inputSize, outputSize),
    B(1, outputSize), 
    dB(1, outputSize), 
    Z(1, outputSize), 
    dZ(1, outputSize), 
    A(1, outputSize) {

    }

/* Executes the forward pass of the neural network Layer */
Matrix& Layer::forward(const Matrix& inputMatrix) {
    int batchSize = inputMatrix.getRows();

    // Save the input matrix into the persistent attribute for future backpropagation gradient tracking
    X.copyFrom(inputMatrix); 

    // Reassign the pre-activation Matrix dimensions dynamically to match the current batch size
    Z.reshape(inputMatrix.getRows(), B.getColumns());
    A.reshape(batchSize, B.getColumns());

    dX.reshape(batchSize, inputSize);
    dZ.reshape(batchSize, outputSize);
    
    // Compute the linear combination product (X * W) and store the result in Z
    Z.dot(X, W);

    // Perform broadcast addition of the bias vector B onto every row of the pre-activation Matrix Z
    Z.add(B);

    // Deep copy the linear results into the activation matrix A using copy-and-swap assignment
    A.copyFrom(Z);

    // Apply the Rectified Linear Unit non-linear activation in-place on the final output Matrix
    A.computeReLU();
    return A;
}

/* Weight Matrix setter with constant reference tracking */
void Layer::setW (const Matrix& w) {
    // Perform a deep copy of weight parameters into the persistent layer attribut
    W = w;
}

/* Bias Matrix setter with constant reference tracking */
void Layer::setB (const Matrix& b) {
    // Perform a deep copy of bias parameters into the persistent layer attribute
    B = b;
}

void Layer::backward(const Matrix& dL_dA, Matrix& dL_dX) {
    dZ.copyFrom(dL_dA);
    dZ.d_computeReLU(Z);
    Matrix::d_mult(X, true, dZ, false, dW);
    dZ.sum(dB);
    Matrix::d_mult(dZ, false, W, true, dL_dX);
}

Matrix& Layer::getdX() {
    return dX;
}

const Matrix& Layer::getA() {
    return A;
}

const Matrix& Layer::getW() {
    return W;
}

const Matrix& Layer::getB() {
    return B;
}

const Matrix& Layer::getdW() {
    return dW;
}

const Matrix& Layer::getdB() {
    return dB;
}

/* void Layer::update(const double alpha) {
    const double oppAlpha = -alpha;
    W.update(dW, oppAlpha);
    B.update(dB, oppAlpha);
} */

void Layer::update(Optimizer& optimizer, int layerIndex) {
    optimizer.updateLayer(layerIndex, W, B, dW, dB);
}