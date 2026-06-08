/**
* @file Layer.cpp
* @brief File containing Layer class implementation
* @author Magdi.V
* @date June 8, 2026
*/
#include "Layer.h"

/* Layer class constructor */
Layer::Layer(int inputSize, int outputSize) : 
    inputSize(inputSize > 0 ? inputSize : 0), 
    outputSize(outputSize > 0 ? outputSize : 0),
    X(1, inputSize), 
    W(inputSize, outputSize), 
    B(1, outputSize), 
    Z(1, outputSize), 
    A(1, outputSize) {

    }

/* Executes the forward pass of the neural network Layer */
Matrix& Layer::forward(const Matrix& inputMatrix) {
    // Save the input matrix into the persistent attribute for future backpropagation gradient tracking
    X = inputMatrix; 

    // Reassign the pre-activation Matrix dimensions dynamically to match the current batch size
    Z = Matrix(inputMatrix.getRows(), B.getColumns());

    // Compute the linear combination product (X * W) and store the result in Z
    Z.dot(X, W);

    // Perform broadcast addition of the bias vector B onto every row of the pre-activation Matrix Z
    Z.add(B);

    // Deep copy the linear results into the activation matrix A using copy-and-swap assignment
    A = Z;

    // Apply the Rectified Linear Unit non-linear activation in-place on the final output Matrix
    A.computeReLU();
    return A;
}

/* Weight Matrix setter with constant reference tracking */
void Layer::setW (Matrix& w) {
    // Perform a deep copy of weight parameters into the persistent layer attribut
    W = w;
}

/* Bias Matrix setter with constant reference tracking */
void Layer::setB (Matrix& b) {
    // Perform a deep copy of bias parameters into the persistent layer attribute
    B = b;
}