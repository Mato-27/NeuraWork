/**
* @file Layer.h
* @brief File containing Layer class definition
* @author Magdi.V
* @date June 8, 2026
*/

#ifndef LAYER_H
#define LAYER_H

#include "Matrix.h"

/**
* @class Layer
* @brief This class allows the use of Layers
*/
class Layer {
    private:
        /// Number of input neurons (N)
        int inputSize; 
        /// Number of output neurons (M)
        int outputSize;
        /// Saved input (1*N size)
        Matrix X; 
        /// Weights (N*M size)
        Matrix W; 
        /// dWeights (N*M size)
        Matrix dW; 
        /// Bias (1*M size)
        Matrix B; 
        /// dBias (1*M size)
        Matrix dB; 
        /// Pre-activation (1*M size) Z = X . W + B
        Matrix Z; 
        /// dPre-activation (Batch Size * Output Neurons)
        Matrix dZ; 
        /// Output Matrix (1*M size) A = Z.computeReLU()
        Matrix A; 

    public:
        /**
        * @brief Layer class constructor
        * @param inputSize Number of input neurons
        * @param outputSize Number of output neurons
        */
        Layer(int inputSize = 0, int outputSize = 0);

        /**
        * @brief Executes the forward pass of the neural network Layer
        * @details Stores the input Matrix for backpropagation, reassigns the pre-activation
        * matrix dimensions to fit the batch size, computes the linear transformation
        * (X * W + B) without heap allocation overhead, and applies the activation function
        * @param inputMatrix Input data matrix of dimensions (Batch Size x Input Size)
        * @return Reference to the activated output Matrix A of dimensions (Batch Size x Output Size)
        */
        Matrix& forward(const Matrix& inputMatrix);

        /**
        * @brief Configures the internal weight matrix with constant deep-copy validation
        * @param w Constant reference to the weight Matrix source of dimensions (Input Size x Output Size)
        */
        void setW (const Matrix& w);

        /**
        * @brief Configures the internal bias vector with constant deep-copy validation
        * @param b Constant reference to the bias vector source of dimensions (1 x Output Size)
        */
        void setB (const Matrix& b);

        void backward(const Matrix& dL_dA, Matrix& dL_dX);
};

#endif