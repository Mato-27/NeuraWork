/**
* @file Layer.h
* @brief File containing Layer class definition
* @author Magdi.V
* @date June 8, 2026
*/

#ifndef LAYER_H
#define LAYER_H

#include "Matrix.h"

class Optimizer;

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
        /// Internal buffer storing the gradient of the loss (1*N size)
        Matrix dX; 
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
        * @details Stores the input matrix, resizes internal buffers dynamically 
        * without systematic reallocations, and computes the activation states.
        * @param inputMatrix Constant reference to the incoming activation Matrix
        * @return Reference to the internal post-activation A Matrix
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

        /**
        * @brief Executes the complete backward pass for the individual layer
        * @details Computes gradients for weights (dW) and biases (dB) using static calls,
        * then propagates the error back to the previous layer via dL_dX without reallocation.
        * @param dL_dA Constant reference to the incoming gradient from the downstream layer
        * @param dL_dX Reference to the destination matrix where the calculated input gradient is written
        */
        void backward(const Matrix& dL_dA, Matrix& dL_dX);

        /**
        * @brief Provides mutable access to the internal input gradient buffer
        * @details Used by the orchestrator to pass the gradient downstream during backpropagation
        * @return Reference to the internal dX Matrix
        */
        Matrix& getdX();

        /**
        * @brief Provides read-only access to the layer post-activation output buffer
        * @details Used to feed forward activation states into the subsequent layer of the network
        * @return Constant reference to the internal A Matrix
        */
        const Matrix& getA();

        /**
        * @brief Provides read-only access to the internal weight matrix
        * @return Constant reference to the weight Matrix of dimensions (Input Size x Output Size) 
        */
        const Matrix& getW();
        
        /**
        * @brief Provides read-only access to the internal bias vector
        * @return Constant reference to the bias Matrix of dimensions (1 x Output Size) 
        */
        const Matrix& getB();

        /**
        * @brief Provides read-only access to the internal weight gradient buffer
        * @return Constant reference to the dW Matrix accumulated during backpropagation
        */
        const Matrix& getdW();
        
        /**
        * @brief Provides read-only access to the internal bias gradient buffer
        * @return Constant reference to the dB Matrix accumulated during backpropagation
        */
        const Matrix& getdB();

        /** 
        * @brief Updates layer parameters using calculated gradients and a learning rate
        * @details Executes in-place parameter modifications on weights and biases by scaling internal gradient buffers
        * @param alpha The learning rate scalar applied to the gradient descent step 
        */
        /*void update(const double alpha);*/

        /**
        * @brief Delegates parameter optimization to an external strategy
        * @details Utilizes a double dispatch mechanism to pass internal parameter 
        * states to the running optimizer engine without exposing mutable accessors
        * @param optimizer Reference to the abstract Optimizer execution pipeline
        * @param layerIndex The geometrix position index of this layer within the network topology  
        */
        void update(Optimizer& optimizer, int layerIndex);
};

#endif