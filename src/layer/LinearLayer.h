/**
* @file LinearLayer.h
* @brief File containing LinearLayer class definition
* @author Magdi.V
* @date June 21, 2026
*/

#ifndef LINEARLAYER_H
#define LINEARLAYER_H

#include "Layer.h"

/**
 * @class LinearLayer
 * @brief Fully connected affine transformation component
 * @details Executes contiguous matrix multiplication and bias reduction loops, 
 * maintaining local historical tensors for weights and structural gradients
 */
class LinearLayer : public Layer {
    private:
        /// Number of input neurons (N)
        int inputSize; 
        /// Number of output neurons (M)
        int outputSize;
        /// Saved input (1*N size)
        Matrix X_old; 
        /// Weights (N*M size)
        Matrix W; 
        /// dWeights (N*M size)
        Matrix dW; 
        /// Bias (1*M size)
        Matrix B; 
        /// dBias (1*M size)
        Matrix dB; 
        /// Output Matrix (1*M size)
        Matrix A; 
        /// Gradient buffer (BatchSize * N) passed to the preceding layer
        Matrix dL_dX;

    public:
        /**
        * @brief Parameterized constructor initializing dimensions and allocating parameter tensors
        * @details Allocates memory weights and biases dynamically based on features specifications 
        * and triggers the normal randomization sequence for weights layers
        * @param inputSize Total number of input features (N)
        * @param outputSize Total number of output neurons (M)
        */
        LinearLayer(int inputSize, int outputSize);

        
        /**
        * @brief LinearLayer class destructor
        */
        ~LinearLayer() = default;

        /**
        * @brief Executes the forward propagation pass through the component
        * @param X Constant reference to the input Matrix
        * @return Constant reference to the computed activation Matrix
        */
        const Matrix& forward(const Matrix& X) override;
        
        /**
        * @brief Propagates the structural error gradient backward through the component
        * @param dL_dA Constant reference to the incoming gradient Matrix from the subsequent layer
        * @return Constant reference to the calculated input gradient Matrix (dL_dX)
        */
        const Matrix& backward(const Matrix& dL_dA) override;

        /**
        * @brief Delegates parameter optimization via double dispatch
        * @param optimizer Reference to the running Optimizer strategy
        * @param layerIndex The geometric position index within the network topology
        */
        void update(Optimizer& optimizer, int layerIndex) override;

        /**
        * @brief Provides read-only access to the layer post-activation output buffer
        * @details Used to feed forward activation states into the subsequent layer of the network
        * @return Constant reference to the internal A Matrix
        */
        const Matrix& getA() const override;

        /**
        * @brief Provides read-only access to the internal weight matrix
        * @return Constant reference to the weight Matrix of dimensions (Input Size x Output Size) 
        */
        const Matrix& getW() const;
    
        /**
        * @brief Provides read-only access to the internal bias vector
        * @return Constant reference to the bias Matrix of dimensions (1 x Output Size) 
        */
        const Matrix& getB() const;

        /**
        * @brief Provides read-only access to the internal weight gradient buffer
        * @return Constant reference to the dW Matrix accumulated during backpropagation
        */
        const Matrix& getdW() const;
        
        /**
        * @brief Provides read-only access to the internal bias gradient buffer
        * @return Constant reference to the dB Matrix accumulated during backpropagation
        */
        const Matrix& getdB() const;

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
};

#endif