/**
* @file Sequential.h
* @brief File containing Sequential class definition
* @author Magdi.V
* @date June 14, 2026
*/

#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "./optimizer/SGDOptimizer.h"
#include "./optimizer/AdamOptimizer.h"

/**
* @class Sequential
* @brief This class orchestrates and chain neural network layers into a linear computational graph
*/
class Sequential {
        /// Internal dynamic container holding pointers to the network's layers
        std::vector<std::unique_ptr<Layer>> array;
    public:
        /**
        * @brief Appends a polymorphic layer to the end of the architecture
        * @details Transfers ownership of the layer instance into the internal container,
        * expanding the network topolgy before the execution phase
        * @param layer A unique pointer to the Layer instance to be appended
        */
        void add(std::unique_ptr<Layer> layer);

        /**
        * @brief Propagates the input matrix forward through the entire network topology
        * @details Sequentially pipes the activation output of layer L into the input space 
        * of layer L+1. This operation guarantees zero heap allocation during execution
        * @param input Constant reference to the initial feature Matrix for the mini-batch
        * @return Constant reference to the activation Matrix of the final layer (predictions)
        */
        const Matrix& forward(const Matrix& input);

        /**
        * @brief Propagates the error gradient backward from the loss output to the first layer
        * @details Iterates through the layer graph in reverse order, passing the calculated 
        * input gradient of Layer L as the incoming output gradient for layer L-1
        * @param dL_dA Constant reference to the gradient of the loss with respect to the network outputs
        */
       void backward(const Matrix& dL_dA); 

       /**
       * @brief Provides read-only access to the weight Matrix of the initial network layer
       * @return Constant reference to the dW Matrix of the first layer
       */
       const Matrix& getFirstLayerW();

       /**
       * @brief Provides read-only access to the bias vector of the initial network layer
       * @return Constant reference to the dB Matrix of the first layer
       */
       const Matrix& getFirstLayerB();

       /**
       * @brief Provides read-only access to the weight gradient buffer of the initial network layer
       * @return Constant reference to the dW Matrix of the first layer
       */
       const Matrix& getFirstLayerdW();

       /**
       * @brief Provides read-only access to the bias gradient buffer of the initial network layer
       * @return Constant reference to the dB Matrix of the first layer
       */
       const Matrix& getFirstLayerdB();

       /**
       * @brief Delegates parameter optimization to an external polymorphic strategy
       * @details Passes the internal layers container directly to the optimizer instance to execute in-place parameter adjustments
       * @param optimizer Reference to the abstract Optimizer engine execution pipeline
       */
       void update(Optimizer& optimizer);
};

#endif