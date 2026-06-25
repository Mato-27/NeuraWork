/**
* @file SoftmaxLayer.h
* @brief File containing SoftmaxLayer class definition
* @author Magdi.V
* @date June 26, 2026
*/

#ifndef SOFTMAXLAYER_H
#define SOFTMAXLAYER_H

#include "Layer.h"

class SoftmaxLayer : public Layer {
    private:
        /// Persistent activation output matrix (BatchSize * Features) storing the recitifed states
        /// Also serves as the mathematical look-up cache for backpropagation derivatives
        Matrix A;
        /// Persistent error gradient matrix with respect to the input (BatchSize * Features) passed downstream
        Matrix dL_dX;

    public:
        /**
        * @brief Default constructor initializing an empty non-parametric activation layer
        * @details Buffers remain unallocated (0x0) until the first forward pass triggers dynamic reshaping
        */
        SoftmaxLayer();
        
        /**
        * @brief Default polymorphic destructor leveraging RAII cleanup for internal buffers
        */
        ~SoftmaxLayer() = default;

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
};

#endif