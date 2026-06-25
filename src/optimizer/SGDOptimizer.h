/**
* @file SGDOptimizer.h
* @brief File containing SGDOptimizer class definition
* @author Magdi.V
* @date June 14, 2026
*/

#ifndef SGDOPTIMIZER_H
#define SGDOPTIMIZER_H

#include "Optimizer.h"

/**
* @class SGDOptimizer
* @brief Concrete implementation of the Stochastic Gradient Descent (SGD) parameter update strategy
*/
class SGDOptimizer : public Optimizer {
    private:
        /// Static learning rate multiplier applied directly to the directional gradients
        double alpha;

    public:
        /**
        * @brief Constructs an SGD optimizer with a dedicated learning rate
        * @param al The learning rate (alpha) value applied to parameter updates 
        */
        SGDOptimizer(const double al);

        /**
        * @brief Iterates through the model layers to trigger their internal parameter update states
        * @details Loop through each polymorphic layer sequentially to apply the gradient descent step in-place
        * @param layers Reference to the dynamic container holding polymorphic network layers
        */
        void update(std::vector<std::unique_ptr<Layer>>& layers) override;

        /**
        * @brief Executes a standard linear gradient descent step in-place
        * @details Ignores historical tracking vectors to perform a fast contiguous AXPY accumulation 
        * directly onto the layer parameters scaled by a negative static learning rate factor
        * @param layerIndex The geometric position index of the targeted layer
        * @param W Mutable reference to the weight Matrix layout
        * @param B Mutable reference to the bias vector Matrix layout
        * @param dW Constant reference to the accumulated weight gradient Matrix
        * @param dB Constant reference to the accumulated bias gradient Matrix
        */
        void updateLayer(int layerIndex, Matrix& W, Matrix& B, const Matrix& dW, const Matrix& dB) override;
};

#endif