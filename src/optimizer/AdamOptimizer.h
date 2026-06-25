/**
* @file AdamOptimizer.h
* @brief File containing AdamOptimizer class definition
* @author Magdi.V
* @date June 15, 2026
*/

#ifndef ADAMOPTIMIZER_H
#define ADAMOPTIMIZER_H

#include "Optimizer.h"

/**
* @class AdamOptimizer
* @brief Concrete implementation of the Adaptive Moment Estimation (Adam) algorithm
*/
class AdamOptimizer : public Optimizer {
    private:
        /// Dynamic container storing the first-moment vector (exponential moving average of gradients) for each layer
        std::vector<Matrix> mW;
        /// Dynamic container storing the second-moment vector (exponential moving average of squared gradients) for each layer
        std::vector<Matrix> vW;
        /// Dynamic container storing the first-moment vector for the bias parameters of each layers
        std::vector<Matrix> mB;
        /// Dynamic container storing the second-moment vector for the bias parameters of each layers
        std::vector<Matrix> vB;
        /// Global learning rate scalar determining the base step size
        double alpha;
        /// Exponential decay rate for the first-moment estimates
        double b1;
        /// Exponential decay rate for the second-moment estimates
        double b2;
        /// Numerical stability factor preventing division by zero during parameter scaling
        double epsilon;
        /// Global time-step counter tracking the total number of optimization mini-batches processed
        int t;

    public:
        /**
        * @brief Parameterized constructor initializing the adaptive moment estimation configuration
        * @details Binds structural hyper-parameters and initializes the global simulation time-step to zero
        * @param al Global base learning rate scalar (alpha)
        * @param b1 Exponential decay rate for the first-moment estimates (default: 0.9)
        * @param b2 Exponential decay rate for the second-moment estimates (default: 0.999)
        * @param epsilon Numerical stability smoothing factor preventing singular divisions (default: 1e-8)
        */
        AdamOptimizer(const double al, const double b1 = 0.9, const double b2 = 0.999, const double epsilon = 1e-8);

        /**
        * @brief Iterates through the model layers to trigger their internal parameter update states
        * @details Loop through each polymorphic layer sequentially to apply the gradient descent step in-place
        * @param layers Reference to the dynamic container holding polymorphic network layers
        */
        void update(std::vector<std::unique_ptr<Layer>>& layers) override;

        /**
        * @brief Computes in-place parameter mutations utilizing adaptive moment estimation
        * @details Analyzes layer geometry, instantiates historical state buffers on the first iteration, 
        * and applies element-wise scaling based on first and second rolling moments
        * @param layerIndex The geometric position index of the targeted layer
        * @param W Mutable reference to the weight Matrix layout
        * @param B Mutable reference to the bias vector Matrix layout
        * @param dW Constant reference to the accumulated weight gradient Matrix
        * @param dB Constant reference to the accumulated bias gradient Matrix
        */
        void updateLayer(int layerIndex, Matrix& W, Matrix& B, const Matrix& dW, const Matrix& dB) override;
};

#endif