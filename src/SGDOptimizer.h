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
};

#endif