/**
* @file Optimizer.h
* @brief File containing Optimizer class definition
* @author Magdi.V
* @date June 14, 2026
*/

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Layer.h"
#include <memory>

class Optimizer {
    public:
    /**
    * @brief Optimizer class destructor
    */
    virtual ~Optimizer() = default;

    /**
    * @brief Pure virtual method establishing the interface for parameter optimization strategies
    * @details Iterates over the polymorphic layer sequence to adjust internal weights and biases
    * @param layers Reference to the dynamic container holding exclusive pointers to the network architecture 
    */
    virtual void update(std::vector<std::unique_ptr<Layer>>& layers) = 0;
};

#endif