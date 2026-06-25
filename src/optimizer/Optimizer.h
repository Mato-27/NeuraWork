/**
* @file Optimizer.h
* @brief File containing Optimizer class definition
* @author Magdi.V
* @date June 14, 2026
*/

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "../layer/LinearLayer.h" 
#include "../layer/ReLULayer.h" 
#include <memory>

/**
* @class Optimizer
* @brief Abstract base class defining the contract for parameter optimization strategies 
*/
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

        /**
        * @brief Interface method for mutating a single layer's parameters in-place
        * @details Pure virtual execution block providing direct references of parameters and
        * gradients to execute targeted optimization steps across contiguous memory structures
        * @param layerIndex The geometric position index of the targeted layer
        * @param W Mutable reference to the weight Matrix layout
        * @param B Mutable reference to the bias vector Matrix layout
        * @param dW Constant reference to the accumulated weight gradient Matrix
        * @param dB Constant reference to the accumulated bias gradient Matrix
        */
        virtual void updateLayer(int layerIndex, Matrix& W, Matrix& B, const Matrix& dW, const Matrix& dB) = 0;
};

#endif