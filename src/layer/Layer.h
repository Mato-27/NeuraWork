/**
* @file Layer.h
* @brief File containing Layer class definition
* @author Magdi.V
* @date June 8, 2026
*/

#ifndef LAYER_H
#define LAYER_H

#include "../Matrix.h"

class Optimizer;

/**
 * @class Layer
 * @brief Abstract base class defining the execution contract for all network components
 */
class Layer {
public:
    virtual ~Layer() = default;

    /**
     * @brief Executes the forward propagation pass through the component
     * @param X Constant reference to the input Matrix
     * @return Constant reference to the computed activation Matrix
     */
    virtual const Matrix& forward(const Matrix& X) = 0;

    /**
     * @brief Propagates the structural error gradient backward through the component
     * @param dL_dA Constant reference to the incoming gradient Matrix from the subsequent layer
     * @return Constant reference to the calculated input gradient Matrix (dL_dX)
     */
    virtual const Matrix& backward(const Matrix& dL_dA) = 0;

    /**
     * @brief Delegates parameter optimization via double dispatch
     * @param optimizer Reference to the running Optimizer strategy
     * @param layerIndex The geometric position index within the network topology
     */
    virtual void update(Optimizer& optimizer, int layerIndex) = 0;

    /**
     * @brief Extracts the internal activation cache state
     * @return Constant reference to the cached output Matrix
     */
    virtual const Matrix& getA() const = 0;
};

#endif