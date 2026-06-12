/**
* @file Loss.h
* @brief File containing Loss class definition
* @author Magdi.V
* @date June 12, 2026
*/

#ifndef LOSS_H
#define LOSS_H

#include "Matrix.h"

/**
* @class Loss
* @brief This class instances the virtual use of Losses
*/
class Loss {
    public:
    /**
    * @brief Loss class destructor
    */
    virtual ~Loss() = default;

    /**
    * @brief Computes the global scalar loss value between predictions and targets
    * @details Pure virtual evaluation function that analyzes network error metrics
    * without modifying the internal state of the loss instance
    * @param A Constant reference to the prediction Matrix of dimensions (Batch Size x Output Size)
    * @param Y Constant reference to the ground truth target Matrix of dimensions (Batch Size x Output Size)
    * @return The evaluated global scalar loss value as a double 
    */
    virtual double forward(const Matrix& A, const Matrix& Y) const = 0;

    /**
    * @brief Computes the gradient of the loss function with respect to the input activations
    * @details Pure virtual mathematical execution block utilizing destination-driven computation
    * to populate the output gradient matrix in-place, maintaining a zero-allocation pipeline
    * @param A Constant reference to the prediction Matrix of dimensions (Batch Size x Output Size)
    * @param Y Constant reference to the ground truth target Matrix of dimensions (Batch Size x Output Size)
    * @param dL_dA Pre-allocated destination Matrix reference to store the partial derivatives (Batch Size x Output Size)
    */
    virtual void backward(const Matrix& A, const Matrix& Y, Matrix& dL_dA) = 0;
};

#endif