/**
* @file MSELoss.h
* @brief File containing MSELoss class definition
* @author Magdi.V
* @date June 12, 2026
*/

#ifndef MSELOSS_H
#define MSELOSS_H

#include "Loss.h"

/**
* @class MSELoss
* @brief This class instances the inherited use of Losses
*/
class MSELoss : public Loss {
    public:
    /**
    * @brief Computes the global scalar loss value between predictions and targets
    * @details Overridden evaluation function that implements the structural MSE formula
    * over the flattened contiguous memory array without altering input metrics
    * @param A Constant reference to the prediction Matrix of dimensions (Batch Size x Output Size)
    * @param Y Constant reference to the ground truth target Matrix of dimensions (Batch Size x Output Size)
    * @return The evaluated global scalar loss value as a double 
    */
    double forward(const Matrix& A, const Matrix& Y) const override;

    /**
    * @brief Computes the gradient of the loss function with respect to the input activations
    * @details Overridden mathematical execution block that writes partial derivatives 
    * directly into the pre-allocated destination matrix using a zero-allocation stride
    * @param A Constant reference to the prediction Matrix of dimensions (Batch Size x Output Size)
    * @param Y Constant reference to the ground truth target Matrix of dimensions (Batch Size x Output Size)
    * @param dL_dA Pre-allocated destination Matrix reference to store the partial derivatives (Batch Size x Output Size)
    */
    void backward(const Matrix& A, const Matrix& Y, Matrix& dL_dA) override;
};

#endif