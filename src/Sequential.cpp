/**
* @file Sequential.cpp
* @brief File containing Sequential class implementation
* @author Magdi.V
* @date June 14, 2026
*/

#include "Sequential.h"

void Sequential::add(std::unique_ptr<Layer> layer) {
    array.push_back(std::move(layer));
}

const Matrix& Sequential::forward(const Matrix& input) {
    // Establish a tracking pointer to cascade activation flows throough the graph
    const Matrix* currentInput = &input;

    // Iterate sequentially through all polymorphic architectural components
    for (auto& layer : array) currentInput = &(layer->forward(*currentInput));
    
    // Return a constant reference to the final operational network output
    return *currentInput;
}

void Sequential::backward(const Matrix& dL_dA) {
    // Establish a tracking pointer to cascade error gradients upstream
    const Matrix* currentGradient = &dL_dA;

    // Traverse the execution graph in reverse chronological order (Chain rule)
    for (int i = array.size() - 1; i >= 0; i--) currentGradient = &(array[i]->backward(*currentGradient));
}

const Matrix& Sequential::getFirstLayerW() {
    // Execute a secure runtime downcast to inspect localized parameters
    LinearLayer* linear = dynamic_cast<LinearLayer*>(array.front().get());
    if (!linear) throw std::runtime_error("Architecture Error: First layer is not a parametric LinearLayer");
    return linear->getW();
}

const Matrix& Sequential::getFirstLayerB() {
    // Execute a secure runtime downcast to inspect localized parameters
    LinearLayer* linear = dynamic_cast<LinearLayer*>(array.front().get());
    if (!linear) throw std::runtime_error("Architecture Error: First layer is not a parametric LinearLayer");
    return linear->getB();
}

const Matrix& Sequential::getFirstLayerdW() {
    // Execute a secure runtime downcast to inspect localized parameters
    LinearLayer* linear = dynamic_cast<LinearLayer*>(array.front().get());
    if (!linear) throw std::runtime_error("Architecture Error: First layer is not a parametric LinearLayer");
    return linear->getdW();
}

const Matrix& Sequential::getFirstLayerdB() {
    // Execute a secure runtime downcast to inspect localized parameters
    LinearLayer* linear = dynamic_cast<LinearLayer*>(array.front().get());
    if (!linear) throw std::runtime_error("Architecture Error: First layer is not a parametric LinearLayer");
    return linear->getdB();
}

void Sequential::update(Optimizer& optimizer) {
    optimizer.update(array);
}