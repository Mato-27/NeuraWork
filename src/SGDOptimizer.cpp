/**
* @file SGDOptimizer.cpp
* @brief File containing SGDOptimizer class implementation
* @author Magdi.V
* @date June 14, 2026
*/

#include "SGDOptimizer.h"

/* SGDOptimizer class constructor */
SGDOptimizer::SGDOptimizer(const double al) : 
    alpha(al) {

    }

void SGDOptimizer::update(std::vector<std::unique_ptr<Layer>>& layers) {
    for (auto& layer : layers) layer->update(alpha);
}