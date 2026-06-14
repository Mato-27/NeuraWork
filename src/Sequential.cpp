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
    array[0]->forward(input);
    for (unsigned int i = 1; i < array.size(); i++) {
        array[i]->forward(array[i-1]->getA());
    }
    return array.back()->getA();
}

void Sequential::backward(const Matrix& dL_dA) {
    int size = array.size() - 1;
    array[size]->backward(dL_dA, array[size]->getdX());
    for (int i = size - 1; i >= 0; i--) {
        array[i]->backward(array[i+1]->getdX(), array[i]->getdX());
    }
}

const Matrix& Sequential::getFirstLayerW() {
    return array[0]->getW();
}

const Matrix& Sequential::getFirstLayerB() {
    return array[0]->getB();
}

const Matrix& Sequential::getFirstLayerdW() {
    return array[0]->getdW();
}

const Matrix& Sequential::getFirstLayerdB() {
    return array[0]->getdB();
}

void Sequential::update(Optimizer& optimizer) {
    optimizer.update(array);
}