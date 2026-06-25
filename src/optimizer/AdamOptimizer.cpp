/**
* @file AdamOptimizer.cpp
* @brief File containing AdamOptimizer class implementation
* @author Magdi.V
* @date June 15, 2026
*/
#include "AdamOptimizer.h"

/* AdamOptimizer class constructor */
AdamOptimizer::AdamOptimizer(const double al, const double b1, const double b2, const double epsilon) : 
    alpha(al),
    b1(b1),
    b2(b2),
    epsilon(epsilon),
    t(0) {

    }

void AdamOptimizer::update(std::vector<std::unique_ptr<Layer>>& layers) {
    t++;
    if (mW.size() == 0) {
        mW.resize(layers.size());
        vW.resize(layers.size());
        mB.resize(layers.size());
        vB.resize(layers.size());
    }
    for (int i = 0; i < (int)layers.size(); i++) {
        layers[i]->update(*this, i);
    }
}

void AdamOptimizer::updateLayer(int layerIndex, Matrix& W, Matrix& B, const Matrix& dW, const Matrix& dB) {
    int rowsW = W.getRows();
    int colsW = W.getColumns();
    int rowsB = B.getRows();
    int colsB = B.getColumns();
    if (mW[layerIndex].getRows() != rowsW || mW[layerIndex].getColumns() != colsW) {
        mW[layerIndex].reshape(rowsW, colsW);
        vW[layerIndex].reshape(rowsW, colsW);
        mB[layerIndex].reshape(rowsB, colsB);
        vB[layerIndex].reshape(rowsB, colsB);
    }

    int sizeW = rowsW * colsW;
    for (int k = 0; k < sizeW; k++) {
        mW[layerIndex][k] = b1 * mW[layerIndex][k] + (1.0 - b1) * dW[k];
        vW[layerIndex][k] = b2 * vW[layerIndex][k] + (1.0 - b2) * (dW[k] * dW[k]);
        double mHat = mW[layerIndex][k] / (1.0 - std::pow(b1, t));
        double vHat = vW[layerIndex][k] / (1.0 - std::pow(b2, t));
        W[k] = W[k] - (alpha / (std::sqrt(vHat) + epsilon)) * mHat;
    }
    
    int sizeB = rowsB * colsB;
    for (int j = 0; j < sizeB; j++) {
        mB[layerIndex][j] = b1 * mB[layerIndex][j] + (1.0 - b1) * dB[j];
        vB[layerIndex][j] = b2 * vB[layerIndex][j] + (1.0 - b2) * (dB[j] * dB[j]);
        double mHat = mB[layerIndex][j] / (1.0 - std::pow(b1, t));
        double vHat = vB[layerIndex][j] / (1.0 - std::pow(b2, t));
        B[j] = B[j] - (alpha / (std::sqrt(vHat) + epsilon)) * mHat;
    }
}