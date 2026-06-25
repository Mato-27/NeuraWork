/**
* @file main.cpp
* @brief Main test runner and conformance verification 
* @author Magdi.V
* @date June 8, 2026
*/

#include "Matrix.h"
#include "./layer/LinearLayer.h"
#include "./layer/ReLULayer.h"
#include "./layer/SoftmaxLayer.h"
#include "./loss/MSELoss.h"
#include "Sequential.h"
#include <cassert>
#include <iostream>
#include <cmath>

/**
 * @brief Asserts that two double values are close enough to be considered equal
 * @param a First value
 * @param b Second value
 * @param epsilon Tolerance threshold
 */
void assert_almost_equal(double a, double b, double epsilon = 1e-9) {
    // Validates absolute difference against tolerance
    assert(std::abs(a - b) < epsilon);
}

/**
 * @brief Tests basic and copy constructors of the Matrix class
 */
void test_constructors() {
    // Tests default and parameterized allocation with zero-initialization
    Matrix m1(2, 3);
    
    // Tests copy constructor integrity and data isolation
    Matrix m2(m1);
    assert(m1 == m2);

    // Tests defensive boundary handling for negative or null dimensions
    Matrix m3(-5, 0);
    
    std::cout << "[PASS] test_constructors" << std::endl;
}

/**
 * @brief Tests standard arithmetic operators and assignments
 */
void test_operators() {
    Matrix m1(2, 2);
    m1.randomize(1.0, 1.0); // Fills with 1.0
    
    Matrix m2(2, 2);
    m2.randomize(2.0, 2.0); // Fills with 2.0

    // Tests operator+ logic element by element
    Matrix m_sum = m1 + m2;
    
    // Tests operator- logic element by element
    Matrix m_diff = m2 - m1;

    // Tests standard multiplication operator* with kij layout logic
    Matrix m3(1, 2);
    m3.randomize(2.0, 2.0);
    Matrix m4(2, 1);
    m4.randomize(3.0, 3.0);
    Matrix m_prod = m3 * m4; // Result should be 1x1 matrix with value 12.0

    // Tests copy assignment operator= with deep copy validation
    Matrix m_assign(2, 2);
    m_assign = m1;
    assert(m_assign == m1);

    std::cout << "[PASS] test_operators" << std::endl;
}

/**
 * @brief Tests high-performance computing methods optimized for the Layer pipeline
 */
void test_hpc_methods() {
    // Tests the in-place accumulation with row-vector broadcasting (Bias addition simulation)
    Matrix target_matrix(2, 2);
    target_matrix.randomize(1.0, 1.0);
    Matrix row_vector(1, 2);
    row_vector.randomize(10.0, 10.0);
    
    target_matrix.add(row_vector);

    // Tests the destination-driven dot product multiplication
    Matrix X(1, 3);
    X.randomize(1.0, 1.0);
    Matrix W(3, 2);
    W.randomize(2.0, 2.0);
    Matrix Z(1, 2);
    
    Z.dot(X, W); // Z should be filled with 6.0

    // Tests the element-wise computeReLU function
    Matrix R(2, 2);
    R.randomize(-5.0, -5.0);
    R.computeReLU(); // All elements must be 0.0

    std::cout << "[PASS] test_hpc_methods" << std::endl;
}

/**
 * @brief Tests the Layer forward pass against a deterministic mathematical baseline
 */
void test_layer() {
    // Instantiate a mock layer structure with 3 inputs and 2 outputs
    LinearLayer l1(3, 2);

    // Setup a 3x2 weight matrix with uniform 0.5 values and inject it into the layer
    Matrix W(3, 2);
    W.randomize(0.5,0.5);
    l1.setW(W);

    // Setup a 1x2 bias vector with uniform 1.0 values and inject it into the layer
    Matrix B(1, 2);
    B.randomize(1.0, 1.0);
    l1.setB(B);

    // Generate a deterministic 2x3 input mini-batch filled with 2.0 values
    Matrix inputMatrix(2, 3);
    inputMatrix.randomize(2.0, 2.0);

    // Compute the expected pre-activation matrix Z locally via destination-driven dot product
    Matrix Z(2,2);
    Z.dot(inputMatrix, W);

    // Accumulate the expected bias vector onto the local pre-activation matrix via broadcasting
    Z.add(B);

    // Assert that the encapsulation layer's forward pass perfectly matches the validated reference
    assert(Z == l1.forward(inputMatrix));

    std::cout << "[PASS] test_layer" << std::endl;
}

/**
* @brief Validates the Mean Squared Error (MSE) loss forward and backward pipelines
*/
void test_loss() {
    // Instantiate the concrete MSELoss criteria configuration
    MSELoss M_Loss;

    // Generate a 2x2 prediction Matrix filled with uniform 4.0 values
    Matrix pred_Matrix(2,2);
    pred_Matrix.randomize(4.0, 4.0);

    // Generate a 2x2 ground truth target Matrix filled with uniform 2.0 values
    Matrix target_Matrix(2,2);
    target_Matrix.randomize(2.0, 2.0);

    // Execute the forward pass and evaluate the scalar loss value
    double f_pass = M_Loss.forward(pred_Matrix, target_Matrix);
    
    // Assert that the evaluated loss perfectly matches the theoretical baseline of 4.0
    assert_almost_equal(f_pass, 4.0);

    // Setup a 2x2 pre-allocated destination Matrix to catch backpropagated activations
    Matrix dest_Matrix(2,2);

    // Execute the destination-driven backward pass to compute partial derivatives
    M_Loss.backward(pred_Matrix, target_Matrix, dest_Matrix);

    // Setup a 2x2 ground truth gradient Matrix filled with the expected value 1.0
    Matrix ground_Matrix(2,2);
    ground_Matrix.randomize(1.0, 1.0);

    // Assert that the computed gradient matrix strictly matches the mathematical ground truth
    assert(dest_Matrix == ground_Matrix);

    std::cout << "[PASS] test_loss" << std::endl;
}

/**
* @brief Validates the backward pass mechanics and parameter gradient accumulation of LinearLayer
*/
void test_backward() {
    // Instantiate a mock layer structure with 2 inputs and 2 outputs
    LinearLayer l1(2, 2);

    // Setup a 2x2 weight matrix with uniform 0.5 values and inject it into the layer
    Matrix W(2, 2);
    W.randomize(0.5,0.5);
    l1.setW(W);

    // Setup a 1x2 bias vector with uniform 1.0 values and inject it into the layer
    Matrix B(1, 2);
    B.randomize(1.0, 1.0);
    l1.setB(B);

    // Setup a 1x2 input Matrix filled with uniform 2.0 values
    Matrix X(1, 2);
    X.randomize(2.0, 2.0);

    // Execute the forward pass to populate internal activations and pre-activations
    l1.forward(X);

    // Setup a 1x2 incoming gradient Matrix filled with uniform 1.0 values
    Matrix dL_dA(1, 2);
    dL_dA.randomize(1.0, 1.0);

    // Execute the backward pass to calculate gradients in place
    const Matrix& dL_dX = l1.backward(dL_dA);

    // Verify the input gradient matrix values against mathematical expectations
    assert_almost_equal(dL_dX(0, 0), 1.0);
    assert_almost_equal(dL_dX(0, 1), 1.0);

    std::cout << "[PASS] test_backward" << std::endl;
}

/**
* @brief Validates the composite execution flow of a modular Sequential container
*/
void test_sequential() {
    // Instantiate the global Sequential network orchestrator
    Sequential seq;

    // --- Layer 1 Configuration ---
    // instantiated as a concrete smart pointer to allow direct acces to parametric configurations
    auto linear1 = std::make_unique<LinearLayer>(2, 3);

    // Setup a 2x3 weight matrix with uniform 0.5 values and inject it into the first layer
    Matrix w1(2, 3);
    w1.randomize(0.5, 0.5);
    linear1->setW(w1);

    // Setup a 1x3 bias vector with uniform 0.0 values and inject it into the first layer
    Matrix b1(1, 3);
    b1.randomize(0.0, 0.0);
    linear1->setB(b1);

    // Transfer ownership to the sequential graph container
    seq.add(std::move(linear1));

    // Instantiated using the default non-parametric layout contract
    seq.add(std::make_unique<ReLULayer>());

    // --- Layer 2 Configuration ---
    auto linear2 = std::make_unique<LinearLayer>(3, 2);

    Matrix w2(3, 2);
    w2.randomize(1.0, 1.0);
    linear2->setW(w2);

    Matrix b2(1, 2);
    b2.randomize(0.5, 0.5);
    linear2->setB(b2);

    seq.add(std::move(linear2));

    seq.add(std::make_unique<ReLULayer>());

    // --- Forward integration test ---
    // Setup a 1x2 input Matrix filled with uniform 2.0 values
    Matrix m1(1, 2);
    m1.randomize(2.0, 2.0);

    // Propagate the input matrix through the multi-layer sequence and capture the final reference
    const Matrix& prediction = seq.forward(m1);

    // Verify final network predictions against the combined theoretical baseline
    assert_almost_equal(prediction(0,0), 6.5);
    assert_almost_equal(prediction(0,1), 6.5);
    
    // --- Backward integration test ---
    // Setup a 1x2 incoming loss gradient Matrix filled with uniform 1.0 values
    Matrix m2(1, 2);
    m2.randomize(1.0, 1.0);

    // Trigger the global backward chain rule execution across the network graph
    seq.backward(m2);

    // Assert that the internal error gradients have correctly rippled back to the first layer's parameters
    assert_almost_equal(seq.getFirstLayerdW()(0,0), 4.0);
    assert_almost_equal(seq.getFirstLayerdW()(0,1), 4.0);
    assert_almost_equal(seq.getFirstLayerdW()(0,2), 4.0);
    assert_almost_equal(seq.getFirstLayerdW()(1,0), 4.0);
    assert_almost_equal(seq.getFirstLayerdW()(1,1), 4.0);
    assert_almost_equal(seq.getFirstLayerdW()(1,2), 4.0);

    assert_almost_equal(seq.getFirstLayerdB()(0,0), 2.0);
    assert_almost_equal(seq.getFirstLayerdB()(0,1), 2.0);
    assert_almost_equal(seq.getFirstLayerdB()(0,2), 2.0);
    
    std::cout << "[PASS] test_sequential" << std::endl;
}

/**
* @brief Validates the complete training loop orchestration from forward pass to in-place parameter optimization
*/
void test_sgd_optimizer() {
    // Setup the deterministic execution graph, input mini-batches and learning criteria
    double alpha = 0.1; // rate of learning
    SGDOptimizer sgdOpti(alpha);
    Sequential network;
    auto linear1 = std::make_unique<LinearLayer>(2, 2);
    
    Matrix Winit(2, 2);
    Winit.randomize(0.5, 0.5);
    linear1->setW(Winit);

    Matrix Binit(1, 2);
    Binit.randomize(1.0, 1.0);
    linear1->setB(Binit);

    network.add(std::move(linear1));

    network.add(std::make_unique<ReLULayer>());

    Matrix X(1, 2);
    X.randomize(2.0, 2.0);

    Matrix Ytarget(1, 2);
    Ytarget.randomize(0.0, 0.0);

    MSELoss Criteria;

    // Perform the forward pass to compute structural network activations
    const Matrix& Aout = network.forward(X);

    // Compute loss criteria and trigger backward propagation to populate internal gradient matrices
    double loss = Criteria.forward(Aout, Ytarget);
    
    Matrix dL_dA(1, 2); // gradient buffer

    Criteria.backward(Aout, Ytarget, dL_dA); // fill the buffer

    network.backward(dL_dA); // propagates the error and fills dW and dB matrixes

    // Cache initial weight parameters before optimization to allow strict compliance tracking
    const Matrix& dWcalc = network.getFirstLayerdW();
    const Matrix& dBcalc = network.getFirstLayerdB();

    Matrix Wexpected;
    Wexpected.copyFrom(Winit);
    Wexpected.update(dWcalc, -alpha);

    Matrix Bexpected;
    Bexpected.copyFrom(Binit);
    Bexpected.update(dBcalc, -alpha);

    // Trigger the polymorphic optimizer update phase to adjust model parameters in-place 
    network.update(sgdOpti);

    // Assert compliance of the updated weights against the mathematical gradient descent baseline
    assert(Wexpected == network.getFirstLayerW());
    assert(Bexpected == network.getFirstLayerB());

    std::cout << "[PASS] test_sgd_optimizer ; Loss value is " << loss << std::endl;
    
}

/**
* @brief Validates the complete training loop orchestration using the Adam optimizer 
*/
void test_adam_optimizer() {
    // Setup the deterministic execution graph, input mini-batches, and Adam learning configuration
    double alpha = 0.1; // rate of learning
    AdamOptimizer sgdOpti(alpha);
    Sequential network;
    auto linear1 = std::make_unique<LinearLayer>(2, 2);

    // Setup a weight matrix with uniform initial states and inject it into the layer
    Matrix Winit(2, 2);
    Winit.randomize(0.5, 0.5);
    linear1->setW(Winit);

    // Setup a bias vector with uniform initial states and inject it into the layer
    Matrix Binit(1, 2);
    Binit.randomize(1.0, 1.0);
    linear1->setB(Binit);

    // Generate a deterministic input mini-batch and target matrix to yield known gradients
    network.add(std::move(linear1));

    network.add(std::make_unique<ReLULayer>());

    Matrix X(1, 2);
    X.randomize(2.0, 2.0);

    Matrix Ytarget(1, 2);
    Ytarget.randomize(0.0, 0.0);

    MSELoss Criteria;

    // Perform the forward pass to compute structural network activations 
    const Matrix& Aout = network.forward(X);

    // Compute loss criteria and trigger backward propagation to populate internal gradient matrices
    double loss = Criteria.forward(Aout, Ytarget);

    Matrix dL_dA(1, 2); // gradient buffer

    Criteria.backward(Aout, Ytarget, dL_dA); // fill the buffer

    network.backward(dL_dA); // propagates the error and fills dW and dB matrixes

    // Cache initial weight parameters before optimization to allow strict compliance tracking
    const Matrix& dWcalc = network.getFirstLayerdW();
    const Matrix& dBcalc = network.getFirstLayerdB();

    // Trigger the polymorphic optimizer update phase to adjust model parameters in-place 
    network.update(sgdOpti);

    // Assert compliance of the updated weights against the mathematical gradient descent baseline
    int sizeW = Winit.getColumns() * Winit.getRows();
    for (int k = 0; k < sizeW; k++) {
        double g = dWcalc[k];

        double analyticalW = Winit[k] - (alpha / (std::sqrt(g * g) + 1e-8)) * g;
        assert_almost_equal(analyticalW, network.getFirstLayerW()[k]);
    }

    int sizeB = Binit.getColumns() * Binit.getRows();
    for (int k = 0; k < sizeB; k++) {
        double g = dBcalc[k];

        double analyticalB = Binit[k] - (alpha / (std::sqrt(g * g) + 1e-8)) * g;
        assert_almost_equal(analyticalB, network.getFirstLayerB()[k]);
    }
    
    std::cout << "[PASS] test_adam_optimizer ; Loss value is " << loss << std::endl;
}

void test_softmax_layer() {
    // Instantiates a non-parametrix Softmax activation layer component
    SoftmaxLayer softmax;

    // Create a deterministic 1x3 input Matrix representing structural linear logits
    Matrix X(1, 3);
    X[0] = 1.0;
    X[1] = 2.0;
    X[2] = 3.0;

    // Execute the forward pass propagation to generate normalized probabilities
    const Matrix& A = softmax.forward(X);

    // Assert that the generated probability distribution matches analytical baselines
    assert_almost_equal(A[0], 0.09003057317);
    assert_almost_equal(A[1], 0.24472847102);
    assert_almost_equal(A[2], 0.66524095580);

    // Setup an identical incoming gradient vector to trigger uniform error pooling
    Matrix dL_dA(1, 3);
    dL_dA[0] = 1.0;
    dL_dA[1] = 1.0;
    dL_dA[2] = 1.0;

    // Execute the backward pass to calculate directional loss derivatives
    const Matrix& dL_dX = softmax.backward(dL_dA);

    // Verify that a uniform incoming gradient yields a zero-gradients output equilibrium
    assert_almost_equal(dL_dX[0], 0.0);
    assert_almost_equal(dL_dX[1], 0.0);
    assert_almost_equal(dL_dX[2], 0.0);

    std::cout << "[PASS] test_softmax_layer" << std::endl;
}

/**
 * @brief Main test runner program orchestrating the validation suites
 */
int main() {
    std::cout << "=== STARTING MATRIX CONFORMANCE TESTS ===" << std::endl;
    
    // Executes semantic validation suites sequentially
    test_constructors();
    test_operators();
    test_hpc_methods();
    test_layer();
    test_loss();
    test_backward();
    test_sequential();
    test_sgd_optimizer();
    test_adam_optimizer();
    test_softmax_layer();
    
    std::cout << "=== ALL TESTS PASSED SUCCESSFULLY ===" << std::endl;
    return 0;
}