/**
* @file main.cpp
* @brief Main test runner and conformance verification 
* @author Magdi.V
* @date June 8, 2026
*/

#include "Matrix.h"
#include "Layer.h"
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
    Layer l1(3, 2);

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

    // Deep copy the local linear results to isolate the activation baseline execution
    Matrix A = Z;
    // Apply the local element-wise Rectified Linear Unit activation function on the baseline matrix
    A.computeReLU();

    // Define a static ground truth verification matrix filled with the mathematically expected value 4.0
    Matrix R(2, 2);
    R.randomize(4.0, 4.0);

    // Assert that the local execution pipeline matches the calculated ground truth matrix
    assert(A == R);

    // Assert that the encapsulation layer's forward pass perfectly matches the validated reference
    assert(A == l1.forward(inputMatrix));

    std::cout << "[PASS] test_layer" << std::endl;
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
    
    std::cout << "=== ALL TESTS PASSED SUCCESSFULLY ===" << std::endl;
    return 0;
}