/**
* @file Matrix.h
* @brief File containing Matrix class definition
* @author Magdi.V
* @date May 27, 2026
*/

#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <random>
#include <fstream>
#include <string>
#include <sstream>

/**
* @class Matrix
* @brief This class allows the use of Matrices
*/
class Matrix {
    private:
        /// number of rows of the Matrix
        int m; 
        /// number of columns of the Matrix
        int n; 
        /// array containing the Matrix values
        double* array; 

    public:
        /**
        * @brief Matrix class constructor
        * @param rows number of rows of the Matrix
        * @param columns number of columns of the Matrix
        */
        Matrix(int rows = 0, int columns = 0);

        /**
        * @brief Matrix class copy constructor
        * @param mat the copied Matrix
        */
        Matrix(const Matrix& mat);

        /**
        * @brief Matrix class destructor
        */
        ~Matrix();

        /**
        * @brief Get rows function
        * @return number of rows of the Matrix
        */
        int getRows () const;

        /**
        * @brief Get columns function
        * @return number of columns of the Matrix
        */
        int getColumns () const;

        /**
        * @brief Operator * overloading
        * @param mat multiplied Matrix
        * @return the product of *this and mat
        */
        Matrix operator * (const Matrix& mat) const;

        /**
        * @brief Operator + overloading
        * @param mat added Matrix
        * @return the sum of *this and mat
        */
        Matrix operator + (const Matrix& mat) const;

        /**
        * @brief Operator - overloading
        * @param mat subtracted Matrix
        * @return the difference between *this and mat
        */
        Matrix operator - (const Matrix& mat) const;

        /**
        * @brief Operator == overloading
        * @param mat Matrix to compare
        * @return the comparison of *this and mat
        */
        bool operator == (const Matrix& mat) const;

        /**
        * @brief Operator = overloading
        * @param mat copied matrix
        * @return *this
        */
        Matrix& operator = (const Matrix& mat);

        /**
        * @brief Operator << overloading
        * @param os Output Stream
        * @param mat Matrix to show
        */
        friend std::ostream& operator << (std::ostream& os, const Matrix& mat);

        /**
        * @brief Fills the matrix with random value between min and max
        * @param min the minimum value
        * @param max the maximum value
        */
        void randomize(double min, double max);

        /**
        * @brief Creates the matrix represented in the file filePath
        * @param filePath the path to the used file containing the Matrix
        * @return *this with the values read in the file filePath
        */
        static Matrix load_csv(const std::string& filePath);

        /**
        * @brief Performs an in-place accumulation with optional row-vector broadcasting
        * @details Automatically detects if the incoming Matrix is a identical layout or a
        * diffusible row vector (1 x N). Dispatches calculations using stride-based 1D index mapping
        * @param mat The Matrix or broadcastable row vector to be accumulated into *this
        * @throws std::invalid_argument If columns do not match, or rows match neither 1 nor target height
        */
        void add (const Matrix& mat);
        
        /**
        * @brief Computes a destination-driven matrix dot product: *this = A * B
        * @details Implements a high-performance tiled GEMM kernel utilizing the 'kij' loop
        * ordering to maximize CPU spatial cache locality and prevent cache misses.
        * @param A The left-hand side multiplying Matrix of dimensions (M x K)
        * @param B The right-hand side multiplying Matrix of dimensions (K x N)
        * @throws std::invalid_argument If inner dimensions mismatch, or if *this layout is not (M x N)
        */
        void dot (const Matrix& A, const Matrix& B);
        
        /**
        * @brief Computes the element-wise Recitfied Liner Unit (ReLU) activation function
        * @details Iterates linearly over the contiguous 1D flattened array to clamp all negative
        * values directly to 0.0 in-place, achieving zero-allocation performance overhaead
        */
        void computeReLU ();

        /**
        * @brief Constant operator () overloading
        * @param i Row index
        * @param j Column index
        * @return the value at the index (i,j) of the Matrix
        */
        double operator () (const int i, const int j) const {
            return array[n*i+j];
        }

        /**
        * @brief Operator () overloading
        * @param i Row index
        * @param j Column index
        * @return the value at the index (i,j) of the Matrix
        */
        double& operator () (const int i, const int j) {
            return array[n*i+j];
        }

        /**
        * @brief Computes the in-place gradient of the ReLU activation function
        * @details Fuses the local derivative computation and the Hadamard product
        * Mutates the current gradient matrix (dL_dA) into the pre-activation gradient
        * by clamping elements to 0.0 if the corresponding element in Z is less than or equal to 0.0
        * @param Z Constant reference to the pre-activation Matrix containing the original linear states
        * @throws std::invalid_argument If the dimensions of Z do not perfectly match the dimensions of *this
        */
        void d_computeReLU (const Matrix& Z);
};

#endif