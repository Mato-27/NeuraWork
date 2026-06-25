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
        * @brief Reshapes the matrix to a new layout
        * @details If the new total capacity matches the existing one, the memory is preserved 
        * and only dimensions are updated. Otherwise, reallocation clears old states to zeros
        * @param newRows The target number of rows
        * @param newColumns The target number of columns
        */
        void reshape(int newRows, int newColumns);
        
        /**
        * @brief Copies the dimensions and content from a source matrix
        * @details Reuses or reallocates internal storage by calling reshape, 
        * then performs a contiguous memory copy of the underlying data.
        * @param source The Matrix instance to copy from
        */
        void copyFrom(const Matrix& source);
        
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
        * @brief Constant operator [] overloading
        * @param i index
        * @return the value at the index [i] of the Matrix array
        */
        double operator [] (const int i) const {
            return array[i];
        }

        /**
        * @brief Operator [] overloading
        * @param i index
        * @return the value at the index [i] of the Matrix
        */
        double& operator [] (const int i) {
            return array[i];
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

        /**
        * @brief Performs a high-performance matrix multiplication with implicit transposition
        * @details Evaluates dimensions based on flags, resets the destination matrix, and executes
        * an optimized loop nesting layout to maximize CPU cache locality
        * @param A Constant reference to the left operand Matrix
        * @param transA Flag forcing an implicit row/column swap layout for Matrix A 
        * @param B Constant reference to the right operand Matrix
        * @param transB Flag forcing an implicit row/column swap layout for Matrix B 
        * @param C Reference to the pre-allocated destination Matrix where resultes are accumulated
        */
        static void d_mult(const Matrix& A, bool transA, const Matrix& B, bool transB, Matrix& C);

        /***
        * @brief Computes a column-wise reduction sum
        * @details Accumulates elements accros rows for each individual column to build a row-vector
        * @param A Reference to the pre-allocated 1xN destination vector Matrix 
        */
        void sum(Matrix& A);

        /** 
        * @brief Performs an in-place element-wise matrix accumulation scaled by a factor
        * @details Iterates through the flattened 1D array to accumulate values directly onto the current instance
        * @param mat Constant reference to the operand Matrix containing incoming values or gradients
        * @param alpha Scalar multiplier applied to the operand matrix elements before accumulation
        * @throws std::invalid_argument If the dimensions of the operand matrix do not match the current instance 
        */
        void update(const Matrix& mat, const double alpha);

        /**
        * @brief Fills the Matrix with the value given in parameters
        * @param value The value to fill the Matrix with
        */
        void fill(const double value);
};

#endif