/**
* @file Matrix.cpp
* @brief File containing Matrix class implementation
* @author Magdi.V
* @date May 27, 2026
*/

#include "Matrix.h"
#include <algorithm>
#include <cmath>
#include <utility>

/* Matrix class constructor */
Matrix::Matrix(int rows, int columns) : m(rows > 0 ? rows : 0), n(columns > 0 ? columns : 0) {
    array = nullptr;
    if (m > 0 && n > 0) {
        array = new double[m * n];
        for (int i = 0; i < m * n; i++) {
            array[i] = 0;
        }
    }
}

/* Matrix class destructor */
Matrix::~Matrix() {
    if (array != nullptr) {
        delete[] array;
        array = nullptr;
    }
    m = 0;
    n = 0;
}

/* Get rows function */
int Matrix::getRows() const {
    return this->m;
}

/* Get columns function */
int Matrix::getColumns() const {
    return this->n;
}

/* Matrix class copy constructor */
Matrix::Matrix(const Matrix& mat) : m(mat.m), n(mat.n){
    array = nullptr;
    if (mat.array != nullptr) {
        if (m > 0 && n > 0) {
            array = new double[m * n];
            for (int i = 0; i < m * n; i++) {
                array[i] = mat.array[i];
            }
        }
    } 
}

/* Operator = overloading */
Matrix& Matrix::operator = (const Matrix& mat) {
    Matrix tmpMat(mat);

    std::swap(tmpMat.m, m);
    std::swap(tmpMat.n, n);
    std::swap(tmpMat.array, array);

    return *this;
}

/* Operator + overloading */
Matrix Matrix::operator + (const Matrix& mat) const {
    if (m != mat.m || n != mat.n) throw std::invalid_argument("Matrix dimensions must match.");
    Matrix mat2 = Matrix(m, n);
    for (int i = 0; i < m * n; i++) {
        mat2.array[i] = array[i] + mat.array[i];
    }
    return mat2;
}

/* Operator - overloading */
Matrix Matrix::operator - (const Matrix& mat) const {
    if (m != mat.m || n != mat.n) throw std::invalid_argument("Matrix dimensions must match.");
    Matrix mat2 = Matrix(m, n);
    for (int i = 0; i < m * n; i++) {
        mat2.array[i] = array[i] - mat.array[i];
    }
    return mat2;
}

/* Operator * overloading */
Matrix Matrix::operator * (const Matrix& mat) const {
    if (n != mat.m) throw std::invalid_argument("Incompatible matrix dimensions for multiplication.");
    Matrix mat2 = Matrix(m, mat.n);
    for (int k = 0; k< n; k++) {
        for (int i = 0; i<mat2.m; i++) {
            for (int j = 0; j<mat2.n; j++) {
                mat2.array[mat2.n * i + j] = mat2.array[mat2.n * i + j] + array[n*i + k] * mat.array[mat.n * k + j];
            }
        }
    }
    return mat2;
}

/* Operator == overloading */
bool Matrix::operator == (const Matrix& mat) const {
    bool isEqual = true;
    if (m != mat.m || n != mat.n) return false;
    for (int i = 0; i < m * n; i++) {
        isEqual = isEqual && (std::abs(array[i] - mat.array[i]) < 1e-9);
        if (!isEqual) break;
    }
    return isEqual;
}

/* Operator << overloading */
std::ostream& operator << (std::ostream& os, const Matrix& mat) {
    for (int i = 0; i< mat.m; i++) {
        for (int j = 0; j < mat.n; j++) {
            os << mat.array[mat.n * i + j] << " ";
        }
        os << std::endl;
    }
    return os;
}

/* Fills the matrix with random value between min and max */
void Matrix::randomize (double min, double max) {
    /* Random implementation */
    static std::random_device random_device;
    static std::mt19937 random_engine{random_device()};
    std::uniform_real_distribution distribution{min, max};
    for (int i = 0; i < m * n; i++) {
        array[i] = distribution(random_engine);
    }
}

/* Creates the matrix represented in the file filePath */
Matrix Matrix::load_csv(const std::string& filePath) {
    std::ifstream myFile (filePath);
    std::string line;
    std::vector<double> d;
    int columns = 0;
    int firstLineColumns = 0;
    int rows = 0;
    if (myFile.is_open()) {
        while (std::getline(myFile, line)) {
            if (!line.empty()) {
                std::stringstream ss(line);
                std::string value;
                while (getline(ss, value, ',')) {
                    d.push_back(std::stod(value));
                    columns++;
                }
                if (rows == 0) firstLineColumns = columns;
                if (columns != firstLineColumns && rows > 0) throw std::invalid_argument("The Matrix in the file " + filePath + " doesn't have the same amount of columns on each rows.");
                else columns = 0;
                rows++;
            }
        }
        if (rows == 0) throw std::invalid_argument("The Matrix in the file " + filePath + " has no rows.");
        Matrix mat(rows, firstLineColumns);
        std::copy(d.begin(), d.end(), mat.array);
        myFile.close();
        return mat;
    } else {
        throw std::invalid_argument("Couldn't open the file : " + filePath);
    }
}

/* Sum of two matrixes in this instance */
void Matrix::add(const Matrix& mat) {
    if (mat.n != n || (mat.m != m && mat.m != 1)) throw std::invalid_argument("Matrix dimensions must match.");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++ ) {
            this->array[n * i + j] += mat.array[mat.m == 1 ? j : mat.n * i + j];
        }
    }
}

/* Product of two matrices */
void Matrix::dot (const Matrix& A, const Matrix& B) {
    if (A.n != B.m) throw std::invalid_argument("Incompatible matrix dimensions for multiplication.");
    if (this->m != A.m || this->n != B.n) throw std::invalid_argument("Incompatible matrix dimensions for multiplication.");
    for (int i = 0; i < A.m * B.n; i++) this->array[i] = 0.0;
    for (int k = 0; k < A.n; k++) {
        for (int i = 0; i<A.m; i++) {
            for (int j = 0; j<B.n; j++) {
                this->array[this->n * i + j] += A.array[A.n*i + k] * B.array[B.n * k + j];
            }
        }
    }
}

/* Compute Rectified Linear Unit */
void Matrix::computeReLU () {
    for (int i = 0; i < m*n; i++) {
        array[i] = std::max(0.0, array[i]);
    }
}

/* Computes the in-place gradient of the ReLU activation function */
void Matrix::d_computeReLU (const Matrix& Z) {
    if (n != Z.getColumns() || m != Z.getRows()) throw std::invalid_argument("Matrix dimensions must match.");
    for (int i = 0; i < n * m; i++) {
        if (Z.array[i] <= 0.0) array[i] = 0.0;
    }
}