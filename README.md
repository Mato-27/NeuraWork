# NeuraWork - Deep Learning Framework From Scratch in C++

NeuraWork is a high-performance, lightweight Deep Learning engine built entirely from scratch in C++. Designed with mechanical sympathy and strict systems engineering principles, this framework completely bypasses external library dependencies to expose the low-level mechanics of neural computing.

Optimized for maximum computational efficiency, the architecture prioritizes CPU cache locality, contiguity of memory structures, and a zero-allocation pipeline during network inference.

---

## 🚀 Core Architectural Features

### ⚡ Systems Optimization & Memory Rigor
* **Contiguous 1D Allocation:** All multi-dimensional structures are flattened into a single-dimensional array (`double* array`), eliminating pointer-chasing and maximizing spatial cache locality.
* **Row-Major Memory Layout:** Implements standard row-major indexing (Index = n * i + j) to achieve optimal cache line utilization.
* **CPU Cache Locality (`kij` Loop Ordering):** The matrix multiplication kernel (`Matrix::dot`) is architected using nested `kij` loops to guarantee sequential memory access on the inner loop, minimizing CPU cache misses.
* **RAII & Memory Safety:** Full resource management via standard Resource Acquisition Is Initialization (RAII), featuring automated memory deallocation and clean pointer lifetimes.
* **Exception-Safe Assignments:** Structural reassignments employ the secure Copy-and-Swap idiom (`std::swap`), avoiding self-assignment corruption and offering strong exception guarantees.

### 🧠 Mathematical Operations
* **Zero-Allocation Inference Pipeline:** Neural network layers reuse persistent data structures (`X`, `W`, `B`, `Z`, `A`) across inputs, ensuring zero heap allocations during the forward pass.
* **Industrial Broadcasting:** The accumulation engine (`Matrix::add`) dynamically detects input shapes, enabling seamless addition of a $1 \times M$ bias vector to a multi-batch $B \times M$ pre-activation matrix through stride-based 1D index mapping.
* **In-Place Activations:** The Rectified Linear Unit (ReLU) activation functions directly over the contiguous memory blocks to minimize operational overhead.
* **Robust Text Data Factory:** Native static factory (`Matrix::load_csv`) allowing deep-copy ingestion of comma-separated values with dynamic validation of row symmetry.

---

## 📁 Repository Structure
```text
NeuraWork/
├── bin/            # Compiled binary outputs (executable runners)
├── doc/            # Documentation configurations (Doxyfile)
├── src/            # Core source and header engineering files
│   ├── Matrix.h    # High-performance contiguous matrix specification
│   ├── Matrix.cpp  # Tiled linear algebra algorithms
│   ├── Layer.h     # Structural layer abstraction specification
│   ├── Layer.cpp   # Forward pass pipeline orchestration
│   └── main.cpp    # Conformance testing and validation suite
└── Makefile        # System build orchestration file
```

## 🛠️ Build & Execution Instructions
### Prerequisites
* A standard C++17 or C++20 compliant compiler (g++ or clang++).
* GNU Make tool installed.
* Optional: Doxygen for automated technical documentation 

### Compilation
To compile the core engine along with its exhaustive unit testing suite, execute the following command at the root of the project:
```bash
    make
```
### Running the Conformance Suite
Execute the generated binary execution runner to trigger the validation harness (constructors, arithmetic operators, high-performance computing components, and mini-batch layer broadcasting):

```bash
    ./bin/main.out
```

### Technical Documentation Compilation
To generate a complete, searchable, high-level developer reference using the integrated Doxygen configuration, execute the following command at the root of the project:
```bash
    make docu
```

## 🧪 Comprehensive Conformance Testing
The framework includes a rigid validation architecture within ```src/main.cpp```. It ensures structural and algorithmic accuracy across four major layers:
* **Structural Allocations:** Asserts deep isolation, copy integrity, and boundary conditions for zero/negative matrix geometries.
* **Linear Algebra Correctness:** Validates addition, subtraction, and nested kij multiplication.
* **High-Performance Methods:** Tests the robustness of in-place memory mutations and the array-flattening indexing logic.
* **Mini-Batch Layer Verification:** Asserts forward pass correctness by executing a mock $3 \times 2$ layer layout over an incoming $2 \times 3$ data mini-batch, confronting actual results against a deterministic mathematical baseline.
* **Loss Infrastrucure Validation:** Asserts the mathematical exactness of the Mean Squared Error (MSE) forward scalar cost calculation and validates the destination-driven partial derivatives against a deterministic mathematical baseline.
* **Backpropagation Engine Validation:** Mocks a miniature network topology to verify the mathematical exactness of the generalized chain rule. Validates in-place input gradients ($\frac{\partial L}{\partial X}$), weight updates ($\frac{\partial L}{\partial W}$), and column-wise bias reductions ($\frac{\partial L}{\partial B}$) against deterministic baselines.

## 🗺️ Engineering Roadmap
* **[x] Phase 1:** Core Matrix Engine, RAII Memory Management, Cache-Optimized Multiplication.

* **[x] Phase 2:** Static Ingestion Engines (CSV Loader), Robust Structural Guard Clauses.

* **[x] Phase 3:** Layer Object Abstraction, Dynamic Mini-Batching, Zero-Allocation Broadcast Engine.

* **[X] Phase 4:** Loss Function Infrastructure (MSE, Cross-Entropy) and Output Gradient Vectors.

* **[X] Phase 5:** Backpropagation Engine (Chain Rule Matrix Differentiation and Backward Pass Tracking).

* **[X] Phase 6:** Optimization Engine (Stochastic Gradient Descent and Adaptive Moment Estimation - Adam).

* **[X] Phase 7:** Network Container Orchestration (Sequential Network Pipeline Builder).

## 📄 Author & Specifications
* **Author:** *Magdi Vitteau* (Claude Bernard University Lyon 1)

* **Status:** Phase 6 & 7 Functionnal - Sequential Orchestration & SGD Engine Complete

* **License:** Open for Academic and High-Performance Software Review