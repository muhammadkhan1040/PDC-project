
# Multistage Contraction Algorithm Repository

This repository contains the implementation of the multistage contraction algorithm (`ComPar`) designed for the efficient simulation of quantum circuits. The algorithm leverages parallel tensor network contraction methods to represent and compute complex quantum circuits efficiently. 

The development of this repository is based on the [QXTools package for Julia](https://juliaqx.github.io/QXTools.jl). QXTools is an open-source framework that facilitates the creation, manipulation, and simulation of tensor networks associated with quantum circuits. It integrates state-of-the-art libraries for tensor operations and supports advanced parallelization techniques on both CPUs and GPUs.

The repository includes several variants of the multistage algorithm, leveraging both multicore multiprocessors and GPUs for parallel tensor contraction. These implementations align with the algorithms and results described in the article.

**Reference**: Pastor, A. M., Badia, J. M., Castillo, M. I. (2025). A Community Detection-Based Parallel Algorithm for Quantum Circuit Simulation Using Tensor Networks. *The Journal of Supercomputing*.

## Contents

### `src`
The `src` directory contains the core implementation of the multistage contraction algorithm. This includes:

- The multistage algorithm described in the article.
- Variants using different parallelization strategies for tensor contraction on multicore processors and GPUs.

### Jupyter Notebooks
Three commented Jupyter notebooks are provided to test the algorithms and replicate experimental results included in the article. These notebooks guide users through setting up and running the experiments step by step.

## Prerequisites

To test the code, ensure the following software is installed:

1. **Julia**
   - Download and install Julia from [https://julialang.org/](https://julialang.org/).

2. **Python** (for running the Jupyter notebooks)
   - Install Python from [https://www.python.org/](https://www.python.org/).
   - Install Jupyter Notebook by running:
     ```bash
     pip install notebook
     ```

> **Note:** There is no need to preinstall QXTools or other related packages. The required packages can be installed interactively within the Julia environment when running the code or via the provided Jupyter notebooks.

## Quick Start Example

Here is a simple "Hello World" example to run the `ComPar` algorithm on a GHZ circuit of a specific size, using a multicore CPU in all stages.

### Julia Code Example

```julia
# Add necessary packages
import Pkg; 
Pkg.add("QXTools")
Pkg.add("QXGraphDecompositions")
Pkg.add("QXZoo")
Pkg.add("DataStructures")
Pkg.add("QXTns")
Pkg.add("NDTensors")
Pkg.add("ITensors")
Pkg.add("LightGraphs")
Pkg.add("PyCall")


# Using required modules
using QXTools
using QXTns
using QXZoo
using PyCall
using QXGraphDecompositions
using LightGraphs
using DataStructures
using TimerOutputs
using ITensors
using LinearAlgebra
using NDTensors

# Load custom functions from the folder src
include("../src/funcions_article.jl");

# Create a GHZ circuit with 10 qubits
circuit = create_ghz_circuit(10)

# Convert the circuit to a tensor network circuit (TNC)
tnc = convert_to_tnc(circuit)

# Configure the contraction algorithm
num_communities = 4  # Number of communities for the multistage algorithm
input_state = "0" ^ 10  # All qubits initialized to 0
output_state = "1" ^ 10 # Target output state

# Run the ComPar algorithm using multicore CPU
result = ComParCPU(circuit, input_state, output_state, num_communities;
                    timings=true, decompose=true)

# Print results
println("Contraction completed. Results:")
println(result)
```

### Running the Code
Save the above code in a file, e.g., `run_example.jl`. Then, run the file from the Julia REPL, using the -t auto option to be able to use threads:

```bash
julia -t auto run_example.jl
```

## Using Jupyter Notebooks

To explore the algorithms further, open one of the provided notebooks:

1. Navigate to the `notebooks` directory.
2. Launch the Jupyter Notebook server:
   ```bash
   jupyter notebook
   ```
   Although you can use ```IJulia``` from Julia REPL. Please visit [here](https://julialang.github.io/IJulia.jl/stable/manual/installation/):
   ```
   using IJulia
   notebook()
    ```
    This action launchs the IJulia notebook in your browser.
4. Open a notebook and follow the instructions provided.

---

We hope this repository provides valuable resources for exploring and experimenting with the multistage contraction algorithms!
