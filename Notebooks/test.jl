import Pkg; 
#Pkg.add("QXTools")
#Pkg.add("QXGraphDecompositions")
#Pkg.add("QXZoo")
#Pkg.add("DataStructures")
#Pkg.add("QXTns")
#Pkg.add("NDTensors")
#Pkg.add("ITensors")
#Pkg.add("LightGraphs")
#Pkg.add("PyCall")


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

