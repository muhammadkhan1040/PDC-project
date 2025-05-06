using MPI
using QXTools
using LLVMOpenMP_jll
using QXTns
using QXZoo
using QXGraphDecompositions
using LightGraphs
using DataStructures
using ITensors
using LinearAlgebra
using NDTensors
using TimerOutputs

# Explicit imports
import QXTools.Circuits: create_qft_circuit
import QXTools: convert_to_tnc

# Initialize MPI
MPI.Init()

comm = MPI.COMM_WORLD
mpi_rank = MPI.Comm_rank(comm)
mpi_size = MPI.Comm_size(comm)

# Fix path issues - use absolute paths
const PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))
const SRC_PATH = joinpath(PROJECT_ROOT, "src")

# Load functions on all processes first
try
    include(joinpath(SRC_PATH, "functions_article.jl"))
    include(joinpath(SRC_PATH, "TensorContraction_OpenMP.jl"))
catch e
    println("Rank $mpi_rank failed to include files: ", e)
    MPI.Abort(comm, 1)
end

# Parameters - same on all ranks
const n = 10
const input = "0"^n
const output = "0"^n
const num_communities = 4

# Create circuit and perform computation
let
    # Create circuit locally on each rank (deterministic operation)
    circuit = try
        create_qft_circuit(n)
    catch e
        println("Rank $mpi_rank failed to create circuit: ", e)
        MPI.Abort(comm, 1)
        nothing
    end

    if mpi_rank == 0
        println("All ranks created QFT circuit with $n qubits")
        println("Memory before conversion: ", Sys.free_memory()/2^30, " GB free")
    end

    try
        # Convert circuit
        tnc = convert_to_tnc(circuit; input=input, output=output, decompose=true)
        
        if mpi_rank == 0
            println("Successfully converted to TNC")
        end

        # Set OpenMP threads
        set_openmp_threads(4)  # Use the function from TensorContraction_OpenMP.jl
        
        # Perform computation
        local_result = ComParCPU_OpenMP(circuit, input, output, num_communities, 4)
        
        # Verify results (optional)
        if mpi_rank == 0
            println("Final result: ", local_result)
        end

    catch e
        println("Rank $mpi_rank error during computation: ", e)
        showerror(stdout, e, catch_backtrace())
        MPI.Abort(comm, 1)
    end
end

MPI.Barrier(comm)
MPI.Finalize()
