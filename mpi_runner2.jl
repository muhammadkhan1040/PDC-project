using Pkg
using MPI
using Distributed
using QXTools
using QXTns
using QXZoo
using QXGraphDecompositions
using LightGraphs
using DataStructures
using TimerOutputs
using ITensors
using LinearAlgebra
using NDTensors
using Serialization
using LLVMOpenMP_jll


# Include your existing files
include("../src/functions_article.jl")
include("../src/TensorContraction_OpenMP.jl")

# Check and add missing packages
required_packages = [
    "QXTools", "QXGraphDecompositions", "QXZoo", "DataStructures", 
    "QXTns", "NDTensors", "ITensors", "LightGraphs", "PyCall", 
    "LLVMOpenMP_jll", "ParallelStencil", "MPI"
]

for pkg in required_packages
    if !(pkg in keys(Pkg.project().dependencies))
        @info "Adding package $pkg..."
        Pkg.add(pkg)
    else
        @info "Package $pkg already installed, skipping..."
    end
end

function serialize_circuit(circuit)
    return serialize(IOBuffer(), circuit)  # Serialize to a buffer
end

function deserialize_circuit(serialized_data)
    return deserialize(IOBuffer(serialized_data))  # Deserialize from the buffer
end

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    
    # Only root process creates the circuit
    if rank == 0
        n = 10
        circuit = create_qft_circuit(n)
        input = "0"^n
        output = "0"^n
        num_communities = 4
        num_threads = 8
        
        # Convert circuit to TNC (only needs to be done once)
        convert_to_tnc(circuit; input=input, output=output, decompose=true)
        println("Root: Successfully converted to TNC")
        
        # Serialize and broadcast circuit data
        circuit_data = serialize_circuit(circuit)
        MPI.bcast(circuit_data, 0, comm)
    else
        # Receive serialized circuit data and deserialize it
        circuit_data = MPI.bcast(nothing, 0, comm)
        circuit = deserialize_circuit(circuit_data)
    end
    
    # All processes perform their part of the computation
    try
        result = ComParCPU_OpenMP(circuit, input, output, num_communities, num_threads)
        println("Rank $rank completed computation")
        
        # Gather results at root
        all_results = MPI.gather(result, 0, comm)
        
        if rank == 0
            # Process final results
            println("Final results: ", all_results)
        end
    catch e
        println("Rank $rank encountered error: ", e)
        MPI.Abort(comm, 1)
    end
    
    MPI.Finalize()
end

main()

