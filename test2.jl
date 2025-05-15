using MPI
using QXTools
using QXTns
using QXZoo
using QXGraphDecompositions
using LightGraphs
using DataStructures
using ITensors
using LinearAlgebra
using NDTensors
using TimerOutputs
using Metis
using SparseArrays
using Statistics

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
    include(joinpath(SRC_PATH, "METIS_partitioning.jl"))
catch e
    println("Rank $mpi_rank failed to include files: ", e)
    MPI.Abort(comm, 1)
end

# Parameters - same on all ranks
const n = 10
const input = "0"^n
const output = "0"^n
const num_communities = 4
const num_threads = 4

# Only rank 0 runs the main test functions
if mpi_rank == 0
    function run_basic_metis_test()
        # Create a simple test graph
        n = 4  # number of vertices
        adjmat = sparse([1,1,2,2,3,3,4,4], [2,3,1,4,1,4,2,3], 1, n, n)
        
        # Convert to symmetric matrix (undirected graph)
        adjmat_sym = adjmat + adjmat'
        
        # Partition the graph
        nparts = 2
        edgecut, part = Metis.partition(adjmat_sym, nparts)
        
        println("Basic METIS test:")
        println("Edge cuts: ", edgecut)
        println("Partition vector: ", part)
    end

    function run_quantum_circuit_with_metis(n::Int, num_communities::Int, num_threads::Int)
        println("\n===== Testing Quantum Circuit with METIS Partitioning =====")
        println("Creating quantum circuit with $n qubits...")
        
        # Validate number of communities
        if num_communities < 1 || num_communities > n
            error("Number of communities must be between 1 and number of qubits ($n)")
        end
        
        circuit = create_qft_circuit(n)
        input = "0"^n
        output = "0"^n
        
        println("Circuit created. Running contraction with METIS partitioning...")
        println("- Number of qubits: $n")
        println("- Number of communities: $num_communities")
        println("- Number of threads: $num_threads")
        
        try
            # First, let's verify the TensorNetwork structure
            tnc = convert_to_tnc(circuit; no_input=false, no_output=false, 
                input=input, output=output, decompose=true)
            
            println("TensorNetwork fields: ", fieldnames(typeof(tnc)))
            
            # Now run the actual contraction
            result = ComParCPU_OpenMP(circuit, input, output, num_communities, num_threads)
            println("\nContraction successful!")
            println("Result: ", result)
        catch e
            println("Error during contraction: ", e)
            showerror(stdout, e)
            println()
        end
    end

    function compare_community_detection_methods(n::Int, num_communities::Int)
        println("\n===== Comparing Community Detection Methods =====")
        
        # Create quantum circuit
        circuit = create_qft_circuit(n)
        input = "0"^n
        output = "0"^n
        
        # Convert to tensor network
        tnc = convert_to_tnc(circuit; no_input=false, no_output=false, input=input, output=output)
        light_graf = convert_to_graph(tnc)
        
        # Get communities using different methods
        reset_timer!()
        
        @timeit "METIS partitioning" begin
            metis_communities = metis_partition_graph(light_graf, num_communities)
        end
        
        @timeit "Girvan-Newman" begin
            labeled_light_graf = LabeledGraph(light_graf)
            gn_communities, _, gn_modularity = labelg_to_communitats_between(labeled_light_graf, num_communities)
        end
        
        @timeit "Fast Greedy" begin
            labeled_light_graf = LabeledGraph(light_graf)
            fg_communities, _, fg_modularity = labelg_to_communitats_fastgreedy(labeled_light_graf)
        end
        
        print_timer()
        
        # Print statistics about communities
        println("\nCommunity size statistics:")
        
        println("METIS ($num_communities communities):")
        metis_sizes = [length(c) for c in metis_communities]
        println("  Min: $(minimum(metis_sizes)), Max: $(maximum(metis_sizes)), Avg: $(sum(metis_sizes)/length(metis_sizes))")
        
        println("Girvan-Newman ($(length(gn_communities)) communities, modularity: $gn_modularity):")
        gn_sizes = [length(c) for c in gn_communities]
        println("  Min: $(minimum(gn_sizes)), Max: $(maximum(gn_sizes)), Avg: $(sum(gn_sizes)/length(gn_sizes))")
        
        println("Fast Greedy ($(length(fg_communities)) communities, modularity: $fg_modularity):")
        fg_sizes = [length(c) for c in fg_communities]
        println("  Min: $(minimum(fg_sizes)), Max: $(maximum(fg_sizes)), Avg: $(sum(fg_sizes)/length(fg_sizes))")
    end

    # Main execution - only on rank 0
    println("Running tests on rank 0")
    
    # Run basic METIS test
    run_basic_metis_test()
    
    # Test with small circuit
    run_quantum_circuit_with_metis(8, 2, num_threads)
    
    # Compare community detection methods on medium circuit
    compare_community_detection_methods(12, 4)
    
    # Test with larger circuit if system can handle it
    println("\nDo you want to run a test with a larger circuit (20 qubits)? (y/n)")
    response = readline()
    if lowercase(response) == "y"
        run_quantum_circuit_with_metis(20, 8, num_threads)
    end
end

# Other ranks wait
MPI.Barrier(comm)

# Create circuit and perform computation on all ranks
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
        println("\nAll ranks created QFT circuit with $n qubits")
        println("Memory before conversion: ", Sys.free_memory()/2^30, " GB free")
    end

    try
        # Convert circuit
        tnc = convert_to_tnc(circuit; input=input, output=output, decompose=true)
        
        if mpi_rank == 0
            println("Successfully converted to TNC")
        end

        # Set OpenMP threads
        set_openmp_threads(num_threads)
        
        # Perform computation
        local_result = ComParCPU_OpenMP(circuit, input, output, num_communities, num_threads)
        
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
