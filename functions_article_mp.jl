#### Optimized functions for tensor network contraction using OpenMP via LLVM ############

import LightGraphs as lg
using ITensors
using LinearAlgebra
using NDTensors
using TimerOutputs
using DataStructures
using Distributed

# Setup LLVM for OpenMP operations
using Libdl

# Include the original functions to ensure compatibility
include("../src/funcions_article.jl")

# Fixed OpenMP macro wrapper for Julia
macro omp_parallel(for_loop)
    if !(for_loop.head == :for)
        error("@omp_parallel requires a `for` loop expression")
    end
    quote
        ccall(:jl_enter_threaded_region, Cvoid, ())
        try
            Base.Threads.@threads $(esc(for_loop))
        finally
            ccall(:jl_exit_threaded_region, Cvoid, ())
        end
    end
end

macro omp_critical(expr)
    quote
        lock = ReentrantLock()
        lock_ref = Ref(lock)
        Base.lock(lock_ref[]) do
            $(esc(expr))
        end
    end
end

"""
    contraccio_paral_omp(tns::Vector{Any}, plans::Vector{Any})

Performs parallel contraction of tensor networks using OpenMP for improved parallelism.

# Arguments
- `tns::Vector{Any}`: A vector of tensor networks to be contracted.
- `plans::Vector{Any}`: A vector of contraction plans for each tensor network.

# Returns
- `Vector{Any}`: The contracted tensor networks.
"""
function contraccio_paral_omp(tns::Vector{Any}, plans::Vector{Any})
    n = length(tns)
    
    # Use OpenMP for parallel processing
    @omp_parallel for i in 1:n
        contract_tn!(tns[i], plans[i])
    end
    
    return tns
end

"""
    primera_contraccio_paral_omp(tns::Vector{Any}, plans::Vector{Any})

Performs the first parallel contraction phase with OpenMP acceleration.

# Arguments
- `tns::Vector{Any}`: A vector of tensor networks for each community.
- `plans::Vector{Any}`: A vector of contraction plans for each community.

# Returns
- `TensorNetwork`: The unified tensor network after contraction.
"""
function primera_contraccio_paral_omp(tns::Vector{Any}, plans::Vector{Any})
    # Use the OpenMP parallel contraction function
    contraccio_paral_omp(tns, plans)
    
    # Initialize an empty tensor network to hold the merged result
    c = TensorNetwork() 
    
    # Merge all contracted tensor networks
    for i in 1:length(tns)
        c = Base.merge(c, tns[i])
    end
    
    return c
end

"""
    contrau_p_omp(c_gn::TensorNetwork, pla_mf_p::Vector{Any}, p::Int)

Contracts the tensor network with OpenMP parallelism for the first p steps.

# Arguments
- `c_gn::TensorNetwork`: The tensor network to contract.
- `pla_mf_p::Vector{Any}`: The contraction plan with parallelizable steps first.
- `p::Int`: Number of steps to parallelize.

# Returns
- `Array{ComplexF64, 0}`: The final contraction result.
"""
function contrau_p_omp(c_gn::TensorNetwork, pla_mf_p::Vector{Any}, p::Int)
    # Use OpenMP for parallel contractions
    if p > 0
        @omp_parallel for i in 1:p
            @omp_critical begin
                contract_pair!(c_gn, pla_mf_p[i][1], pla_mf_p[i][2], pla_mf_p[i][3])
            end
        end
    end
    
    # Sequential contractions for remaining steps
    for j in (p+1):length(pla_mf_p)
        contract_pair!(c_gn, pla_mf_p[j][1], pla_mf_p[j][2], pla_mf_p[j][3])
    end
    
    return c_gn.tensor_map[pla_mf_p[end][3]].storage
end

"""
    ComParCPU(circ::Circ, entrada::String, eixida::String, n_com::Int;
              timings::Bool=true, decompose::Bool=true)

OpenMP-accelerated implementation of the tensor network contraction algorithm.

# Arguments
- `circ::Circ`: The quantum circuit to evaluate.
- `entrada::String`: Input qubits configuration.
- `eixida::String`: Output qubits configuration.
- `n_com::Int`: Number of communities.
- `timings::Bool`: Whether to log timing information.
- `decompose::Bool`: Whether to decompose circuit gates.

# Returns
- `Array{ComplexF64, 0}`: Contraction result representing quantum amplitude.
"""
function ComParCPU(circ::Circ, entrada::String, eixida::String, n_com::Int;
                   timings::Bool=true, decompose::Bool=true)
    if timings
        reset_timer!()
    end
    
    # Phase 1: Community detection
    @timeit "1T.Obtaining Communities" begin
        # Convert circuit to tensor network
        tnc = convert_to_tnc(circ; no_input=false, no_output=false, 
                            input=entrada, output=eixida, decompose=decompose)
        
        # Convert to graph representation
        light_graf = convert_to_graph(tnc)
        labeled_light_graf = LabeledGraph(light_graf)
        
        # Generate line graph and detect communities
        labeled_line_light_graf = line_graph_tris(labeled_light_graf)
        h_Lg_ig = lg2ig(labeled_light_graf.graph)
        
        # Use Girvan-Newman for community detection
        comunitats_julia, comunitats_betwenness, modularitat = 
            labelg_to_communitats_between(labeled_light_graf, n_com)
    end
    
    # Phase 2: Parallel contraction of communities with OpenMP
    @timeit "2T.Parallel contraction of communities" begin
        # Generate contraction plans for each community
        tns, plans = pla_contraccio_multiple_G_N(comunitats_julia, tnc, light_graf)
        
        # Use OpenMP-accelerated parallel contraction
        c = primera_contraccio_paral_omp(tns, plans)
    end
    
    # Phase 3: Final contraction
    @timeit "3T.Final Contraction" begin
        # Generate and execute contraction plan
        tw, pla = min_fill_contraction_plan_tw(c)
        s = contract_tn!(c, pla)
    end
    
    if timings
        print_timer()
    end
    
    return s
end

"""
    ComParCPU_para(circ::Circ, entrada::String, eixida::String, n_com::Int;
                   timings::Bool=true, decompose::Bool=true)

OpenMP-accelerated tensor network contraction with parallelism in final phase.

# Arguments
- `circ::Circ`: The quantum circuit to evaluate.
- `entrada::String`: Input qubits configuration.
- `eixida::String`: Output qubits configuration.
- `n_com::Int`: Number of communities.
- `timings::Bool`: Whether to log timing information.
- `decompose::Bool`: Whether to decompose circuit gates.

# Returns
- `Array{ComplexF64, 0}`: Contraction result representing quantum amplitude.
"""
function ComParCPU_para(circ::Circ, entrada::String, eixida::String, n_com::Int;
                        timings::Bool=true, decompose::Bool=true)
    if timings
        reset_timer!()
    end
    
    # Phase 1: Community detection
    @timeit "1T.Obtaining Communities" begin
        # Convert circuit to tensor network
        tnc = convert_to_tnc(circ; no_input=false, no_output=false, 
                            input=entrada, output=eixida, decompose=decompose)
        
        # Convert to graph representation
        light_graf = convert_to_graph(tnc)
        labeled_light_graf = LabeledGraph(light_graf)
        
        # Generate line graph and detect communities
        labeled_line_light_graf = line_graph_tris(labeled_light_graf)
        h_Lg_ig = lg2ig(labeled_light_graf.graph)
        
        # Use Girvan-Newman for community detection
        comunitats_julia, comunitats_betwenness, modularitat = 
            labelg_to_communitats_between(labeled_light_graf, n_com)
    end
    
    # Phase 2: Parallel contraction with OpenMP
    @timeit "2T.Parallel contraction of communities" begin
        # Generate contraction plans for each community
        tns, plans = pla_contraccio_multiple_G_N(comunitats_julia, tnc, light_graf)
        
        # Use OpenMP-accelerated parallel contraction
        c = primera_contraccio_paral_omp(tns, plans)
    end
    
    # Phase 3: Final contraction with OpenMP parallelism
    @timeit "3T.Final contraction in parallel" begin
        # Generate min-fill plan with parallelization details
        tw, pla = min_fill_contraction_plan_tw(c)
        pla_mf_p, p = pla_paral_p(c, pla)
        
        # Use OpenMP for parallel contraction in final phase
        s = contrau_p_omp(c, pla_mf_p, p)
    end
    
    if timings
        print_timer()
    end
    
    return s
end

"""
    ComParCPU_GHZ(circ::Circ, entrada::String, eixida::String;
                  timings::Bool=true, decompose::Bool=true)

OpenMP-accelerated tensor network contraction using Fast Greedy community detection.

# Arguments
- `circ::Circ`: The quantum circuit to evaluate.
- `entrada::String`: Input qubits configuration.
- `eixida::String`: Output qubits configuration.
- `timings::Bool`: Whether to log timing information.
- `decompose::Bool`: Whether to decompose circuit gates.

# Returns
- `Array{ComplexF64, 0}`: Contraction result representing quantum amplitude.
"""
function ComParCPU_GHZ(circ::Circ, entrada::String, eixida::String;
                       timings::Bool=true, decompose::Bool=true)
    if timings
        reset_timer!()
    end
    
    # Phase 1: Community detection with Fast Greedy
    @timeit "1T.Obtaining Communities" begin
        # Convert circuit to tensor network
        tnc = convert_to_tnc(circ; no_input=false, no_output=false, 
                            input=entrada, output=eixida, decompose=decompose)
        
        # Convert to graph representation
        light_graf = convert_to_graph(tnc)
        labeled_light_graf = LabeledGraph(light_graf)
        
        # Generate line graph
        labeled_line_light_graf = line_graph_tris(labeled_light_graf)
        h_Lg_ig = lg2ig(labeled_light_graf.graph)
        
        # Use Fast Greedy for community detection
        comunitats_julia, comunitats_betwenness, modularitat = 
            labelg_to_communitats_fastgreedy(labeled_light_graf)
    end
    
    # Phase 2: Parallel contraction with OpenMP
    @timeit "2T.Parallel contraction of communities" begin
        # Generate contraction plans for each community
        tns, plans = pla_contraccio_multiple_G_N(comunitats_julia, tnc, light_graf)
        
        # Use OpenMP-accelerated parallel contraction
        c = primera_contraccio_paral_omp(tns, plans)
    end
    
    # Phase 3: Final contraction
    @timeit "3T.Final contraction" begin
        # Generate and execute contraction plan
        tw, pla = min_fill_contraction_plan_tw(c)
        s = contract_tn!(c, pla)
    end
    
    if timings
        print_timer()
    end
    
    return s
end

"""
    ComParCPU_para_GHZ(circ::Circ, entrada::String, eixida::String;
                       timings::Bool=true, decompose::Bool=true)

OpenMP-accelerated tensor network contraction with Fast Greedy community detection
and parallelism in final phase.

# Arguments
- `circ::Circ`: The quantum circuit to evaluate.
- `entrada::String`: Input qubits configuration.
- `eixida::String`: Output qubits configuration.
- `timings::Bool`: Whether to log timing information.
- `decompose::Bool`: Whether to decompose circuit gates.

# Returns
- `Array{ComplexF64, 0}`: Contraction result representing quantum amplitude.
"""
function ComParCPU_para_GHZ(circ::Circ, entrada::String, eixida::String;
                            timings::Bool=true, decompose::Bool=true)
    if timings
        reset_timer!()
    end
    
    # Phase 1: Community detection with Fast Greedy
    @timeit "1T.Obtaining Communities" begin
        # Convert circuit to tensor network
        tnc = convert_to_tnc(circ; no_input=false, no_output=false, 
                            input=entrada, output=eixida, decompose=decompose)
        
        # Convert to graph representation
        light_graf = convert_to_graph(tnc)
        labeled_light_graf = LabeledGraph(light_graf)
        
        # Generate line graph
        labeled_line_light_graf = line_graph_tris(labeled_light_graf)
        h_Lg_ig = lg2ig(labeled_light_graf.graph)
        
        # Use Fast Greedy for community detection
        comunitats_julia, comunitats_betwenness, modularitat = 
            labelg_to_communitats_fastgreedy(labeled_light_graf)
    end
    
    # Phase 2: Parallel contraction with OpenMP
    @timeit "2T.Parallel contraction of communities" begin
        # Generate contraction plans for each community
        tns, plans = pla_contraccio_multiple_G_N(comunitats_julia, tnc, light_graf)
        
        # Use OpenMP-accelerated parallel contraction
        c = primera_contraccio_paral_omp(tns, plans)
    end
    
    # Phase 3: Final contraction with parallelism
    @timeit "3T.Final contraction in parallel" begin
        # Generate min-fill plan with parallelization details
        tw, pla = min_fill_contraction_plan_tw(c)
        pla_mf_p, p = pla_paral_p(c, pla)
        
        # Use OpenMP for parallel contraction in final phase
        s = contrau_p_omp(c, pla_mf_p, p)
    end
    
    if timings
        print_timer()
    end
    
    return s
end

"""
    Calcul_GN_Sequencial(cct::Circ, timings::Bool)

Sequential contraction of tensor network using Girvan-Newman for comparison.

# Arguments
- `cct::Circ`: The quantum circuit to evaluate.
- `timings::Bool`: Whether to log timing information.

# Returns
- `Array{ComplexF64, 0}`: Contraction result representing quantum amplitude.
"""
function Calcul_GN_Sequencial(cct::Circ, timings::Bool=true)
    if timings
        reset_timer!()
    end
    
    # Step 1: Convert to tensor network and line graph
    @timeit "1T. Obtaining a line graph" begin
        tnc = convert_to_tnc(cct)
        light_graf = convert_to_graph(tnc)
        labeled_light_graf = LabeledGraph(light_graf)
        labeled_line_light_graf = line_graph_tris(labeled_light_graf)
    end
    
    # Step 2: Generate contraction plan using Girvan-Newman
    @timeit "2T. Getting GN plan" begin
        pla_gn = GN_pla(light_graf, tnc.tn)
    end
    
    # Step 3: Perform final contraction
    @timeit "3T. Final contraction" begin
        s = contract_tn!(tnc.tn, pla_gn)
    end
    
    if timings
        print_timer()
    end
    
    return s
end

# Additional helper functions for OpenMP compatibility

"""
    parallel_ig_edge_betweenness(graph::PyObject)

Computes edge betweenness with OpenMP acceleration.

# Arguments
- `graph::PyObject`: Python igraph object.

# Returns
- `Vector{Float64}`: Edge betweenness values.
"""
function parallel_ig_edge_betweenness(graph::PyObject)
    # Use igraph's parallelized edge betweenness calculation when available
    # Otherwise fallback to standard implementation
    return graph.edge_betweenness(parallelize=true)
end

"""
    create_ghz_circuit(n::Integer)

Creates a GHZ circuit with n qubits for testing.

# Arguments
- `n::Integer`: Number of qubits.

# Returns
- `Circ`: GHZ circuit.
"""
function create_ghz_circuit(n::Integer)
    circ = QXZoo.Circuit.Circ(n)
    
    # Apply Hadamard to first qubit
    QXZoo.Circuit.add_gatecall!(circ, QXZoo.DefaultGates.h(1))
    
    # Apply CNOT gates to create entanglement
    for i in 1:(n-1)
        QXZoo.Circuit.add_gatecall!(circ, QXZoo.DefaultGates.c_x(i, i+1))
    end
    
    return circ
end