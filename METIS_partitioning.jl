"""
    metis_partition_graph(g::AbstractGraph, nparts::Int)

Uses METIS to partition a graph into `nparts` communities.

# Arguments
- `g::AbstractGraph`: The input graph to partition
- `nparts::Int`: Number of partitions/communities to create

# Returns
- `Vector{Vector{Int}}`: A list of communities, where each community is represented as a vector of vertex indices

# Notes
This function creates a sparse adjacency matrix from the graph and uses METIS.partition to find
optimal partitioning that minimizes edge cuts between communities.
"""
function metis_partition_graph(g::AbstractGraph, nparts::Int)
    n = LightGraphs.nv(g)
    
    # Create adjacency matrix
    src_indices = Int[]
    dst_indices = Int[]
    for e in LightGraphs.edges(g)
        push!(src_indices, LightGraphs.src(e))
        push!(dst_indices, LightGraphs.dst(e))
    end
    adjmat = sparse(src_indices, dst_indices, ones(length(src_indices)), n, n)
    adjmat_sym = adjmat + adjmat'
    
    # Partition with METIS
    edgecut, part_vector = Metis.partition(adjmat_sym, nparts)
    
    # Ensure non-empty communities
    communities = Vector{Any}(undef, nparts)
    for i in 1:nparts
        communities[i] = Int[]
    end
    
    for (vertex, community) in enumerate(part_vector)
        push!(communities[community], vertex)
    end
    
    filter!(!isempty, communities)
    
    return communities
end

"""
    ComParCPU_METIS(circ::Circ, entrada::String, eixida::String, n_com::Int;
                    timings::Bool=true, decompose::Bool=true)

Implements a three-phase tensor network contraction algorithm using METIS for community detection.

# Arguments
- `circ::Circ`: The quantum circuit to evaluate.
- `entrada::String`: List of input qubits.
- `eixida::String`: List of output qubits.
- `n_com::Int`: Number of communities to create with METIS.
- `timings::Bool=true`: If `true`, logs and prints timing results for each phase.
- `decompose::Bool=true`: If `true`, decomposes circuit gates into simpler gates.

# Returns
- `Array{ComplexF64, 0}`: The result of the final contraction.

# Notes
Three phases:
1. Use METIS to partition the tensor network graph into communities
2. Perform parallel contraction within each community
3. Perform the final contraction with parallelism
"""
function ComParCPU_METIS(circ::Circ, entrada::String, eixida::String, n_com::Int;
    timings::Bool=true, decompose::Bool=true)
    if timings
        reset_timer!()
    end

    @timeit "1T.METIS Partitioning" begin
        tnc = convert_to_tnc(circ; no_input=false, no_output=false, 
            input=entrada, output=eixida, decompose=decompose)
        light_graf = convert_to_graph(tnc)
        comunitats_julia = metis_partition_graph(light_graf, n_com)

        # Ensure we have at least one non-empty community
        if isempty(comunitats_julia)
            error("METIS partitioning resulted in empty communities")
        end
    end

    @timeit "2T.Parallel contraction of communities" begin
        tns, plans = pla_contraccio_multiple_G_N(comunitats_julia, tnc, light_graf)
        c = primera_contraccio_paral(tns, plans)

        # Check if contraction produced valid result
        # Modified: Use the tensor_map field instead of tensors
        if isempty(c.tensor_map)
            error("Community contraction produced empty tensor network")
        end
    end

    @timeit "3T.Final contraction in parallel" begin
        tw, pla = min_fill_contraction_plan_tw(c)
        pla_mf_p, p = pla_paral_p(c, pla)

        # Ensure valid contraction plan
        if p < 1 || isempty(pla_mf_p)
            error("Invalid contraction plan generated")
        end

        s = contrau_p(c, pla_mf_p, p)
    end

    if timings
        print_timer()
    end

    return s
end

"""
    ComParCPU_METIS_OpenMP(circ::Circ, entrada::String, eixida::String, n_com::Int, num_threads::Int;
                           timings::Bool=true, decompose::Bool=true)

Implements a three-phase tensor network contraction algorithm using METIS for community detection
and OpenMP for parallelization.

# Arguments
- `circ::Circ`: The quantum circuit to evaluate.
- `entrada::String`: List of input qubits.
- `eixida::String`: List of output qubits.
- `n_com::Int`: Number of communities to create with METIS.
- `num_threads::Int`: Number of OpenMP threads to use.
- `timings::Bool=true`: If `true`, logs and prints timing results for each phase.
- `decompose::Bool=true`: If `true`, decomposes circuit gates into simpler gates.

# Returns
- `Array{ComplexF64, 0}`: The result of the final contraction.
"""
function ComParCPU_METIS_OpenMP(circ::Circ, entrada::String, eixida::String, n_com::Int, num_threads::Int;
                               timings::Bool=true, decompose::Bool=true)
    # Set OpenMP threads
    set_openmp_threads(num_threads)
    
    # Reset timing information if enabled
    if timings
        reset_timer!()
    end

    # Phase 1: METIS-based community detection and graph preparation
    @timeit "1T.METIS Partitioning" begin
        # Convert circuit to tensor network
        tnc = convert_to_tnc(circ; no_input=false, no_output=false, input=entrada, output=eixida, decompose=decompose)

        # Convert tensor network to graph
        light_graf = convert_to_graph(tnc)
        
        # Use METIS to partition the graph
        comunitats_julia = metis_partition_graph(light_graf, n_com)
    end

    # Phase 2: Parallel contraction within communities using OpenMP
    @timeit "2T.OpenMP Parallel contraction of communities" begin
        # Generate contraction plans and tensor networks for each community
        tns, plans = pla_contraccio_multiple_G_N(comunitats_julia, tnc, light_graf)
        
        # Use OpenMP for the parallel contraction phase
        c = openmp_contraccio_paral(tns, plans, num_threads)
    end

    # Phase 3: Final contraction with OpenMP parallelism
    @timeit "3T.Final OpenMP contraction" begin
        # Generate min-fill plan and parallelization details
        tw, pla = min_fill_contraction_plan_tw(c)
        pla_mf_p, p = pla_paral_p(c, pla)
        
        # Perform parallel and sequential contractions using OpenMP
        s = openmp_contrau_p(c, pla_mf_p, p, num_threads)
    end

    # Print timing results if enabled
    if timings
        print_timer()
    end

    # Return the result of the final contraction
    return s
end
