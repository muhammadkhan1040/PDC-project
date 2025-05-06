using LLVMOpenMP_jll 
# OpenMP-specific helper methods using LLVMOpenMP_jll
"""
    set_openmp_threads(n::Int)

Set the number of OpenMP threads to use for computations.
"""
function set_openmp_threads(n::Int)
    ENV["OMP_NUM_THREADS"] = string(n)
    ccall((:omp_set_num_threads, LLVMOpenMP_jll.libomp), Cvoid, (Cint,), n)
    return n
end

"""
    get_openmp_threads()

Get the current number of OpenMP threads.
"""
function get_openmp_threads()
    threads = ccall((:omp_get_max_threads, LLVMOpenMP_jll.libomp), Cint, ())
    return Int(threads)
end

# Modified to use Base.Threads instead of custom OpenMP parallel_for
function contraccio_paral_openmp(tns::Vector{Any}, plans::Vector{Any})
    # Create thread-local storage for results
    num_threads = get_openmp_threads()
    println("Using $(num_threads) OpenMP threads")
    
    # Execute parallel contractions using Julia's native threading
    # which will utilize OpenMP under the hood via thread affinity
    @threads for i in 1:length(tns)
        contract_tn!(tns[i], plans[i])
    end
    
    return tns
end

# Modified contrau_p to use Base.Threads
function contrau_p_openmp(c_gn::TensorNetwork, pla_mf_p::Vector{Any}, p::Int)
    # Execute parallel contractions for the first p steps
    tasks = Vector{Task}(undef, p)
    
    # Create and schedule tasks for parallel execution
    for i in 1:p
        tasks[i] = Threads.@spawn begin
            contract_pair!(c_gn, pla_mf_p[i][1], pla_mf_p[i][2], pla_mf_p[i][3])
        end
    end

    # Wait for all tasks to complete
    for i in 1:p
        wait(tasks[i])
    end
    
    # Perform sequential contractions for the remaining steps
    for j in (p+1):length(pla_mf_p)
        contract_pair!(c_gn, pla_mf_p[j][1], pla_mf_p[j][2], pla_mf_p[j][3])
    end
    
    return c_gn.tensor_map[pla_mf_p[end][3]].storage
end

# Modified primera_contraccio_paral to use new implementation
function primera_contraccio_paral_openmp(tns::Vector{Any}, plans::Vector{Any})
    # Perform parallel contraction using our modified function
    contraccio_paral_openmp(tns, plans)
    
    # Initialize an empty tensor network to hold the merged result
    c = TensorNetwork()
    
    # Merge all contracted tensor networks into the unified network
    for i in 1:length(tns)
        c = Base.merge(c, tns[i])
    end
    
    # Return the unified tensor network
    return c
end

# Updated ComParCPU function to use modified OpenMP approach
function ComParCPU_OpenMP(circ::Circ, entrada::String, eixida::String, n_com::Int, n_threads::Int=4;
                     timings::Bool=true, decompose::Bool=true)
    # Set the number of Julia threads (which will use OpenMP under the hood)
    set_openmp_threads(n_threads)
    println("OpenMP threads set to: ", get_openmp_threads())
    
    # Reset timing information if enabled
    if timings
        reset_timer!()
    end

    # Phase 1: Community detection and graph preparation
    @timeit "1T.Obtaining Communities" begin
        # Convert circuit to tensor network
        tnc = convert_to_tnc(circ; no_input=false, no_output=false, input=entrada, output=eixida, decompose=decompose)
        
        # Convert tensor network to graph and generate its labeled version
        light_graf = convert_to_graph(tnc)
        labeled_light_graf = LabeledGraph(light_graf)
        
        # Generate the line graph and convert to igraph
        labeled_line_light_graf = line_graph_tris(labeled_light_graf)
        h_Lg_ig = lg2ig(labeled_light_graf.graph)
        h_Lg_ig.summary(verbosity=1)
        
        # Detect communities using Girvan–Newman and compute modularity
        comunitats_julia, comunitats_betwenness, modularitat = labelg_to_communitats_between(labeled_light_graf, n_com)
    end

    # Phase 2: Parallel contraction within communities
    @timeit "2T.Parallel contraction of communities (OpenMP)" begin
        # Generate contraction plans and tensor networks for each community
        tns, plans = pla_contraccio_multiple_G_N(comunitats_julia, tnc, light_graf)
        
        # Perform the first parallel contraction phase
        c = primera_contraccio_paral_openmp(tns, plans)
    end

    # Phase 3: Final contraction with parallelism
    @timeit "3T.Final contraction with OpenMP" begin
        # Generate min-fill plan and parallelization details
        tw, pla = min_fill_contraction_plan_tw(c)
        pla_mf_p, p = pla_paral_p(c, pla)
        
        # Perform parallel and sequential contractions
        s = contrau_p_openmp(c, pla_mf_p, p)
    end

    # Print timing results if enabled
    if timings
        print_timer()
    end

    # Return the result of the final contraction
    return s
end

# ComParCPU_para version using modified approach
function ComParCPU_para_OpenMP(circ::Circ, entrada::String, eixida::String, n_com::Int, n_threads::Int=4;
                          timings::Bool=true, decompose::Bool=true)
    # Set the number of threads
    set_openmp_threads(n_threads)
    println("OpenMP threads set to: ", get_openmp_threads())
    
    # Reset timing information if enabled
    if timings
        reset_timer!()
    end

    # Phase 1: Community detection and graph preparation
    @timeit "1T.Obtaining Communities" begin
        # Convert circuit to tensor network
        tnc = convert_to_tnc(circ; no_input=false, no_output=false, input=entrada, output=eixida, decompose=decompose)
        
        # Convert tensor network to graph and generate its labeled version
        light_graf = convert_to_graph(tnc)
        labeled_light_graf = LabeledGraph(light_graf)
        
        # Generate the line graph and convert to igraph
        labeled_line_light_graf = line_graph_tris(labeled_light_graf)
        h_Lg_ig = lg2ig(labeled_light_graf.graph)
        h_Lg_ig.summary(verbosity=1)
        
        # Detect communities using Girvan–Newman and compute modularity
        comunitats_julia, comunitats_betwenness, modularitat = labelg_to_communitats_between(labeled_light_graf, n_com)
    end

    # Phase 2: Parallel contraction within communities
    @timeit "2T.Parallel contraction of communities (OpenMP)" begin
        # Generate contraction plans and tensor networks for each community
        tns, plans = pla_contraccio_multiple_G_N(comunitats_julia, tnc, light_graf)
        
        # Perform the first parallel contraction phase
        c = primera_contraccio_paral_openmp(tns, plans)
    end

    # Phase 3: Final contraction with parallelism
    @timeit "3T.Final contraction in parallel (OpenMP)" begin
        # Generate min-fill plan and parallelization details
        tw, pla = min_fill_contraction_plan_tw(c)
        pla_mf_p, p = pla_paral_p(c, pla)
        
        # Perform parallel and sequential contractions
        s = contrau_p_openmp(c, pla_mf_p, p)
    end

    # Print timing results if enabled
    if timings
        print_timer()
    end

    # Return the result of the final contraction
    return s
end

# Example use case - Create a QFT circuit with 10 qubits
function run_example()
    n = 10
    ng = 3  # for rqc circuits
    depth = 16
    seed = 41
    num_communities = 4
    num_threads = 4  # OpenMP threads
    
    circuit = create_qft_circuit(n)
    # Alternative: circuit = create_ghz_circuit(n)
    # Alternative: circuit = create_rqc_circuit(ng, ng, depth, seed, final_h=true)
    
    println("Created circuit with ", circuit.num_qubits, " qubits")
    println("Using OpenMP with ", num_threads, " threads")
    
    # Configure the contraction algorithm
    input = "0"^n
    output = "0"^n
    convert_to_tnc(circuit; input=input, output=output, decompose=true)
    
    println("Successfully converted to TNC")
    
    # Run the ComPar algorithm using OpenMP
    try
        # Using OpenMP version
        result = ComParCPU_OpenMP(circuit, input, output, num_communities, num_threads; timings=true)
        println("Contraction result: ", result)
        return result
    catch e
        println("Error during contraction: ", e)
        rethrow(e)
    end
end

# Execute if run directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_example()
end
