using Graphs
using Metis
using MPI
using QXTns
using QXGraphDecompositions

export partition_with_metis, MetisPartitioner

"""
Handles graph partitioning using METIS algorithm
"""
struct MetisPartitioner
    nparts::Int
    options::Dict{Symbol, Any}
    
    function MetisPartitioner(nparts::Int)
        # Create METIS options dictionary
        options = Dict{Symbol, Any}(
            :ncuts => 1,
            :objtype => :cut,
            :seed => 1
        )
        new(nparts, options)
    end
end

"""
Partitions the graph using METIS and returns partition assignments
"""
function partition_with_metis(g::AbstractGraph, partitioner::MetisPartitioner)
    adj_matrix = adjacency_matrix(g)
    partition_vector = Metis.partition(
        adj_matrix, 
        partitioner.nparts; 
        alg=:KWAY,
        options=partitioner.options
    )
    return partition_vector
end

"""
Creates subnetworks based on METIS partitioning
"""
function create_subnetworks(g::AbstractGraph, tnc_tn::TensorNetwork, partitions::Vector{Int})
    subnetworks = Vector{TensorNetwork}()
    unique_parts = unique(partitions)
    
    for part in unique_parts
        vertices = findall(x -> x == part, partitions)
        subg, vmap = induced_subgraph(g, vertices)
        
        # Create subnetwork from the vertices in this partition
        local_tensors = Dict{Symbol, Any}()
        for v in vertices
            tensor_name = Symbol("t$v")
            if haskey(tnc_tn.tensor_map, tensor_name)
                local_tensors[tensor_name] = tnc_tn.tensor_map[tensor_name]
            end
        end
        
        push!(subnetworks, TensorNetwork(local_tensors))
    end
    
    return subnetworks
end