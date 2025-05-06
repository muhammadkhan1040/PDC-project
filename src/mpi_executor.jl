using MPI
using QXTns
using TimerOutputs
using Graphs

export MPIExecutor, distribute_and_contract

"""
Handles MPI-based distributed tensor network contraction
"""
struct MPIExecutor
    comm::MPI.Comm
    rank::Int
    size::Int
    timer::TimerOutput
    
    function MPIExecutor()
        MPI.Init()
        comm = MPI.COMM_WORLD
        new(comm, 
            MPI.Comm_rank(comm), 
            MPI.Comm_size(comm),
            TimerOutput())
    end
end

"""
Distributes and contracts tensor networks using MPI
"""
function distribute_and_contract(executor::MPIExecutor, subnetworks::Vector{TensorNetwork})
    @timeit executor.timer "distribution" begin
        local_networks = if executor.rank == 0
            chunk_size = ceil(Int, length(subnetworks) / executor.size)
            for i in 1:(executor.size-1)
                start_idx = i * chunk_size + 1
                end_idx = min((i + 1) * chunk_size, length(subnetworks))
                MPI.send(subnetworks[start_idx:end_idx], i, 0, executor.comm)
            end
            subnetworks[1:chunk_size]
        else
            MPI.recv(0, 0, executor.comm)
        end
    end
    
    @timeit executor.timer "contraction" begin
        local_result = contract_subnetworks(local_networks)
    end
    
    @timeit executor.timer "reduction" begin
        final_result = if executor.rank == 0
            for i in 1:(executor.size-1)
                remote_result = MPI.recv(i, 1, executor.comm)
                local_result = combine_results(local_result, remote_result)
            end
            local_result
        else
            MPI.send(local_result, 0, 1, executor.comm)
            nothing
        end
    end
    
    return final_result
end