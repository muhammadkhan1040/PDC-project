using MPI
using QXTools
using QXTns
using QXZoo
using QXGraphDecompositions
using LightGraphs
using Metis
using TimerOutputs

# Initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
mpi_rank = MPI.Comm_rank(comm)

# Configuration
const N_QUBITS = 8
const NUM_COMMUNITIES = 2
const NUM_THREADS = 4
const INPUT = "0"^N_QUBITS
const OUTPUT = "0"^N_QUBITS

# Load required files
@everywhere begin
    include("../src/functions_article.jl")
    include("../src/TensorContraction_OpenMP.jl")
    include("../src/METIS_partitioning.jl")
end

function run_metis_test()
    # Create and partition test graph
    adjmat = sparse([1,1,2,2,3,3,4,4], [2,3,1,4,1,4,2,3], 1, 4, 4)
    adjmat_sym = adjmat + adjmat'
    edgecut, part = Metis.partition(adjmat_sym, 2)
    return edgecut, part
end

function run_quantum_circuit()
    circuit = create_qft_circuit(N_QUBITS)
    tnc = convert_to_tnc(circuit; input=INPUT, output=OUTPUT, decompose=true)
    set_openmp_threads(NUM_THREADS)
    return ComParCPU_OpenMP(circuit, INPUT, OUTPUT, NUM_COMMUNITIES, NUM_THREADS)
end

# Main execution
if mpi_rank == 0
    println("Running METIS test:")
    edgecut, part = run_metis_test()
    println("Edge cuts: ", edgecut)
    println("Partition: ", part)

    println("\nRunning quantum circuit with METIS partitioning:")
    result = run_quantum_circuit()
    println("Final result: ", result)
else
    run_quantum_circuit()  # Other ranks just compute
end

MPI.Finalize()
