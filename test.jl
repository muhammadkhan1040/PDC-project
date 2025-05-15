# import Pkg
# Pkg.add("TimerOutputs")
# Pkg.add("ProfileView")
# using TimerOutputs
# using Profile
# using ProfileView

# const to = TimerOutput()

# @timeit to "Add Necessary Packages" begin
#     Pkg.add("QXTools")
#     Pkg.add("QXGraphDecompositions")
#     Pkg.add("QXZoo")
#     Pkg.add("DataStructures")
#     Pkg.add("QXTns")
#     Pkg.add("NDTensors")
#     Pkg.add("ITensors")
#     Pkg.add("LightGraphs")
#     Pkg.add("PyCall")
#     Pkg.add("FlameGraphs")
#     Pkg.add("LLVMOpenMP_jll")
#     Pkg.add("ParallelStencil")
# end

# @timeit to "Using Required Modules" begin
#     using QXTools
#     using QXTns
#     using QXZoo
#     using PyCall
#     using QXGraphDecompositions
#     using LightGraphs
#     using DataStructures
#     using TimerOutputs
#     using ITensors
#     using LinearAlgebra
#     using NDTensors
#     using FlameGraphs
#     using LLVMOpenMP_jll
#     using ParallelStencil
# end

# @timeit to "Load Custom Functions" begin
#     include("../src/functions_article.jl")
# end

# @timeit to "Main Program" begin
#     # Create a QFT circuit with 10 qubits
#     n = 10
#     ng = 3  # for rqc circuits
#     depth = 16
#     seed = 41

#     @timeit to "Circuit Creation" begin
#         # circuit = create_ghz_circuit(n)
#         # circuit = create_qft_circuit(n)
#         circuit = create_rqc_circuit(ng, ng, depth, seed, final_h=true)
#         println("Created circuit with ", circuit.num_qubits, " qubits")
#     end

#     # Configure the contraction algorithm
#     num_communities = 4
#     input = "0"^100
#     output = "0"^100

#     @timeit to "TNC Conversion" begin
#         convert_to_tnc(circuit; input=input, output=output, decompose=true)
#         println("Successfully converted to TNC")
#     end

#     # Run the ComPar algorithm using multicore CPU
#     @timeit to "Contraction" begin
#         try
#             Profile.@profile begin
#                 result = ComParCPU(circuit, input, output, num_communities; timings=true)
#                 println("Contraction result: ", result)
#                 println(result)
#             end
#         catch e
#             println("Error during contraction: ", e)
#             rethrow(e)
#         end
#     end
# end

# # Display timing results
# println("\n" * repeat("=", 50))
# println("Timing Results:")
# show(to; allocations=true, sortby=:firstexec)
# println("\n" * repeat("=", 50))

# # Generate profile visualization
# println("\nGenerating profile visualization...")
# ProfileView.view()

# # Save timing results to file
# open("profiling_results.txt", "w") do io
#     show(io, to; allocations=true, sortby=:firstexec)
# end

using TimerOutputs
using Profile
using ProfileView

const to = TimerOutput()

@timeit to "Add Necessary Packages" begin
    # List of required packages
    required_packages = [
        "QXTools",
        "QXGraphDecompositions",
        "QXZoo",
        "DataStructures",
        "QXTns",
        "NDTensors",
        "ITensors",
        "LightGraphs",
        "PyCall",
        "FlameGraphs",
        "LLVMOpenMP_jll",
        "ParallelStencil"
    ]
    
    # Check and add only missing packages
    for pkg in required_packages
        if !(pkg in keys(Pkg.project().dependencies))
            @info "Adding package $pkg..."
            Pkg.add(pkg)
        else
            @info "Package $pkg already installed, skipping..."
        end
    end
end

@timeit to "Using Required Modules" begin
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
    using FlameGraphs
    using LLVMOpenMP_jll
    using ParallelStencil
end

@timeit to "Load Custom Functions" begin
    include("../src/functions_article.jl")
end

@timeit to "Main Program" begin
    # Create a QFT circuit with 10 qubits
    n = 10
    ng = 3  # for rqc circuits
    depth = 16
    seed = 41

    @timeit to "Circuit Creation" begin
        # circuit = create_ghz_circuit(n)
        # circuit = create_qft_circuit(n)
        circuit = create_rqc_circuit(ng, ng, depth, seed, final_h=true)
        println("Created circuit with ", circuit.num_qubits, " qubits")
    end

    # Configure the contraction algorithm
    num_communities = 4
    input = "0"^100
    output = "0"^100

    @timeit to "TNC Conversion" begin
        convert_to_tnc(circuit; input=input, output=output, decompose=true)
        println("Successfully converted to TNC")
    end

    # Run the ComPar algorithm using multicore CPU
    @timeit to "Contraction" begin
        try
            Profile.@profile begin
                result = ComParCPU(circuit, input, output, num_communities; timings=true)
                println("Contraction result: ", result)
                println(result)
            end
        catch e
            println("Error during contraction: ", e)
            rethrow(e)
        end
    end
end

# Display timing results
println("\n" * repeat("=", 50))
println("Timing Results:")
show(to; allocations=true, sortby=:firstexec)
println("\n" * repeat("=", 50))

# Generate profile visualization
println("\nGenerating profile visualization...")
ProfileView.view()

# Save timing results to file
open("profiling_results.txt", "w") do io
    show(io, to; allocations=true, sortby=:firstexec)
end