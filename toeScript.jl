# IMPORTS
include("./toeBreakfunction.jl")
include("./toeUtilities.jl")
using Dates
using BenchmarkTools

# A little hack to extract the result from @benchmarkable
@eval BenchmarkTools macro benchmarkable_with_result(args...)
    _, params = prunekwargs(args...)
    bench, result = gensym(), gensym()
    return esc(quote
                   local $bench = $BenchmarkTools.@benchmarkable $(args...)
                   local _, $result = $BenchmarkTools.run_result($bench)
                   $result, $bench
               end)
end

## The function we want to benchmark

#embedMethod(inpVec, dimInf, toeFFT, fftDF, fftDR)

#toeMul(toeOpr, vOrg, cpuOpt)

## Sequence of parameters
# Sequence of array of lengths for the different dimensions from inner most to outer most
#dimInf_seq = vcat([[2^k] for k in 1:26], [repeat([2^k], 2) for k in 1:12], [repeat([2^k], 3) for k in 1:8], [repeat([2^k], 4) for k in 1:6]) #can't go above 2^9 for 3d or simul gets "Killed" ?

dimInf_seq = vcat([[2^k] for k in 1:25], [repeat([2^k], 2) for k in 1:12], [repeat([2^k], 3) for k in 1:8], [repeat([2^k], 4) for k in 1:5])
# /!\ The above variable is an array of arrays /!\
#number of tests for both method
ntrials = length(dimInf_seq)
# Sequence of range of value for the random coeficients
valrange_seq = 100*ones(ntrials)

# Create a benchmark suite (this holds multiple benchmarks)
suite = BenchmarkGroup()
# Benchmarking without parameters for the small lengths
#smallTrials = [1:4 ; 27:30 ; 39:41 ; 47:48]
smallTrials = 1:ntrials

for n in 1:ntrials
    # To follow the current running trial
    println("it runs $n")
    # Do the setup work before running the part we need to measure computational time
    inpVec, toeFFT, fftDF, fftDR, toeOpr, vOrg, cpuOpt = toesetup(dimInf_seq[n], valrange_seq[n])
    if n in smallTrials
        #If the trials tests a small length, then do the benchmarking without the parameters
        # Create a benchmark
        result1, bench1 = BenchmarkTools.@benchmarkable_with_result embedMethod($inpVec, $(dimInf_seq[n]), $toeFFT, $fftDF, $fftDR)
        toeMul!(toeOpr, vOrg, cpuOpt)
        vOrg2 = deepcopy(vOrg)
        result2, bench2 = BenchmarkTools.@benchmarkable_with_result toeMul!($toeOpr, $vOrg, $cpuOpt)
    else
        # Create a benchmark
        result1, bench1 = BenchmarkTools.@benchmarkable_with_result embedMethod($inpVec, $(dimInf_seq[n]), $toeFFT, $fftDF, $fftDR) samples=3 evals=1 seconds=5
        toeMul!(toeOpr, vOrg, cpuOpt)
        vOrg2 = deepcopy(vOrg)
        result2, bench2 = BenchmarkTools.@benchmarkable_with_result toeMul!($toeOpr, $vOrg, $cpuOpt) samples=3 evals=1 seconds=5
    end

    # Check if the result is correct
    if ~(result1 ≈ vOrg2)
        println("Result missmatch at trail number $n")
    end
    # @show result1
    # Add the benchmark to the benchmark suite
    suite["$(n)_embed"] = bench1
    suite["$(n)_splitFFT"] = bench2
end

println("Trials completed")
# Make sure the benchmarks are as correct as possible
# This runs the function a few times to make sure the
# JIT engine does its thing and that the garbage collector
# is not running during the benchmarks.
# Omit this if you are testing
tune!(suite)
# Run the benchmarks
# This will run each function call several times so that you
# can extract out averages, standard deviations, etc.
results = run(suite)

# Save the results to a file
BenchmarkTools.save("myresults@" * Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS") * ".json", results)
