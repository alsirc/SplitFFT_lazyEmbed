###PREAMBLE
using CUDA, Base.Threads, LinearAlgebra.BLAS, AbstractFFTs, FFTW, Random, 
BenchmarkTools
# Set the number of BLAS threads. The number of Julia threads is set as an 
# environment variable. The total number of threads is Julia threads + BLAS 
# threads. VICu is does not call BLAS libraries during threaded operations, 
# so both thread counts can be set near the available number of cores. 
threads = nthreads()
BLAS.set_num_threads(threads)
# Analogous comments apply to FFTW threads. 
FFTW.set_num_threads(threads)
# Confirm thread counts
blasThreads = BLAS.get_num_threads()
fftwThreads = FFTW.get_num_threads()
println("toeBreak initialized with ", nthreads(), 
	" Julia threads, $blasThreads BLAS threads, and $fftwThreads FFTW threads.")
###DATA STRUCTURES
struct toeDat

	dimInfA::AbstractVector{<:Integer} 
	dimInfB::AbstractVector{<:Integer} 
	toeInf::AbstractArray{<:AbstractArray{T}}  where 
	(T <: Union{ComplexF64,ComplexF32})
	fftFwd::AbstractArray{<:AbstractFFTs.Plan} 
	fftRev::AbstractArray{<:AbstractFFTs.ScaledPlan}
	phzInf::AbstractArray{<:AbstractArray{T}} where 
	(T <: Union{ComplexF64,ComplexF32})
end

struct kerOpt
	# boolean representing activation of GPU
	act::Bool 
	# number of threads to use when running GPU kernels
	numTrds::Union{Tuple{},NTuple{3,<:Integer}}
	# number of threads to use when running GPU kernels 
	numBlks::Union{Tuple{},NTuple{3,<:Integer}}
end

struct toeOpt
	# Indicates the processors that should be used to perform the matrix vector 
	# product. The first position is associated with the CPU. The second 
	# position is for a GPU. If both positions are true, all processes are 
	# run on the GPU. 
	oprPrc::Tuple{Bool,Bool}
	# runtime options for running GPU kernels
	devOpt::kerOpt
end
###PROCEDURE
# spot fix for batched execution of GPU FFT on internal dimensions 
function toeMul!(toeOpr::toeDat, vOrg::AbstractArray{<:Number}, 
	cmpOpt::toeOpt=toeOpt((1,0), kerOpt(false, (), ())))::Nothing  
	# GPU execution !SETUP TO USE PINNED MEMORY + PERSISTENT GPU!
	if cmpOpt.oprPrc[2] == true
		toeMulBrn!(toeOpr, 0, 0, vOrg, cmpOpt.devOpt)
	# CPU execution 
	elseif cmpOpt.oprPrc[1] == true 
		toeMulBrn!(toeOpr, 0, 0, vOrg)
	# currently unsupported execution type
	else
		error("Unsupported processor specification.")
	end
	return nothing
end
# algorithm for block Toeplitz matrix vector multiplication
include("toeMatVecPrd.jl")