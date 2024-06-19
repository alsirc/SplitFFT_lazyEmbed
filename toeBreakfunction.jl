## IMPORTS
using Base.Threads
include("./toeUtilities.jl")
## FUNCTIONS
#Embedding the vector for circulant embedding method
function embed_vec(V::AbstractArray{ComplexF64}, sizes::AbstractArray{<:Integer})
	d = length(sizes)
	embV = zeros(ComplexF64, [2*sizes[k] for k in 1:d]...)
	@threads for itr in CartesianIndices(V)
		embV[Tuple(itr)...] = V[Tuple(itr)...]
	end
	return embV
end

function even_odd(i, d, lvls) #for the even/odd repartition of the coefficient of the Toeplitz vector
	return mod(div(i, 2^(lvls - d)), 2)
end

function toesetup(dimInf, val_range) # Setup function before the proper computation
	# number of embedding
	lvls = length(dimInf)
	# dimension of each level
	# computation settings (CPU computation only for now)
	cpuOpt = toeOpt((1,0), kerOpt(false, (), ()))
	# devOpt = toeOpt((0,1), kerOpt(true, (256,2,1), (1,128,256)))
	# storage for Toeplitz multiplication step
	toeInf = Array{Array{ComplexF64}}(undef, ^(2, lvls))
	# toeInfG = Array{CuArray{ComplexF64}}(undef, ^(2, lvls))
	# fft planning memory
	fftP = Array{ComplexF64}(undef, dimInf...)
	# fftPG = CuArray{ComplexF64}(undef, dimInf...)
	# phase information
	phzInf = Array{Array{ComplexF64}}(undef, lvls)
	# phzInfG = Array{CuArray{ComplexF64}}(undef, lvls)

	#DEFINE TINY TEST
	# Fourier transform plans
	#Setup for parallelisation of FFTs
	threads = Threads.nthreads()
	FFTW.set_num_threads(threads)
	# planning areas
	fftPD = Array{ComplexF64}(undef, 2 .* dimInf...)
	fftPH = Array{ComplexF64}(undef, 2 .* dimInf...)
	# plans
	fftDF = plan_fft!(fftPD, 1:lvls; flags = FFTW.MEASURE)
	fftHF = plan_fft!(fftPH, 1:lvls; flags = FFTW.MEASURE)
	fftDR = plan_ifft!(fftPD, 1:lvls; flags = FFTW.MEASURE)
	fftHR = plan_ifft!(fftPH, 1:lvls; flags = FFTW.MEASURE)

	# Toeplitz matrix vector
	toeVec = val_range*rand(ComplexF64, [2*dimInf[k] for k in 1:lvls]...)

	toeFFT = fftDF * toeVec #FFT computation #c

	# input vector
	inpVec = val_range*rand(ComplexF64, dimInf...) #needs to be structured
	vOrg = ones(ComplexF64, dimInf...)
	vOrg[:] .= inpVec[:]
	### MEMORY INITIALIZATION
	# copyto!(vOrgG, vOrg)

	# computation of phase transformation
	for itr in 1:lvls

		phzInf[itr] = [exp(-im * pi * k / dimInf[itr]) for k ∈ 0:(dimInf[itr] - 1)]
		# phzInfG[itr] = CuArray{ComplexF64}(undef, dimInf...)
		# copyto!(selectdim(phzInfG, 1, itr), selectdim(phzInf, 1, itr))
	end
	# Toeplitz matrix information
	eoDim = ^(2, lvls)
	toeInf = Array{Array{ComplexF64}}(undef, eoDim)

	not_a_filter = Tuple([Colon() for i in 1:2^(lvls - 1)])

	# write Toeplitz data
	for eoItr ∈ 0:(eoDim - 1)
		# odd / even branch extraction
		toeInf[eoItr + 1] = Array{ComplexF64}(undef, dimInf...)
		# first division is along smallest stride -> largest binary division
		toeInf[eoItr + 1][not_a_filter...] .= toeFFT[Tuple([(1 + even_odd(eoItr, d, lvls)):2:(2*dimInf[d] - 1 + even_odd(eoItr, d, lvls)) for d in 1:lvls])...]
		# verify that all values are numeric.
		if maximum(isnan.(toeInf[eoItr + 1])) == 1 ||
			maximum(isinf.(toeInf[eoItr + 1])) == 1
			error("Fourier information contains non-numeric values.")
		end
	end

	## Fourier Transform Plans
	fftPF = Array{FFTW.cFFTWPlan}(undef, 1, length(dimInf))
	# # fftPFG = Array{CUDA.CUFFT.cCuFFTPlan}(undef, 1, length(dimInf))
	fftPR = Array{FFTW.ScaledPlan}(undef, 1, length(dimInf))
	# # fftPRG = Array{AbstractFFTs.ScaledPlan}(undef, 1, length(dimInf))
	# # initialize
	for dir in 1:length(dimInf)

		fftPF[dir] = plan_fft!(fftP, [dir]; flags = FFTW.MEASURE)
		fftPR[dir] = plan_ifft!(fftP, [dir]; flags = FFTW.MEASURE)
	# 	# fftPFG[dir] = plan_fft!(fftPG, dir)
	# 	# fftPRG[dir] = plan_ifft!(fftPG, dir)
	end
	# # CUDA.synchronize(CUDA.stream())
	## Toeplitz structure
	toeOpr = toeDat(dimInf, dimInf, toeInf, fftPF, fftPR, phzInf)
	# toeOprG = toeDat(dimInf, dimInfG, toeInfG, fftPFG, fftPRG, phzInfG)
	return inpVec, toeFFT, fftDF, fftDR, toeOpr, vOrg, cpuOpt
end
## output of circulant algorithm and storage of the computational time for the method
function embedMethod(inpVec, dimInf, toeFFT, fftDF, fftDR) #begining of the time computation
	# /!\ inpVec is a structured vector /!\
	#Embedding & forward FFT
	embOutfull = fftDF * embed_vec(inpVec, dimInf)

	#Multiplication with first circulant column
	Threads.@threads for itr in CartesianIndices(embOutfull)
		embOutfull[itr] = toeFFT[itr]*embOutfull[itr] #c
	end #try some tests without the "copy"

	#Inverse FFT
	embOutfull = fftDR*embOutfull #c

	#Projection
	embOut = Array{ComplexF64}(undef, dimInf...)
	@threads for itr in CartesianIndices(embOut)
		embOut[itr] = embOutfull[itr]
	end
	return embOut
end

#CUDA.synchronize(CUDA.stream())
