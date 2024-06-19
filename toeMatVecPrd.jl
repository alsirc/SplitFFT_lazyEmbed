function toeMulBrn!(toeOpr::toeDat, lvl::Integer, bId::Integer,
	vOrg::AbstractArray{<:Number}, gInf::kerOpt=kerOpt(false, (), ()))::Nothing

	if lvl > 0
		# forward FFT
		vOrg = toeOpr.fftFwd[lvl] * vOrg
		# synchronize stream if in GPU is active
		if gInf.act == true
			CUDA.synchronize(CUDA.stream())
		end
	end
	# branch until depth of block structure
	if lvl < length(toeOpr.dimInfA)
		# shallow copy of vOrg -> inner structure of vOrg must be a literal
		if gInf.act == true
			vPrg = CuArray{eltype(vOrg)}(undef, toeOpr.dimInfA...)
			CUDA.synchronize(CUDA.stream())
		else
			vPrg = similar(vOrg)
		end
		# split branch, includes phase operation and stream sync
		sptBrn!(toeOpr.dimInfB, vPrg, lvl + 1,
			toeOpr.phzInf[lvl + 1], vOrg, gInf)
		# execute split branches
		# !asynchronous GPU causes mysterious errors + minimal speed up!
		if gInf.act == true
				# origin branch
				toeMulBrn!(toeOpr, lvl + 1, bId, vOrg, gInf)
				# phase modified branch
				toeMulBrn!(toeOpr, lvl + 1,
					nxtBrnId(length(toeOpr.dimInfA), lvl, bId), vPrg, gInf)
		# !asynchronous CPU is fine + some speed up!
		else
			@sync begin
				# origin branch
				Base.Threads.@spawn toeMulBrn!(toeOpr, lvl + 1, bId, vOrg, gInf)
				# phase modified branch
				Base.Threads.@spawn toeMulBrn!(toeOpr, lvl + 1,
					nxtBrnId(length(toeOpr.dimInfA), lvl, bId), vPrg, gInf)
			end
		end
		# merge branches, includes phase operation and stream sync
		mrgBrn!(toeOpr.dimInfB, vOrg, lvl + 1,
			toeOpr.phzInf[lvl + 1], vPrg, gInf)
	else
		# multiply by Toeplitz vector
		mulBrn!(toeOpr.dimInfB, vOrg, toeOpr.toeInf[bId + 1], gInf)
	end

	if lvl > 0
		# inverse FFT
		vOrg = toeOpr.fftRev[lvl] * vOrg
		if gInf.act == true
			CUDA.synchronize(CUDA.stream())
		end
	end
	# terminate task and return control to previous level
	if gInf.act == true
		CUDA.synchronize(CUDA.stream())
	end
	return nothing
end

function sptBrn!(dimInf::AbstractVector{<:Integer}, vPrg::AbstractArray{T},
	dir::Integer, phzV::AbstractArray{T,1}, vOrg::AbstractArray{T},
	gInf::kerOpt=kerOpt(false, (), ()))::Nothing where
	T <: Union{ComplexF64,ComplexF32}

	if gInf.act == true
		@cuda threads=gInf.numTrds blocks=gInf.numBlks sptKer!(dimInf,
			vPrg, dir, phzV, vOrg)
		CUDA.synchronize(CUDA.stream())
	else
		@threads for itr ∈ CartesianIndices(vOrg)

			vPrg[itr] = phzV[itr[dir]] * vOrg[itr]
		end
	end
	return nothing
end

function sptKer!(dimInf::AbstractVector{<:Integer}, vPrg::AbstractArray{T},
	dir::Integer, phzV::AbstractArray{T,1},
	vOrg::AbstractArray{T})::Nothing where T <: Union{ComplexF64,ComplexF32}
	# grid strides
	strX = gridDim().x * blockDim().x
	strY = gridDim().y * blockDim().y
	strZ = gridDim().z * blockDim().z
	# thread indices
	idX = threadIdx().x + (blockIdx().x - 1) * blockDim().x
	idY = threadIdx().y + (blockIdx().y - 1) * blockDim().y
	idZ = threadIdx().z + (blockIdx().z - 1) * blockDim().z

	dirItr = 0

	for itrZ = idZ:strZ:dimInf[3], itrY = idY:strY:dimInf[2],
		itrX = idX:strX:dimInf[1]

		if dir == 1 		dirItr = itrX
		elseif dir == 2 	dirItr = itrY
		else 				dirItr = itrZ end

		@inbounds vPrg[itrX,itrY,itrZ] = phzV[dirItr] * vOrg[itrX, itrY, itrZ]
	end
	return nothing
end

function mrgBrn!(dimInf::AbstractVector{<:Integer}, vOrg::AbstractArray{T},
	dir::Integer, phzV::AbstractArray{T,1}, vPrg::AbstractArray{T},
	gInf::kerOpt= kerOpt(false, (), ()))::Nothing where
	T <: Union{ComplexF64,ComplexF32}

	if gInf.act == true
		@cuda threads = gInf.numTrds blocks = gInf.numBlks mrgKer!(dimInf,
			vOrg, dir, phzV, vPrg)
		CUDA.synchronize(CUDA.stream())
		CUDA.unsafe_free!(vPrg)
		CUDA.synchronize(CUDA.stream())
	else
		@threads for itr ∈ CartesianIndices(vOrg)

			vOrg[itr] = 0.5 * (vOrg[itr] + conj(phzV[itr[dir]]) * vPrg[itr])
		end
	end
	return nothing
end

function mrgKer!(dimInf::AbstractVector{<:Integer}, vOrg::AbstractArray{T},
	dir::Integer, phzV::AbstractArray{T,1},
	vPrg::AbstractArray{T})::Nothing where T <: Union{ComplexF64,ComplexF32}
	# grid strides
	strX = gridDim().x * blockDim().x
	strY = gridDim().y * blockDim().y
	strZ = gridDim().z * blockDim().z
	# thread indices
	idX = threadIdx().x + (blockIdx().x - 1) * blockDim().x
	idY = threadIdx().y + (blockIdx().y - 1) * blockDim().y
	idZ = threadIdx().z + (blockIdx().z - 1) * blockDim().z

	dirItr = 0

	for itrZ = idZ:strZ:dimInf[3], itrY = idY:strY:dimInf[2],
		itrX = idX:strX:dimInf[1]

		if 		dir == 1 		dirItr = itrX
		elseif 	dir == 2 		dirItr = itrY
		else 					dirItr = itrZ	end

		@inbounds vOrg[itrX,itrY,itrZ] = (vOrg[itrX, itrY, itrZ] +
					conj(phzV[dirItr]) * vPrg[itrX, itrY, itrZ]) / 2.0
	end
	return nothing
end

function mulBrn!(dimInf::AbstractVector{<:Integer}, vOrg::AbstractArray{T},
	vMod::AbstractArray{T}, gInf::kerOpt= kerOpt(false, (), ()))::Nothing where
	T <: Union{ComplexF64,ComplexF32}

	if gInf.act == true
		@cuda threads=gInf.numTrds blocks=gInf.numBlks mulKer!(dimInf,
			vOrg, vMod)
		CUDA.synchronize(CUDA.stream())
	else
		@threads for itr ∈ eachindex(vOrg)

			vOrg[itr] *= vMod[itr]
		end
	end
	return nothing
end

function mulKer!(dimInf::AbstractVector{<:Integer}, vOrg::AbstractArray{T},
	vMod::AbstractArray{T})::Nothing where T <: Union{ComplexF64,ComplexF32}
	# grid strides
	strX = gridDim().x * blockDim().x
	strY = gridDim().y * blockDim().y
	strZ = gridDim().z * blockDim().z
	# thread indices
	idX = threadIdx().x + (blockIdx().x - 1) * blockDim().x
	idY = threadIdx().y + (blockIdx().y - 1) * blockDim().y
	idZ = threadIdx().z + (blockIdx().z - 1) * blockDim().z

	for itrZ = idZ:strZ:dimInf[3], itrY = idY:strY:dimInf[2],
		itrX = idX:strX:dimInf[1]

		@inbounds vOrg[itrX,itrY,itrZ] *= vMod[itrX, itrY, itrZ]
	end
	return nothing
end

@inline function nxtBrnId(maxLvl::Integer, lvl::Integer, bId::Integer)::Integer

	return bId + ^(2, maxLvl - (lvl + 1))
end
