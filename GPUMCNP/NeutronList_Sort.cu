#include "gpumcnp.inl"

__global__
void write_xpindex_array(int* index_array,int nptcls)
{
	int idx = threadIdx.x;
	int gidx = blockIdx.x*blockDim.x+idx;

	while(gidx < nptcls)
	{
		index_array[gidx] = gidx;
		gidx += blockDim.x*gridDim.x;
	}
}

__global__
void find_bin_boundaries(NeutronList particles,Particlebin bins)
{
	int idx = threadIdx.x;
	int gidx = idx+blockIdx.x*blockDim.x;

	int nptcls = particles.nptcls;

	uint binindex;
	uint binindex_left;
	uint binindex_right;

	while(gidx < nptcls)
	{
		if(gidx == 0)
		{
			binindex = particles.binid[gidx];
			bins.ifirstp[binindex] = gidx;
			bins.binid[binindex] = binindex;
		}
		else if(gidx == nptcls-1)
		{
			binindex = particles.binid[gidx];
			bins.nptcls[binindex] = gidx+1;
			bins.binid[binindex] = binindex;
		}
		else
		{
			binindex = particles.binid[gidx];
			binindex_left = particles.binid[max(gidx-1,0)];
			binindex_right = particles.binid[min((gidx+1),(nptcls-1))];

			if(binindex_left != binindex)
			{
				bins.ifirstp[binindex] = gidx;
				bins.binid[binindex] = binindex;
			}

			if(binindex_right != binindex)
			{
				bins.nptcls[binindex] = gidx+1;
				bins.binid[binindex] = binindex;
			}

		}




		gidx += blockDim.x*gridDim.x;
	}
}

__global__
void count_bin_ptcls(Particlebin bins,int nbins)
{
	int idx = threadIdx.x;
	int gidx = idx+blockIdx.x*blockDim.x;


	while(gidx < nbins)
	{
		bins.nptcls[gidx] = bins.nptcls[gidx]-bins.ifirstp[gidx];

		gidx += blockDim.x*gridDim.x;
	}
}

__global__
void find_cell_index_kernel(NeutronList particles,SimulationData sim)
{
	int idx = threadIdx.x;
	int gidx = idx + blockDim.x*blockIdx.x;

	while(gidx < particles.nptcls_allocated)
	{
			particles.binid[gidx] = sim.calc_clusterID(particles.px[gidx],particles.py[gidx]);

		//	if(gidx >= 8388000)
			//	printf("particle %i is in bin %i\n",gidx,particles.binid[gidx]);
			gidx += blockDim.x*gridDim.x;
	}
}

__global__
void reorder_particle_data(float* odata, float* idata,int* index_array,int nptcls)
{
	int idx = threadIdx.x;
	int gidx = idx + blockIdx.x*blockDim.x;



	while(gidx < nptcls)
	{
		int ogidx = index_array[gidx];
		//printf("particle %i now in slot %i\n",ogidx,gidx);

		odata[gidx] = idata[ogidx];

		gidx += blockDim.x*gridDim.x;
	}
}

__host__
void NeutronList::sort(SimulationData* simulation)
{
	int cudaBlockSize = 512;
	int cudaGridSize = 14*4;


	CUDA_SAFE_KERNEL((find_cell_index_kernel<<<cudaGridSize,cudaBlockSize>>>
								 (*this,*simulation)));

	CUDA_SAFE_KERNEL((write_xpindex_array<<<cudaGridSize,cudaBlockSize>>>
								 (pindex,nptcls)));

	// wrap raw device pointers with a device_ptr
	thrust::device_ptr<short int> thrust_keys(binid);
	thrust::device_ptr<int> thrust_values(pindex);

	// Sort the data
	thrust::sort_by_key(thrust_keys,thrust_keys+nptcls,thrust_values);
	cudaDeviceSynchronize();

	for(int i=0;i<NeutronList_nfloats-1;i++)
	{
		float* idata =*(get_float_ptr(i));
		float* odata = buffer;

		CUDA_SAFE_KERNEL((reorder_particle_data<<<cudaGridSize,cudaBlockSize>>>
									 (odata,idata,pindex,nptcls)));
		cudaDeviceSynchronize();
		*(get_float_ptr(i)) = odata;
		buffer = idata;

	}

	for(int i=0;i<NeutronList_nints-1;i++)
	{
		float* idata =  (float*)*(get_int_ptr(i));
		float* odata = buffer;

		CUDA_SAFE_KERNEL((reorder_particle_data<<<cudaGridSize,cudaBlockSize>>>
									 (odata,idata,pindex,nptcls)));
		cudaDeviceSynchronize();
		*(get_int_ptr(i)) = (int*)odata;
		buffer = idata;

	}

	int nbins = (simulation->nclusters_x) * (simulation->nclusters_y);

	CUDA_SAFE_CALL(cudaMemset((simulation->bins.ifirstp),0,nbins*sizeof(int)))
	CUDA_SAFE_CALL(cudaMemset((simulation->bins.nptcls),0,nbins*sizeof(int)))

	// Find the cell-bin boundaries in the particle list
	CUDA_SAFE_KERNEL((find_bin_boundaries<<<cudaGridSize,cudaBlockSize>>>
								 (*this,simulation->bins)));



	// Find the number of particles in each bin
	CUDA_SAFE_KERNEL((count_bin_ptcls<<<cudaGridSize,cudaBlockSize>>>
								 (simulation->bins,nbins)));

	// use exclusive scan to get the first particle in each bin
	CUDA_SAFE_CALL(cudaMemcpy((simulation->bins.ifirstp),(simulation->bins.nptcls),
			nbins*sizeof(int),cudaMemcpyDeviceToDevice));

	thrust::device_ptr<int> nptcls_t((simulation->bins.ifirstp));
	thrust::exclusive_scan(nptcls_t,nptcls_t+nbins,nptcls_t);


}






















