#include "gpumcnp.inl"



__global__
 void Plist_refill(NeutronList neutrons,int* ialive, int* idead,
		int nsplit_global,int nalive, int ndead, int nstride,int nextra)
{
	int idx = threadIdx.x;
	int gidx = idx + blockIdx.x*blockDim.x;
	int nthreads_total = blockDim.x*gridDim.x;

	int myid_in, myid_out;
	int ix,iy,nsplit;
	bool iextra = false;


	if(gidx < nalive)
	{
		iextra = false;
		iy = gidx/nstride;
		ix = gidx - nstride*iy;

		iextra = ((ix==0)&&(iy<nextra)) ? 1:0;
		nsplit = (iextra) ? nsplit_global+1:nsplit_global;
		myid_in = ialive[gidx];

//		nsplit = 1;
		int gidx2 = gidx;
/*		while(gidx2 < ndead)
		{
			nsplit += 1;
			gidx2 += nalive;
		}
*/




		if(nsplit > 1)
		{
			MCNeutron neutron;
			neutron.thid = &myid_in;

			// Read in the parent neutron
			neutron = neutrons;

			// Split the weight
			neutron.weight /= ((float)nsplit);

			if(neutron.dead != 0)
				printf("warning parent neutron %i is dead\n",myid_in);

			// Write the parent back to its original place with the updated weight
			neutrons = neutron;

			// Set the local temporary neutron's id to myid_out
			neutron.thid = &myid_out;


			gidx2 = gidx;

			while(gidx2 < ndead-nextra)
			{
				myid_out = idead[gidx2];

				// Write out the neutron data
				neutrons = neutron;


				gidx2 += nalive;
			}


			gidx2 = ndead-nextra+iy;
			if((iextra))
			{

				myid_out = idead[gidx2];

				// Write out the neutron data
				neutrons = neutron;
			}


		}

		gidx += blockDim.x*gridDim.x;
	}

}


__global__
void condense_index(int* itrue,int* ifalse,int* scan_data,int n_elements)
{
	// Copy data from data_in to a new location in data_out if condition is true
	int idx = threadIdx.x;
	int gidx = blockIdx.x*blockDim.x+idx;



	while(gidx < n_elements)
	{
		int oidxm = 0;
		int oidx = scan_data[gidx];

		if(gidx > 0)
			oidxm = scan_data[gidx-1];

		if(oidx != oidxm)
		{

			itrue[oidx-1] = gidx;
		}
		else
			ifalse[gidx-oidx] = gidx;

		gidx += blockDim.x*gridDim.x;
	}
}

void Refill_ParticleList(
	NeutronList*				neutrons,
	int istep
	)
{
	// Refill the particle list by splitting the particles
	neutrons->refill(istep);


}

void NeutronList::refill(int istep)
{
	int nptcls_dead;
	int nptcls_alive;
	int nptcls_split;
	int nptcls_stride;
	int nsplits_avg,ndead_extra;
	int *idead,*ialive;
	float weight_original, weight_new;



	int cudaGridSize;
	int cudaBlockSize;

	cudaBlockSize = 512;
	cudaGridSize = (nptcls_alive+cudaBlockSize-1)/cudaBlockSize;


	thrust::device_ptr<float> weight_t(buffer);

	// for debugging purposes, calculate the current total weight
	CUDA_SAFE_CALL(cudaMemcpy(buffer,weight,nptcls*sizeof(float),cudaMemcpyDeviceToDevice));
	weight_original = thrust::reduce(weight_t,weight_t+nptcls);
	//printf("weight = %f\n",weight_original);



	// Use the buffer as temporary storage for scan
	// with a simple type re-cast
	int* offsets = (int*)buffer;
	CUDA_SAFE_CALL(cudaMemcpy(offsets,dead,nptcls*sizeof(int),cudaMemcpyDeviceToDevice));

	thrust::device_ptr<int> offests_t(offsets);

	thrust::inclusive_scan(offests_t,offests_t+nptcls,offests_t);

	// The last element of the scan is the sum of all the dead.
	CUDA_SAFE_CALL(cudaMemcpy(&nptcls_dead,offsets+nptcls-1,sizeof(int),cudaMemcpyDeviceToHost));

	nptcls_alive = nptcls-nptcls_dead;

	if(nptcls_alive < 1024)
	{
		// Return if the number of particles alive == 0;
		nptcls = nptcls_alive;
		return;
	}

	//printf("We have %i particles alive, %i dead, out of %i original\n",nptcls_alive,nptcls_dead,nptcls);

	// Calculate the total number of splits for every particle, and the extra splits
	nsplits_avg = (nptcls_dead+nptcls_alive-1)/nptcls_alive;

	ndead_extra = nptcls_dead%nptcls_alive;


	// We are using a stride distance so that we don't split one end of the list more than the other
	// I am trying to evenly distribute the splitting by treating the particle list as a 2D array
	// the stride distance is the number of alive indices that are skipped
	// for every one that is taken.
	nptcls_stride = nptcls_alive/max(ndead_extra,1);

	//printf("nsplits = %i, ndead_extra = %i, nstride = %i\n",nsplits_avg,ndead_extra,nptcls_stride);

	// Allocate memory for compact index arrays.
	// Should probably just be statically allocated
	//CUDA_SAFE_CALL(cudaMalloc((void**)&idead,nptcls_dead*sizeof(int)));
	//CUDA_SAFE_CALL(cudaMalloc((void**)&ialive,nptcls_alive*sizeof(int)));

	idead = pindex;
	ialive = pindex+nptcls_dead;

	// Compact the alive and dead indices
	cudaGridSize = (nptcls+cudaBlockSize-1)/cudaBlockSize;
	CUDA_SAFE_KERNEL((condense_index<<<cudaGridSize,cudaBlockSize>>>
			(idead,ialive,offsets,nptcls)));

	cudaGridSize = (nptcls_alive+cudaBlockSize-1)/cudaBlockSize;
	CUDA_SAFE_KERNEL((Plist_refill<<<cudaGridSize,cudaBlockSize>>>
			(*this,ialive,idead,nsplits_avg,nptcls_alive,nptcls_dead,nptcls_stride,ndead_extra)));

	// for debugging purposes, calculate the new total weight
	//CUDA_SAFE_CALL(cudaMemcpy(buffer,weight,nptcls_allocated*sizeof(float),cudaMemcpyDeviceToDevice));
	//weight_new = thrust::reduce(weight_t,weight_t+nptcls_allocated);

	step_weights[imethod_step][istep] = weight_original/((float)nptcls_allocated);
	step_nptcls[imethod_step][istep] = 1.0*nptcls_alive/((float)nptcls_allocated);

	//printf("Change in total weight = %f\n",weight_original-weight_new);

	nptcls = nptcls_allocated;

	//weight_avg = 2.0*weight_new/((float)nptcls);







}
