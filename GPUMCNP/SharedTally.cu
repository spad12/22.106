#include "MCNeutron_random.inl"
#include "gpumcnp.inl"


__global__
void SharedTally_kernel(
	NeutronList 			neutrons,
	NeutronList			neutrons_next,
	SimulationData 		simulation,
	curandState* 			random_state_g,
	int						nbins,
	int 						subbin_size)
{
	int idx = threadIdx.x;
	int bidx =blockDim.x*blockIdx.x;
	int gidx = idx + bidx;
	int thid = idx;
	int ridx;

	ridx = gidx;



	if(simulation.bins.nptcls[blockIdx.x]>0)
	{
		while(ridx>= (33*33*512))
			ridx -= blockDim.x*gridDim.x;

		__shared__ int3 bin;

		__shared__ float flux_tally[6140];
		__shared__ float flux_tally2[6140];


		if(idx == 0){
			bin.x = simulation.bins.ifirstp[blockIdx.x];
			bin.y = simulation.bins.nptcls[blockIdx.x];
			bin.z = simulation.bins.binid[blockIdx.x];


			//printf("bin %i, starts at %i with %i ptcls\n",blockIdx.x,bin.x,bin.y);

		}


		while(thid <= subbin_size)
		{
			flux_tally[thid] = 0;
			flux_tally2[thid] = 0;
			thid+= blockDim.x;
		}

		thid = idx;

		curandState random_state = random_state_g[ridx];

		MCNeutron neutron(&gidx,&random_state);
		MCNeutron neutron_next(&gidx,&random_state);

		float distance;
		__syncthreads();




		if(bin.y > 0)
		{
			idx = threadIdx.x;



			gidx = bin.x+idx;
			while((gidx < (bin.x+bin.y))&&(gidx < neutrons.nptcls_allocated))
			{

				if((neutrons.dead[gidx]==0))
				{
					neutron = neutrons;
					neutron_next = neutrons_next;



					//if(neutron.binid != bin.z)
						//printf("Warning particle %i bin id %i != %i\n",gidx,neutron.binid,bin.z);


					neutron_next.STally(simulation,neutron,flux_tally,flux_tally2,distance);

				//	if((neutron_next.position().x != neutron_next.px)||(neutron_next.position().y != neutron_next.py))
					//	printf("Warning position() method failed\n");

					// Apply Periodic Boundary condition
					simulation.PeriodicBoundary(neutron_next.px,neutron_next.py,
							neutron_next.mDomain,
							neutron_next.binid);



					neutron_next.check_domain(simulation,neutron);


					// Russian Roulette
					neutron_next.RussianRoulette(neutrons.weight_avg,neutrons.weight_low);



					if(neutron_next.dead == 1)
						neutron_next.weight = 0;

					// Write Local data back to global memory
					neutrons = neutron_next;
					neutrons_next = neutron_next;
				}
				else
				{
					neutrons_next.dead[gidx] = 1;
					neutrons_next.weight[gidx] = 0.0;
				}


				gidx += blockDim.x;
			}


			__syncthreads();
			// Write the tallies in shared memory to global memory
			simulation.AtomicAdd_StoG(flux_tally,flux_tally2,bin.z);

			// Need to save the random state
			random_state_g[ridx] = random_state;
		}
	}

}


extern "C" void SharedTally(
	SimulationData*			simulation,
	NeutronList*				neutrons,
	NeutronList*				neutrons_next,
	curandState* 				random_states)
{
	dim3 cudaGridSize(1,1,1);
	dim3 cudaBlockSize(512,1,1);
	// Tallies
	cudaGridSize.x = simulation->nbins;
	cudaGridSize.y = 1;

	int subbinsize = ((simulation->nghost_cluster+simulation->CellspClusterx)
			*(simulation->nghost_cluster+simulation->CellspClustery));

	CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

	CUDA_SAFE_KERNEL((SharedTally_kernel<<<cudaGridSize,cudaBlockSize>>>(
			*neutrons,*neutrons_next,*simulation,random_states,simulation->nbins,subbinsize)));



}
