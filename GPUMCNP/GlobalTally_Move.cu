#include "gpumcnp.inl"



__global__
void GlobalTally_Move_kernel(
	NeutronList 			neutrons,
	NeutronList			neutrons_next,
	SimulationData 		simulation,
	curandState* 			random_state_g,
	int						qScattering,
	int 						niter_subcycle)
{
	int idx = threadIdx.x;
	int bidx =blockDim.x*blockIdx.x;
	int gidx = idx + bidx;
	int thid = gidx;

	curandState random_state = random_state_g[thid];
	MCNeutron neutron(&gidx,&random_state);
	MCNeutron neutron_next(&gidx,&random_state);

	float distance;
	__syncthreads();


	while(gidx < neutrons.nptcls_allocated)
	{

		if((!neutrons.dead[gidx]))
		{
			neutron = neutrons;

			int iter = 0;


			while(iter < niter_subcycle)
			{
				//printf("iter = %i\n",iter);
				// Advance the neutron
				neutron_next = neutron.Advance(simulation,
																distance,
																qScattering);

				neutron_next.Tally(simulation,neutron,distance);

				// Russian Roulette
				neutron_next.RussianRoulette(neutrons.weight_avg,neutrons.weight_low);

				// Apply Periodic Boundary condition
				simulation.PeriodicBoundary(neutron_next.position(),
						neutron_next.mDomain,
						neutron_next.binid);

				// if the neutron is alive, then iter is only incremented by 1
				iter += 1+neutron_next.dead*niter_subcycle;

				neutron = neutron_next;
			}

			// Write Local data back to global memory
			neutrons_next = neutron_next;
		}



		gidx += blockDim.x*gridDim.x;

	}


	// Need to save the random state
	random_state_g[thid] = random_state;


}

__host__
void GlobalTally_Move(
	SimulationData*			simulation,
	NeutronList*				neutrons,
	NeutronList*				neutrons_next,
	curandState*				random_states,
	int							qRefillList,
	int							qScattering
	)
{
	int cudaGridSize;
	int cudaBlockSize;
	int nptcls_return, nptcls_dead;
	int niter_subcycle = 100;

	if(qRefillList)
	{
		// We want to refill the particle list after every step
		// so we are only going to take 1 subcycle step
		// and then refill
		niter_subcycle = 100;
	}
	else
	{
		// Just do a ton of steps
		niter_subcycle = 10000;
	}

	// Advance the neutrons
	cudaBlockSize = 256;
	cudaGridSize = 1536/cudaBlockSize * 14;

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	CUDA_SAFE_KERNEL((GlobalTally_Move_kernel<<<cudaGridSize,cudaBlockSize>>>(
			*neutrons,*neutrons_next,*simulation,random_states,qScattering,niter_subcycle)));

	cudaDeviceSynchronize();


}





















