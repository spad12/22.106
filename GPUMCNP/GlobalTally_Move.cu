#include "MCNeutron_random.inl"
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
	int qglobal = 1;

	curandState random_state = random_state_g[thid];
	MCNeutron neutron(&gidx,&random_state);
	MCNeutron neutron_next(&gidx,&random_state);

	float distance;
	__syncthreads();


	while(gidx < neutrons.nptcls_allocated)
	{

		if((neutrons.dead[gidx]==0))
		{
			neutron = neutrons;

			int iter = 0;


			while(iter < niter_subcycle)
			{
				//printf("iter = %i\n",iter);
				// Advance the neutron
				neutron_next = neutron.Advance(simulation,
																distance,
																qglobal);

				neutron_next.Tally(simulation,neutron,distance);

			//	if((neutron_next.position().x != neutron_next.px)||(neutron_next.position().y != neutron_next.py))
				//	printf("Warning position() method failed\n");

				// Apply Periodic Boundary condition
				simulation.PeriodicBoundary(neutron_next.px,neutron_next.py,
						neutron_next.mDomain,
						neutron_next.binid);



				neutron_next.check_domain(simulation,neutron);


				// Russian Roulette
				neutron_next.RussianRoulette(neutrons.weight_avg,neutrons.weight_low);



				// if the neutron is alive, then iter is only incremented by 1
				iter += 1+neutron_next.dead*niter_subcycle;



				neutron = neutron_next;
			}


			if(neutron_next.dead != 0)
				neutron_next.weight = 0;

			neutron = neutron_next;


			// Write Local data back to global memory
			neutrons_next = neutron_next;
			neutrons = neutron_next;
		}
		else
		{
			neutrons_next.dead[gidx] = 1;
			neutrons_next.weight[gidx] = 0.0;
			neutrons.weight[gidx] = 0.0;
		}


		gidx += blockDim.x*gridDim.x;

	}


	// Need to save the random state
	random_state_g[thid] = random_state;


}

extern "C" __host__
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
		niter_subcycle = 1;
	}
	else
	{
		// Just do a ton of steps
		niter_subcycle = 1;
	}

	// Advance the neutrons
	cudaBlockSize = 256;
	cudaGridSize = 1536/cudaBlockSize * 14;

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	CUDA_SAFE_KERNEL((GlobalTally_Move_kernel<<<cudaGridSize,cudaBlockSize>>>(
			*neutrons,*neutrons_next,*simulation,random_states,qScattering,niter_subcycle)));

	cudaDeviceSynchronize();


}





















