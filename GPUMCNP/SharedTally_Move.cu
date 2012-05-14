#include "MCNeutron_random.inl"
#include "gpumcnp.inl"

__global__
void SharedTally_Move_kernel(
	NeutronList 			neutrons,
	NeutronList			neutrons_next,
	SimulationData 		simulation,
	curandState* 			random_state_g,
	int						qScattering)
{
	int idx = threadIdx.x;
	int bidx =blockDim.x*blockIdx.x;
	int gidx = idx + bidx;
	int thid = gidx;
	int qglobal = 0;

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


			//printf("iter = %i\n",iter);
			// Advance the neutron
			neutron_next = neutron.Advance(simulation,
															distance,
															qglobal);


			// Write Local data back to global memory
			neutrons_next = neutron_next;
			neutrons = neutron;
		}
		else
		{
			neutrons_next.dead[gidx] = 1;
			neutrons_next.weight[gidx] = 0.0;
		}


		gidx += blockDim.x*gridDim.x;

	}


	// Need to save the random state
	random_state_g[thid] = random_state;


}


void SharedTally_Move(
	SimulationData*			simulation,
	NeutronList*				neutrons,
	NeutronList*				neutrons_next,
	curandState*				random_states,
	int							qScattering
	)
{
	int cudaGridSize;
	int cudaBlockSize;
	int nptcls_return, nptcls_dead;
	int niter_subcycle = 1;

	// Advance the neutrons
	cudaBlockSize = 256;
	cudaGridSize = 1536/cudaBlockSize * 14;

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	CUDA_SAFE_KERNEL((SharedTally_Move_kernel<<<cudaGridSize,cudaBlockSize>>>(
			*neutrons,*neutrons_next,*simulation,random_states,qScattering)));

	cudaDeviceSynchronize();


}
