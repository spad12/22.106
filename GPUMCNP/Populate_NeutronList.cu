#include "gpumcnp.inl"

template void Populate_NeutronList<PointSource>(
		SimulationData*			simulation,
		NeutronList*				neutrons,
		curandState*				random_states,
		PointSource				source);


template<class SourceObject>
__global__
void Populate_NeutronList_kernel(
	SimulationData						simulation,
	NeutronList							neutrons,
	curandState*							random_states_g,
	SourceObject						source)
{
	int gidx = threadIdx.x+blockIdx.x*blockDim.x;

	curandState random_state = random_states_g[gidx];
	MCNeutron neutron(&gidx,&random_state);

	while(gidx < neutrons.nptcls_allocated)
	{

		// calculate the source of the neutron
		// from the inputed source function
		source.SourceFunction(simulation,neutron);

		// Save the neutron to global memory
		neutrons = neutron;

		gidx += blockDim.x*gridDim.x;
	}

	// Save the random state
	random_states_g[threadIdx.x+blockIdx.x*blockDim.x] = random_state;
}


template<class SourceObject>
__host__
void Populate_NeutronList(
	SimulationData*			simulation,
	NeutronList*				neutrons,
	curandState*				random_states,
	SourceObject			source
)
{

	int cudaGridSize;
	int cudaBlockSize;

	cudaBlockSize = 256;
	cudaGridSize = 1536/cudaBlockSize * 14;

	CUDA_SAFE_KERNEL((Populate_NeutronList_kernel<<<cudaGridSize,cudaBlockSize>>>(
			*simulation,*neutrons,random_states,source)));

	neutrons->nptcls = neutrons->nptcls_allocated;

}
