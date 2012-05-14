
#include "curand_kernel.h"
#include "curand.h"
#include "gpumcnp.inl"

int myid_g;

extern "C" void gpumcnp_setup(
		SimulationData**			simulation_out,
		NeutronList**				neutrons_out,
		NeutronList**				neutrons_next_out,
		float						xdim,
		float						ydim,
		float						radius,
		float						emin,
		float						emax,
		float						TimeStep,
		float						weight_avg,
		float						weight_low,
		int							nptcls,
		int							nx,
		int							ny,
		int							nE,
		int 							myid)
{

	myid_g = myid;
	/********************************/
	/* Setup the simulation data */

	//printf("Setting Up Simulation Data\n");

	SimulationData simulation;
	simulation.allocate(nx,ny,nE);
	simulation.setup(xdim,ydim,radius,emin,emax,TimeStep);

	float cxA[5*4];
	float cxS[5*4];


	for(int i=0;i<5;i++)
	{
		cxA[4*i] =4.05;
		cxA[4*i+1] =10.05;
		cxA[4*i+2] = 10.05;
		cxA[4*i+3] = 0.05;

		cxS[4*i] = 0.5;
		cxS[4*i+1] = 2.0;
		cxS[4*i+2] = 2.0;
		cxS[4*i+3] = 2.0;

	}

	cxA[16] = 2.5;
	cxA[17] = 0.5;
	cxA[18] = 1.0;
	cxA[19] = 1.0e-2;

	cxS[16] = 1.5;
	cxS[17] = 2.5;
	cxS[18] = 6.0;
	cxS[19] = 5.5;

	for(int i=0;i<4;i++)
	{
		cxA[4*i] = 2.5;
		cxA[4*i+1] = 0.25;
		cxA[4*i+2] = 2.5e-2;
		cxA[4*i+3] = 0.25;

		cxS[4*i] = 1.5;
		cxS[4*i+1] = 1.5;
		cxS[4*i+2] = 2.0;
		cxS[4*i+3] = 2.5;

	}

	for(int i=1;i<3;i++)
	{
		cxA[4*i] = 1.5;
		cxA[4*i+1] = 5.0e-2;
		cxA[4*i+2] = 5.0e-1;
		cxA[4*i+3] = 5.0e-2;

		cxS[4*i] = 0.5;
		cxS[4*i+1] = 2.0;
		cxS[4*i+2] = 1.5;
		cxS[4*i+3] = 1;

	}







	simulation.sigmaA.cudaMatrixcpy(cxA,cudaMemcpyHostToDevice);
	simulation.sigmaES.cudaMatrixcpy(cxS,cudaMemcpyHostToDevice);

	/********************************/
	/********************************/
	/* Setup the neutron data */
	//printf("Setting Up Neutron Data\n");
	NeutronList neutrons(nptcls);
	NeutronList neutrons_next(nptcls);

	neutrons.weight_avg = weight_avg;
	neutrons.weight_low = weight_low;
	neutrons_next.weight_avg = weight_avg;
	neutrons_next.weight_low = weight_low;
	/********************************/

	*neutrons_out = (NeutronList*)malloc(sizeof(NeutronList));
	*neutrons_next_out = (NeutronList*)malloc(sizeof(NeutronList));
	*simulation_out = (SimulationData*)malloc(sizeof(SimulationData));

	**neutrons_out = neutrons;
	**neutrons_next_out = neutrons_next;
	**simulation_out = simulation;


}

extern "C" void SetGPU(int id)
{
	cudaSetDevice(id);
}

__global__
void random_init(curandState* random_states,int seed, int nstates)
{
	int gidx = threadIdx.x+blockDim.x*blockIdx.x;

	if(gidx < nstates)
	{
		curand_init(seed+gidx,gidx+1,gidx,(random_states+gidx));
	}
}

extern "C" void gpumcnp_run(
	SimulationData*			simulation,
	NeutronList*				neutrons,
	NeutronList*				neutrons_next,
	float*						plotvals,
	float*						plotvals2,
	float*						time_out,
	int							qGlobalTallies,
	int							qRefillList,
	int							qScattering,
	int							seed)
{
	/********************************/
	/* Setup the random numbers */
	curandState* random_states;
	CUDA_SAFE_CALL(cudaMalloc((void**)&random_states,512*33*33*sizeof(curandState)));



	// Initialize the random number generators
	int cudaGridSize = 33*33;
	int cudaBlockSize = 512;
	int nthreads = cudaGridSize*cudaBlockSize;
	CUDA_SAFE_KERNEL((random_init<<<cudaGridSize,cudaBlockSize>>>
								 (random_states,seed,nthreads)));
	/********************************/
	/********************************/
	/* Setup timing stuff */
	uint timer;
	float elapsed_time = 0.0;
	cutCreateTimer(&timer);


	simulation->flux_tally.cudaMatrixSet(0);
	simulation->flux_tally2.cudaMatrixSet(0);

	if(myid_g == 0)
	printf("Entering Super Step\n");

	cutStartTimer(timer);
	gpumcnp_super_step(simulation,
			neutrons,neutrons_next,
			random_states,qGlobalTallies,
			qRefillList,qScattering);
	cutStopTimer(timer);

	simulation->flux_tally.cudaMatrixcpy(plotvals,cudaMemcpyDeviceToHost);
	simulation->flux_tally2.cudaMatrixcpy(plotvals2,cudaMemcpyDeviceToHost);

	elapsed_time = cutGetTimerValue(timer);

	if(myid_g == 0)
	printf("Super step took %f ms\n",elapsed_time);

	*time_out = elapsed_time;

	cutDeleteTimer(timer);

	cudaFree(random_states);




}














/*  Primary MC subroutine. Move, Tally, repeat
 *
 */

void gpumcnp_super_step(
	SimulationData* 		simulation,
	NeutronList* 				neutrons,
	NeutronList* 				neutrons_next,
	curandState*				random_states,
	int 							qGlobalTallies,
	int							qRefillList,
	int							qScattering
	)
{
	int nptcls_left;
	int isteps = 0;
	int isteps_max = 100;

	if(!qGlobalTallies)
		qRefillList = 1;



	/********************************/
	/*	Populate Empty particle slots
	 * using source distribution in simulation
	 * source distribution is either from a previous step
	 * or provided for first step
	 */

	Populate_NeutronList(simulation,neutrons,random_states,PointSource(0,0,9.0e2));


	/********************************/

	/********************************/
	/* Sub-stepping loop
	 * 1) Move the neutrons
	 * 2) Do Tallies (inside move for qGlobalTallies==True)
	 * 4) Refill by splitting if qRefillList flag is step
	 * 5) Update neutron bin id's if we are doing sorted tallies
	 * 6) Sort the neutrons according to binid
	 */

	nptcls_left = neutrons->nptcls;
	neutrons_next->nptcls = nptcls_left;

	if(myid_g == 0)
		printf("\nStep: ");

	while(nptcls_left > 1024)
	{
		if(myid_g == 0)
			printf("%i ",isteps);

		// Move the neutrons depending on the call flags
		if(qGlobalTallies)
		{ /* Global tallies during the move */

			GlobalTally_Move(simulation,neutrons,neutrons_next,random_states,qRefillList,qScattering);


		}
		else
		{
			 /* Update the particle binid's and sort the particle list */
			neutrons->sort(simulation);
			/* Going to do tallies in shared memory, so different move */

			SharedTally_Move(simulation,neutrons,neutrons_next,random_states,qScattering);

			// Do tallies in shared memory
			SharedTally(simulation,neutrons,neutrons_next,random_states);

		}


		// Swap the neutrons at i+1 with the ones at i.
		NeutronList	temp = *neutrons;
		*neutrons = *neutrons_next;
		*neutrons_next = temp;

		// From here on out we are operating on the i+1 list, neutrons
		// neutrons_next is now the i+2 list.





		if(qRefillList)
		{ /* Split the remaining particles to fill up the empty slots */
			Refill_ParticleList(neutrons,isteps);
		}
		else
		{
			// Still need to count up the number of particles that are alive
			neutrons->nptcls = neutrons->nptcls_allocated-neutrons->CountDeadNeutrons(isteps);

		}





		nptcls_left = neutrons->nptcls;
		neutrons_next->nptcls = nptcls_left;

		//printf("nptcls left at step %i = %i \n",isteps,nptcls_left);

		isteps++;

		if(isteps >= isteps_max)
			break;

	}


}
