#include "gpumcnp.inl"
#include "curand_kernel.h"

__global__
void random_init(curandState* random_states,int seed, int nstates)
{
	int gidx = threadIdx.x+blockDim.x*blockIdx.x;

	if(gidx < nstates)
	{
		curand_init(seed,gidx,gidx,(random_states+gidx));
	}
}

int main(void)
{
	cudaSetDevice(1);
	int qGlobalTallies = 1;
	int qScattering = 0;
	int qRefillList = 0;

	int nptcls = 10.0e6;
	int nx = 512;
	int ny = 512;
	int nE = 2;

	int seed = 209384756;

	float xdim = 20.0; // cm
	float ydim = 20.0; // cm
	float radius = 10.0; // cm
	float emin = 0.1; // ev
	float emax = 1.0e6; // ev
	float TimeStep = 1.0; // seconds

	float weight_avg = 1.0f;
	float weight_low = 0.1;

	/********************************/
	/********************************/
	/* Setup the simulation data */

	printf("Setting Up Simulation Data\n");

	SimulationData simulation;
	simulation.allocate(nx,ny,nE);
	simulation.setup(xdim,ydim,radius,emin,emax,TimeStep);


	/********************************/
	/********************************/
	/* Setup the neutron data */
	printf("Setting Up Neutron Data\n");
	NeutronList neutrons(nptcls);
	NeutronList neutrons_next(nptcls);

	neutrons.weight_avg = weight_avg;
	neutrons.weight_low = weight_low;
	neutrons_next.weight_avg = weight_avg;
	neutrons_next.weight_low = weight_low;
	/********************************/
	/********************************/
	/* Setup the random numbers */
	curandState* random_states;

	CUDA_SAFE_CALL(cudaMalloc((void**)&random_states,512*6*14*sizeof(curandState)));

	// Initialize the random number generators
	int cudaGridSize = 6*14;
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

	printf("Entering Super Step\n");

	cutStartTimer(timer);
	gpumcnp_super_step(&simulation,
			&neutrons,&neutrons_next,
			random_states,qGlobalTallies,
			qRefillList,qScattering);
	cutStopTimer(timer);

	elapsed_time = cutGetTimerValue(timer);
	printf("Super step took %f ms\n",elapsed_time);

	float* xvals = (float*)malloc((nx+1)*(ny+1)*sizeof(float));
	float* yvals = (float*)malloc((nx+1)*(ny+1)*sizeof(float));
	float* plotvals = (float*)malloc((nx+1)*(ny+1)*(nE+1)*sizeof(float));

	simulation.flux_tally.cudaMatrixcpy(plotvals,cudaMemcpyDeviceToHost);

	for(int i=0;i<(nx+1);i++)
	{
		for(int j=0;j<ny+1;j++)
		{
			xvals[i] = -xdim + (2*xdim)/((float)nx)*i;
			yvals[j] = -ydim + (2*ydim)/((float)ny)*j;
		}
	}

	gnuplot_ctrl* plot;

	plot = gnuplot_init();

	gnuplot_cmd(plot,"set xlabel \"X\"");
	gnuplot_cmd(plot,"set ylabel \"Y\"");


	gnuplot_plot_xyz(plot,xvals,yvals,plotvals,(nx+1),(ny+1),"");


	printf("Press 'Enter' to continue\n");
	getchar();
	gnuplot_cmd(plot,"set size square");
	gnuplot_cmd(plot,"set pm3d map");
	gnuplot_cmd(plot,"set samples 50; set isosamples 100");


	gnuplot_save_pdf(plot,"flux_profile");
	gnuplot_close(plot);



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
	int isteps_max = 10;

	/********************************/
	/*	Populate Empty particle slots
	 * using source distribution in simulation
	 * source distribution is either from a previous step
	 * or provided for first step
	 */

	Populate_NeutronList(simulation,neutrons,random_states,PointSource(0,0,0.5));


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

	while(nptcls_left > 1000)
	{
		// Move the neutrons depending on the call flags
		if(qGlobalTallies)
		{ /* Global tallies during the move */

			GlobalTally_Move(simulation,neutrons,neutrons_next,random_states,qRefillList,qScattering);
		}
		else
		{ /* Going to do tallies in shared memory, so different move */

			SharedTally_Move(simulation,neutrons,neutrons_next,random_states,qScattering);

			// Do tallies in shared memory
			SharedTally(simulation,neutrons,neutrons_next);

		}

		// Swap the neutrons at i+1 with the ones at i.
		NeutronList*	temp = neutrons;
		neutrons = neutrons_next;
		neutrons_next = temp;

		// From here on out we are operating on the i+1 list, neutrons
		// neutrons_next is now the i+2 list.



		if(qRefillList)
		{ /* Split the remaining particles to fill up the empty slots */
			Refill_ParticleList(neutrons);
		}
		else
		{
			// Still need to count up the number of particles that are alive
			neutrons->nptcls -= neutrons->CountDeadNeutrons();
		}


		if(!qGlobalTallies)
		{ /* Update the particle binid's and sort the particle list */

			neutrons->sort(simulation);

		}

		nptcls_left = neutrons->nptcls;

		isteps++;

		if(isteps >= isteps_max)
			break;

	}


}
