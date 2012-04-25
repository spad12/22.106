#include "problem_definitions.cuh"

__global__
void random_init(curandState* random_states,int seed, int nstates)
{
	int gidx = threadIdx.x+blockDim.x*blockIdx.x;

	if(gidx < nstates)
	{
		curand_init(seed,gidx,gidx,random_states+gidx);
	}
}

template<class op>
__global__
void neutron_init(NeutronList neutrons, op source,
							 curandState* random_state_g,float new_weight,int offset)
{
	int idx = threadIdx.x;
	int gidx = idx + blockDim.x*blockIdx.x;

	curandState random_state = random_state_g[gidx];

	while((gidx+offset) < neutrons.nptcls_allocated)
	{

		float4 source_values = source(&random_state);

		neutrons.px[gidx] = source_values.x;
		neutrons.mu[gidx] = source_values.y;
		neutrons.energy[gidx] = source_values.z;
		neutrons.domain[gidx] = floor(source_values.w);

		neutrons.weight[gidx] = new_weight;
		neutrons.time_done[gidx] = 0;
		neutrons.dead[gidx] = 0;
		neutrons.finished_time[gidx] = 0;


		gidx += blockDim.x*gridDim.x;

	}

	random_state_g[idx+blockDim.x*blockIdx.x] = random_state;

}

__global__
void neutron_advance(NeutronList neutrons, Domain* materials,curandState* random_state_g,
									 float time_max,int niter_subcycle)
{
	int idx = threadIdx.x;
	int bidx =blockDim.x*blockIdx.x;
	int gidx = idx + bidx;
	int thid = gidx;

	__shared__ float flux_tally[n_domains*512];
	__shared__ float absorptions[n_domains*32];
	__shared__ float collisions[n_domains*32];
	__shared__ Domain materials_s[n_domains];

	// Initialize and populate shared variables
	while(idx < n_domains)
	{
		materials_s[idx] = materials[idx];
		materials_s[idx].flux_tally = flux_tally+512*idx;
		materials_s[idx].collisions = collisions+32*idx;
		materials_s[idx].absorptions = absorptions+32*idx;
		idx += blockDim.x;
	}

	idx = threadIdx.x;

	while(idx < n_domains*512)
	{
		flux_tally[idx] = 0;
		idx += blockDim.x;
	}

	idx = threadIdx.x;

	while(idx < n_domains*32)
	{
		collisions[idx] = 0;
		absorptions[idx] = 0;
		idx += blockDim.x;
	}

	curandState random_state = random_state_g[gidx];

	Neutron_data storage;

	Neutron my_neutron(storage,&gidx);
	my_neutron.parentID = &gidx;
	my_neutron.random_state = &random_state;

	__syncthreads();


	while(gidx < neutrons.nptcls)
	{

		if((!neutrons.dead[gidx])&&(neutrons.time_done[gidx]<time_max))
		{
			// Copy neutron data from global memory to registers
			my_neutron = neutrons;

			int iter = 0;


			while(iter < niter_subcycle)
			{

				// Advance the neutron
				my_neutron.advance(materials_s,time_max);
				// Russian Roulette
				my_neutron.russian_roulette();
				// if the neutron is alive, then iter is only incremented by 1
				iter += 1+my_neutron.check_subexit()*niter_subcycle;



			}





			// Write the local data back to global memory
			my_neutron.merge(neutrons);
		}

		gidx += blockDim.x*gridDim.x;

	}


	// Need to save the random state
	random_state_g[threadIdx.x+blockDim.x*blockIdx.x] = random_state;

	__syncthreads();

	// Write tallies back to global memory
	idx = 1;
	while(idx < 3)
	{
		int idy = threadIdx.x;
		while(idy < materials_s[idx].ncells)
		{
			atomicAdd(materials[idx].flux_tally+idy,materials_s[idx].flux_tally[idy]);
			idy += blockDim.x;
		}

		idy = threadIdx.x;
		while(idy < 32)
		{
			atomicAdd(materials[idx].absorptions+idy+32*blockIdx.x,materials_s[idx].absorptions[idy]);
			atomicAdd(materials[idx].collisions+idy+32*blockIdx.x,materials_s[idx].collisions[idy]);
			idy += blockDim.x;
		}


		idx++;
	}





}
template<typename T>
__global__
void scan_condense(T* data_out,T* data_in,int* scan_data,int n_elements)
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

			data_out[oidx-1] = data_in[gidx];
		}
		gidx += blockDim.x*gridDim.x;
	}
}


__host__
void NeutronList::pop_and_append(NeutronList& ListOut, int* condition)
{
	/*
	 * This method uses stream compaction to pop neutrons out of the main list
	 * and append them to ListOut
	 */

	// Get an integer pointer to the buffer. Use the buffer array so that we don't mess anything up
	int* buffer_cond = (int*)buffer;
	int nptcls_new;

	int cudaBlockSize = 256;
	int cudaGridSize = min(42,(nptcls+cudaBlockSize-1)/cudaBlockSize);


	// Copy the condition to the buffer
	CUDA_SAFE_CALL(cudaMemcpy(buffer_cond,condition,nptcls*sizeof(int),cudaMemcpyDeviceToDevice));

	// Do a prefix scan of the condition
	thrust::device_ptr<int> thrust_scan(buffer_cond);
	thrust::inclusive_scan(thrust_scan,thrust_scan+nptcls,thrust_scan);

	// Get the number of neutrons that are going to be appended
	CUDA_SAFE_CALL(cudaMemcpy(&nptcls_new,buffer_cond+nptcls-1,sizeof(int),cudaMemcpyDeviceToHost));

	if(nptcls_new > 0)
	{
		if(nptcls_new+ListOut.nptcls <= ListOut.nptcls_allocated)
		{ // Copy the data over to the new list

			// Do the float data
			for(int i=0;i<NeutronList_nfloats-1;i++)
			{
				float* data_in = *get_float_ptr(i);
				float* data_out = *(ListOut.get_float_ptr(i))+ListOut.nptcls;

				// Could convert this to asynchronous launches after it is debuged
				CUDA_SAFE_KERNEL((scan_condense<<<cudaGridSize,cudaBlockSize>>>
											 (data_out,data_in,buffer_cond,nptcls)));
			}

			// Do the int data
			for(int i=0;i<NeutronList_nints-2;i++)
			{
				int* data_in = *get_int_ptr(i);
				int* data_out = *(ListOut.get_int_ptr(i))+ListOut.nptcls;

				// Could convert this to asynchronous launches after it is debuged
				CUDA_SAFE_KERNEL((scan_condense<<<cudaGridSize,cudaBlockSize>>>
											 (data_out,data_in,buffer_cond,nptcls)));
			}

			// Make sure that ListOut knows how many particles it now has
			ListOut.nptcls += nptcls_new;

		}
		else
		{
			// Too many particles we could extend the list but for now we will just throw and error and exit
			printf("Error, trying to append to many particles %i to list with %i open slots\n"
					"Try running more iterations per subcycle\n",
					nptcls_new,ListOut.nptcls_allocated-ListOut.nptcls);
				exit(EXIT_FAILURE);
		}
	}

}
__global__
void invert_condition(int* condition, int nelements)
{
	int idx = threadIdx.x;
	int gidx = idx+blockDim.x*blockIdx.x;

	while(gidx < nelements)
	{
		condition[gidx] = 1 - condition[gidx];
		gidx += blockDim.x*gridDim.x;
	}
}

__host__
void NeutronList::condense_alive(void)
{
	/*
	 * This method uses stream compaction to condense the neutron list
	 * to a shorter list with only condition == true
	 */

	int nptcls_new;

	int cudaBlockSize = 256;
	int cudaGridSize = min(42,(nptcls+cudaBlockSize-1)/cudaBlockSize);

	int* condition = dead;

	// Change the dead array so that we have 1 for alive and 0 for dead
	CUDA_SAFE_KERNEL((invert_condition<<<cudaGridSize,cudaBlockSize>>>
								 (condition,nptcls)));


	// Do a prefix scan of the condition
	thrust::device_ptr<int> thrust_scan(condition);
	thrust::inclusive_scan(thrust_scan,thrust_scan+nptcls,thrust_scan);

	// Get the number of neutrons that are going to be appended
	CUDA_SAFE_CALL(cudaMemcpy(&nptcls_new,condition+nptcls-1,sizeof(int),cudaMemcpyDeviceToHost));


	if(nptcls_new > 0)
	{
		// Do the float data
		for(int i=0;i<NeutronList_nfloats-1;i++)
		{
			float* data_in = *get_float_ptr(i);
			float* data_out = buffer;

			CUDA_SAFE_KERNEL((scan_condense<<<cudaGridSize,cudaBlockSize>>>
										 (data_out,data_in,condition,nptcls)));

			// Make the old data array the buffer
			buffer = data_in;
			*get_float_ptr(i) = data_out;
		}

		// Do the int data
		for(int i=0;i<NeutronList_nints-2;i++)
		{
			int* data_in = *get_int_ptr(i);
			int* data_out = (int*)buffer;

			CUDA_SAFE_KERNEL((scan_condense<<<cudaGridSize,cudaBlockSize>>>
										 (data_out,data_in,condition,nptcls)));

			// Make the old data array the buffer
			buffer = (float*)data_in;
			*get_int_ptr(i) = data_out;
		}
	}


	// Make sure this list knows how many particles it now has
	nptcls = nptcls_new;

	// Make sure that the dead array gets set back to 0
	if(nptcls_new > 0)
	{
		CUDA_SAFE_CALL(cudaMemset(dead,0,nptcls_allocated*sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(finished_time,0,nptcls_allocated*sizeof(int)));
	}


}

__global__
void clear_slots(NeutronList neutrons,int* condition, int value)
{
	int idx = threadIdx.x;
	int gidx = idx+blockDim.x*blockIdx.x;

	while(gidx < neutrons.nptcls)
	{
		if(condition[gidx] == value)
		{
			neutrons.clear_slot(gidx);
		}

		gidx += blockDim.x*gridDim.x;
	}
}

__host__
void NeutronList::pop_finished(NeutronList& ListOut)
{
	/*
	 * Take the neutrons that have finished their time and not been killed or lost
	 * and put them into the list that we are going to be using for the next time step.
	 * Once these neutrons are removed from this list, make sure that their weights
	 * are 0 and 'dead' flag are set to 1 so that other routines know that they no longer
	 * exist in this list.
	 */
	int cudaBlockSize = 256;
	int cudaGridSize = 84;
	// Pop finished neutrons and append them to ListOut
	pop_and_append(ListOut,finished_time);

	// Clear the slots
	CUDA_SAFE_KERNEL((clear_slots<<<cudaGridSize,cudaBlockSize>>>
								 (*this,finished_time,1)));


}

__host__
void NeutronList::update_average_weight(void)
{
	// We need to preserve the weight, so we'll copy it to the buffer
	float* weight_temp = buffer;
	cudaMemcpy(weight_temp,weight,nptcls*sizeof(float),cudaMemcpyDeviceToDevice);

	thrust::device_ptr<float> weight_sum(weight_temp);
	float weight_total = thrust::reduce(weight_sum,weight_sum+nptcls);

	//weight_avg = 1;
	//weight_avg = max(weight_avg,2.0*weight_low);

}

__global__
void neutron_list_double(NeutronList neutrons)
{
	int idx = threadIdx.x;
	int gidx = idx+blockDim.x*blockIdx.x;

	while(gidx < neutrons.nptcls)
	{

		neutrons.split(gidx);

		gidx += blockDim.x*gridDim.x;
	}
}

__host__
void super_cycle(NeutronList& neutrons_d, Domain* materials,
							NeutronList& next_list_d, curandState* random_states_d,
							float time_max,
							int niter_subcycle, int nptcls_min)
{
	int cudaGridSize = 42;
	int cudaBlockSize = 256;

	int iter = 0;
	while(neutrons_d.nptcls > 0)
	{
		printf("Executing super step %i, nptcls left = %i\n",iter,neutrons_d.nptcls);
	/***********************************************/
	/* Advance the current set of neutrons */

	CUDA_SAFE_KERNEL((neutron_advance<<<cudaGridSize,cudaBlockSize>>>
								 (neutrons_d,materials,random_states_d,
								  time_max,niter_subcycle)));

	/***********************************************/

	/***********************************************/
	/* Remove the neutrons that finished their time */

	neutrons_d.pop_finished(next_list_d);


	/***********************************************/

	/***********************************************/
	/* Condense the neutron list, keep only the alive particles */

	neutrons_d.condense_alive();


	/***********************************************/

	/***********************************************/
	/* Update the average weight of the current list */

	neutrons_d.update_average_weight();


	/***********************************************/

	/***********************************************/
	/* Double the number of particles if we drop
	 * below nptcls_min.
	 * This step is so that we can keep the GPU busy
	 * and accelerate killing off particles that are taking a lot of steps
	 */

	if((neutrons_d.nptcls <= nptcls_min)&&(neutrons_d.nptcls > 0))
	{
	CUDA_SAFE_KERNEL((neutron_list_double<<<cudaGridSize,cudaBlockSize>>>
								 (neutrons_d)));

		int nptcls_new = min((2*(neutrons_d.nptcls))-1,(neutrons_d.nptcls_allocated-1));
		printf("Doubling the number of particles from %i to %i\n",neutrons_d.nptcls,nptcls_new);
		neutrons_d.nptcls = nptcls_new;
	}
	/***********************************************/



		iter++;

	}



}

float2 variance_and_mean(int* array_in,int nelements)
{
	float mean = 0;
	float variance = 0;

	float2 result;

	// Calculate the mean
	for(int i=0;i<nelements;i++)
	{
		mean += array_in[i];
	}

	mean /= (float)nelements;

	// Calculate the varience
	for(int i=0;i<nelements;i++)
	{
		variance += pow((array_in[i] - mean),2);
	}

	variance /= (float)nelements;

	result.x = mean;
	result.y = variance;

	return result;
}

int main(void)
{
	cudaSetDevice(1);

	int nptcls = 1.0e7;
	int nbatches = 40;

	int nptcls_min = nptcls/4;
	int niter_subcycle = 10;

	int cudaBlockSize = 256;
	int cudaGridSize = 84;

	int seed = time(NULL);

	//int seed = 4358907;

	int nthreads = cudaBlockSize*cudaGridSize;

	int* absorbed;
	int* left_domain;
	curandState* random_states;

	int batch_absorbed[nbatches];
	int batch_left[nbatches];

	// Allocate the particle list
	NeutronList neutrons(nptcls);
	NeutronList neutrons_next(nptcls);

	neutrons.weight_low = 0.25;
	neutrons.weight_avg = 1.0;

	neutrons_next.weight_low = 0.25;
	neutrons_next.weight_avg = 1.0;

	Domain* domains_d;
	Domain domains_h[n_domains];

	/***********************************************/
	/* Initialize the random number generators */

	// Allocate memory for the reduction arrays and the random generator states
	cudaMalloc((void**)&random_states,nthreads*sizeof(curandState));
	cudaMalloc((void**)&domains_d,n_domains*sizeof(Domain));

	// Initialize the random number generators
	CUDA_SAFE_KERNEL((random_init<<<cudaGridSize,cudaBlockSize>>>
								 (random_states,seed,nthreads)));

	/***********************************************/

	/***********************************************/
	/* Initialize the domains */

	Problem1 problem;

	for(int i=0;i<n_domains;i++)
	{
		domains_h[i] = problem.materials[i];
	}

	printf("Copying domains to device\n");
	CUDA_SAFE_KERNEL((cudaMemcpy(domains_d,domains_h,n_domains*sizeof(Domain),cudaMemcpyHostToDevice)));

	/***********************************************/

	// Run the simulation
	float time_max = 10000.0;


	uint timer;
	cutCreateTimer(&timer);

	float run_time;

	int nsteps = 1;

	float weight = 1.0;

	for(int i=0;i<nsteps;i++)
	{
		cutStartTimer(timer);

		/***********************************************/
		/* Initialize the current set of neutrons */
		printf("Copying domains to device\n");
		CUDA_SAFE_KERNEL((neutron_init<<<cudaGridSize,cudaBlockSize>>>
									 (neutrons,problem,random_states,weight,0)));

		printf("Copying domains to device\n");

		neutrons.nptcls = neutrons.nptcls_allocated;

		/***********************************************/

		/***********************************************/
		/* Run the superstep */

		super_cycle(neutrons,domains_d,neutrons_next,random_states,time_max,niter_subcycle,nptcls_min);

		/***********************************************/

		/***********************************************/
		/* Swap the lists */

		NeutronList neutrons_swap = neutrons;
		neutrons = neutrons_next;
		neutrons_next = neutrons_swap;

		/***********************************************/




		cutStopTimer(timer);

	}

	run_time = cutGetTimerValue(timer);

	float2 absorbed_stats = variance_and_mean(batch_absorbed, nbatches);
	float2 left_stats =  variance_and_mean(batch_left, nbatches);



	float* flux_profile = (float*)malloc(n_domains*512*sizeof(float));
	float* x_profile = (float*)malloc(n_domains*512*sizeof(float));



	Domain slab1 = domains_h[1].copy_to_host();
	Domain slab2 = domains_h[2].copy_to_host();

	for(int i=0;i<slab1.ncells;i++)
	{
		float dxdc = abs((slab1.walls[1]-slab1.walls[0]))/((float)slab1.ncells);
		x_profile[i] = i*dxdc + slab1.walls[0];
	}

	for(int i=0;i<slab2.ncells;i++)
	{
		float dxdc = abs((slab2.walls[1]-slab2.walls[0]))/((float)slab2.ncells);
		x_profile[i+slab1.ncells] = i*dxdc + slab2.walls[0];
	}
	gnuplot_ctrl* plot1d;

	printf("Average Number of Particles absorbed: %f +/- %f\n",slab1.absorptions[0]/nptcls,0);
	printf("Number of Collisions: %f +/- %f\n",slab1.collisions[0]/nptcls,0);
	printf("Average Number of Particles absorbed: %f +/- %f\n",slab2.absorptions[0]/nptcls,0);
	printf("Number of Collisions: %f +/- %f\n",slab2.collisions[0]/nptcls,0);

	printf("Absorption Cross Sections: Slab 1: %f Slab 2 %f\n",slab1.absorptions[0]/slab1.collisions[0],slab2.absorptions[0]/slab2.collisions[0]);
	printf("Total Run Time was %f ms\n",run_time);

	memcpy(flux_profile,slab1.flux_tally,slab1.ncells*sizeof(float));
	memcpy(flux_profile+slab1.ncells,slab2.flux_tally,slab2.ncells*sizeof(float));

	plot1d = gnuplot_init();



	gnuplot_setstyle(plot1d, "lines");

	gnuplot_plot_xy(plot1d,x_profile,flux_profile,512,"Flux Tally");

	printf("Press 'Enter' to continue\n");
	getchar();
	gnuplot_close(plot1d);



	return 0;
}































