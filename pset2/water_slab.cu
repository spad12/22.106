#include "../include/gnuplot_i.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cuda.h"
#include <thrust/reduce.h>
#include "curand.h"
#include "curand_kernel.h"
#include "cutil.h"



#  define CUDA_SAFE_KERNEL(call) {                                         \
	call;																					\
	cudaDeviceSynchronize();														\
	cudaError err = cudaGetLastError();										\
    if ( cudaSuccess != err) {                                               \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
                exit(EXIT_FAILURE);                                                  \
    } }

__constant__ float pi_const = 3.14159265;

class Domain
{
public:
	float SigmaS;
	float SigmaA;
	float SigmaT;
	float width;

};

static int NeutronList_nfloats = 5;

enum NeutronListMember
{
	NeutronList_vx = 0,
	NeutronList_px = 1,
	NeutronList_v0 = 2,
	NeutronList_time = 3,
	NeutronList_buffer = 4
};

class NeutronList
{
public:
	float* vx;
	float* px;
	float* v0;
	float* time;

	float* buffer;

	int nptcls;

	__host__ __device__
	NeutronList(){;}

	__host__
	NeutronList(int nptcls_in){allocate(nptcls_in);}

	__host__
	void allocate(int nptcls_in)
	{
		nptcls = nptcls_in;

		for(int i=0;i<NeutronList_nfloats;i++)
		{
			cudaMalloc((void**)get_float_ptr(i),nptcls*sizeof(float));
		}

	}

	__host__
	void initiialize(float v0_in, float px_in, float time_in)
	{
		cudaMemset(vx,v0_in,nptcls*sizeof(float));
		cudaMemset(v0,v0_in,nptcls*sizeof(float));

		cudaMemset(px,px_in,nptcls*sizeof(float));

		cudaMemset(time,time_in,nptcls*sizeof(float));
	}

	__host__
	float** get_float_ptr(int member)
	{
		// This lets us iterate over the members of this object
		float** result;
		switch(member)
		{
		case 0:
			result = &vx;
			break;
		case 1:
			result = &px;
			break;
		case 2:
			result = &v0;
			break;
		case 3:
			result = &time;
			break;
		case 4:
			result = &buffer;
			break;
		default:
			break;
		}

		return result;

	}

	__host__
	void NeutronListFree(void)
	{
		for(int i=0;i<NeutronList_nfloats;i++)
		{
			cudaFree(*get_float_ptr(i));
		}
	}
};

class Neutron : public NeutronList
{
public:
	int *parentID;
	curandState* random_state;

	__device__
	Neutron(float4& storage,int* parentID_in)
	{

		parentID = parentID_in;

		vx = &(storage.x);
		px = &(storage.y);
		v0 = &(storage.z);
		time = &(storage.w);

	}

	__device__
	Neutron& operator=(const NeutronList& parents)
	{
		*vx = parents.vx[*parentID];
		*px = parents.px[*parentID];
		*v0 = parents.v0[*parentID];
		*time = parents.time[*parentID];

		return *this;
	}

	__device__
	float random(void){return curand_uniform(random_state);}

	__device__
	void advance(Domain& material)
	{
		float dl = -log(random())/material.SigmaT;

		*px += (*vx)/(*v0)*dl;
		*time += *v0/dl;
	}

	__device__
	bool check_domain(Domain& material)
	{
		// Returns true if the neutron is outsied the domain
		return (*px > material.width)||(*px < 0);
	}

	__device__
	bool check_absorb(Domain& material)
	{
		// Returns true if the neutron was absorbed
		return (random() <= material.SigmaA/material.SigmaT);
	}

	__device__
	void isoscatter(void)
	{
		// The new theta direction
		float theta = pi_const*random();

		// The new phi direction
		float phi = 2.0f*pi_const*random();

		*vx = *v0 * cos(phi) * sin(theta);




	}

};


__global__
void random_init(curandState* random_states,int seed, int nstates)
{
	int gidx = threadIdx.x+blockDim.x*blockIdx.x;

	if(gidx < nstates)
	{
		curand_init(seed,gidx,gidx,random_states+gidx);
	}
}

__global__
void neutron_advance(NeutronList neutrons, Domain water,curandState* random_state,
									  int* absorbed, int* left_domain,float time_max)
{
	int gidx = threadIdx.x + blockDim.x*blockIdx.x;
	int thid = gidx;

	__shared__ int Absorbed_distribution[20];

	float4 storage;

	Neutron my_neutron(storage,&gidx);
	my_neutron.parentID = &gidx;
	my_neutron.random_state = random_state+thid;
	
	int absorbed_local = 0;
	int left_domain_local = 0;



	while(gidx < neutrons.nptcls)
	{
		// Copy neutron data from global memory to registers
		my_neutron = neutrons;

		// Neutron starts at some random point in the slab
		my_neutron.px[0] = my_neutron.random()*water.width;
		my_neutron.isoscatter();
		bool qdidileave = false;
		bool qabsorbed = false;

		while(my_neutron.time[0] < time_max)
		{
			if(!qdidileave)
			{
				// Advance the neutron
				my_neutron.advance(water);

				qabsorbed = my_neutron.check_absorb(water);
				qdidileave = my_neutron.check_domain(water);

				// If we left the domain and were absorbed,
				// then we weren't really absorbed.
				qabsorbed = qabsorbed&&(!qdidileave);

				// Contribute to the local tally
				absorbed_local += qabsorbed;
				left_domain_local += qdidileave;

				// If we were absorbed, then for the purposes of this code we have left
				qdidileave = qabsorbed||qdidileave;
				if(!qdidileave)
				{
					// We weren't absorbed and we didn't leave, so we scattered.
					my_neutron.isoscatter();
				}
				else my_neutron.time[0] = time_max;

			}

		}

		gidx += blockDim.x*gridDim.x;

	}

	absorbed[thid] = absorbed_local;
	left_domain[thid] = left_domain_local;


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

	int nptcls = 5.0e7;
	int nbatches = 40;

	int cudaBlockSize = 256;
	int cudaGridSize = 84;

	int seed = time(NULL);

	int nthreads = cudaBlockSize*cudaGridSize;

	int* absorbed;
	int* left_domain;
	curandState* random_states;

	int batch_absorbed[nbatches];
	int batch_left[nbatches];

	// Allocate the particle list
	NeutronList neutrons(nptcls);

	Domain water;
	water.SigmaS = 1.0;
	water.SigmaA = 0.5;
	water.SigmaT = 1.5;
	water.width = 10.0;

	// Set the particle list initial conditions
	float v0 = 1.0;
	neutrons.initiialize(v0,0.0f,0.0f);

	// Allocate memory for the reduction arrays and the random generator states
	cudaMalloc((void**)&random_states,nthreads*sizeof(curandState));
	cudaMalloc((void**)&absorbed,nthreads*sizeof(int));
	cudaMalloc((void**)&left_domain,nthreads*sizeof(int));

	// Initialize the random number generators
	CUDA_SAFE_KERNEL((random_init<<<cudaGridSize,cudaBlockSize>>>
								 (random_states,seed,nthreads)));

	// Run the simulation
	float time_max = 5.0;

	uint timer;
	cutCreateTimer(&timer);

	float run_time;


	printf("Running %i batches of %i particles each...\n",nbatches,nptcls);
	for(int i=0;i<nbatches;i++)
	{
		cutStartTimer(timer);
	CUDA_SAFE_KERNEL((neutron_advance<<<cudaGridSize,cudaBlockSize>>>
								(neutrons,water,random_states,absorbed,left_domain,time_max)));

	thrust::device_ptr<int> absorbed_t(absorbed);
	thrust::device_ptr<int> left_domain_t(left_domain);


	int total_absorbed = thrust::reduce(absorbed_t,absorbed_t+nthreads);
	int total_left = thrust::reduce(left_domain_t,left_domain_t+nthreads);

	cudaMemset(absorbed,0,nthreads*sizeof(int));
	cudaMemset(left_domain,0,nthreads*sizeof(int));



	run_time = cutGetTimerValue(timer);

	int nremaining = nptcls-(total_absorbed+total_left);
/*
	printf("Total Number of Particles: %i with velocity %f (cm/s)\n",nptcls,v0);
	printf("Number of Particles absorbed: %i\n",total_absorbed);
	printf("Number of Particles escaped: %i\n",total_left);
	printf("Number of Particles remaining: %i\n",nremaining);
	printf("Run Took %f ms\n",run_time);
	*/

	batch_absorbed[i] = total_absorbed;
	batch_left[i] = total_left;

	cutStopTimer(timer);

	}

	float2 absorbed_stats = variance_and_mean(batch_absorbed, nbatches);
	float2 left_stats =  variance_and_mean(batch_left, nbatches);

	printf("Average Number of Particles absorbed: %f +/- %f\n",absorbed_stats.x,sqrt(absorbed_stats.y));
	printf("Number of Particles escaped: %f +/- %f\n",left_stats.x,sqrt(left_stats.y));
	printf("Total Run Time was %f ms\n",run_time);


	return 0;
}































