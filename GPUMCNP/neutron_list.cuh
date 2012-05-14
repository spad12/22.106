#ifndef NEUTRON_LIST_INC
#define NEUTRON_LIST_INC
#include "gpumcnp.h"


const int NeutronList_nfloats = 10;
const int NeutronList_nints = 3;

extern "C" class NeutronList
{
public:
	float* px,*py; // cm
	float* vx,*vy,*vz; // velocities cm/s
	float* weight;
	float* time_done;
	float* dcollide;
	float* dsince_collide;

	float* buffer;

	int* mDomain;

	int* dead; // 0 for alive, 1 for dead

	int* pindex;

	short int* binid;

	float weight_avg;
	float weight_low;


	int nptcls; // Number of particle slots filled
	int nptcls_allocated;


	__host__ __device__
	NeutronList(){;}

	__host__
	NeutronList(int nptcls_in){allocate(nptcls_in);}

	__device__
	NeutronList& operator=(MCNeutron& child);


	__host__ __device__
	float** get_float_ptr(int member)
	{
		// This lets us iterate over the members of this object
		float** result;
		switch(member)
		{
		case 0:
			result = &px;
			break;
		case 1:
			result = &py;
			break;
		case 2:
			result = &vx;
			break;
		case 3:
			result = &vy;
			break;
		case 4:
			result = &vz;
			break;
		case 5:
			result = &weight;
			break;
		case 6:
			result = &time_done;
			break;
		case 7:
			result = &dcollide;
			break;
		case 8:
			result = &dsince_collide;
			break;
		case 9:
			result = &buffer;
			break;
		default:
			break;
		}

		return result;

	}

	__host__ __device__
	int** get_int_ptr(int member)
	{
		// This lets us iterate over the members of this object
		int** result;
		switch(member)
		{
		case 0:
			result = &mDomain;
			break;
		case 1:
			result = &dead;
			break;
		case 2:
			result = &pindex;
			break;
		default:
			break;
		}

		return result;

	}



	__host__
	void allocate(int nptcls_in)
	{
		nptcls_allocated = nptcls_in;
		nptcls = 0;

		for(int i=0;i<NeutronList_nfloats;i++)
		{
			CUDA_SAFE_CALL(cudaMalloc((void**)get_float_ptr(i),nptcls_allocated*sizeof(float)));
		}

		for(int i=0;i<NeutronList_nints;i++)
		{
			CUDA_SAFE_CALL(cudaMalloc((void**)get_int_ptr(i),nptcls_allocated*sizeof(float)));
		}

		CUDA_SAFE_CALL(cudaMalloc((void**)&binid,nptcls_allocated*sizeof(short int)));

		size_t free = 0;
		size_t total = 0;
		// See how much memory is allocated / free
		cudaMemGetInfo(&free,&total);
		printf("Free Memory = %i mb\nUsed mememory = %i mb\n",(int)(free)/(1<<20),(int)(total-free)/(1<<20));


	}

	__host__
	void initiialize(float v0_in, float px_in, float E0_in,float time_in, int nptcls_in)
	{
		nptcls = nptcls_in;

		cudaMemset(weight,1.0,nptcls*sizeof(float));

		cudaMemset(dead,0,nptcls*sizeof(float));
	}

	// Update the particle binid's and sort the particle list
	__host__
	void sort(SimulationData* simulation);

	__host__
	int CountDeadNeutrons(int istep);

	__host__
	void refill(int istep);




};

class  MCNeutron
{
public:
	float px,py; // cm
	float vx,vy,vz; // velocities cm/s
	float weight;
	float time_done;
	float dcollide;
	float dsince_collide;

	int mDomain;

	int dead; // 0 for alive, 1 for dead

	short int binid;
	curandState*	random_state;

	int* thid;

	__host__ __device__
	MCNeutron(){;}

	__host__ __device__
	MCNeutron(int* thid_in,curandState* random_state_in)
	{
		thid = thid_in;
		random_state = random_state_in;
	}



	__host__ __device__
	float* get_float_ptr(int member)
	{
		// This lets us iterate over the members of this object
		float* result;
		switch(member)
		{
		case 0:
			result = &px;
			break;
		case 1:
			result = &py;
			break;
		case 2:
			result = &vx;
			break;
		case 3:
			result = &vy;
			break;
		case 4:
			result = &vz;
			break;
		case 5:
			result = &weight;
			break;
		case 6:
			result = &time_done;
			break;
		case 7:
			result = &dcollide;
			break;
		case 8:
			result = &dsince_collide;
			break;
		default:
			break;
		}

		return result;

	}

	__host__ __device__
	int* get_int_ptr(int member)
	{
		// This lets us iterate over the members of this object
		int* result;
		switch(member)
		{
		case 0:
			result = &mDomain;
			break;
		case 1:
			result = &dead;
			break;
		default:
			break;
		}

		return result;

	}

	__device__
	MCNeutron& operator=(NeutronList& parents);

	__device__
	float random();



//	__device__
///	float2&	position()
//	{
//		return *((float2*)&px);
//	}


	__device__
	MCNeutron Advance(
		SimulationData& 	simulation,
		float&					distance,
		const int&			qScattering
		);

	__device__
	void Tally(
		SimulationData&	simulation,
		MCNeutron&			neutrons_old,
		float&					distance);

	__device__
	void STally(
		SimulationData&	simulation,
		MCNeutron&			neutrons_old,
		float*					s1,
		float*					s2,
		float&					distance);

	__device__
	void RussianRoulette(
		const float&		weight_avg,
		const float&		weight_low);

	__device__
	void check_domain(
		SimulationData& 	simulation,
		MCNeutron& 			neutron_old);
};




















#endif


