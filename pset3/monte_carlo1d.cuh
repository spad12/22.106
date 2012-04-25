#include "../include/gnuplot_i.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cuda.h"
#include <thrust/reduce.h>
#include <thrust/scan.h>
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


#define n_domains 5

__constant__ float pi_const = 3.14159265;
__constant__ float mass_neutron = 1.04541e-12; //ev cm^-2 s^2

__constant__ float tungsten_endf_e[6] = {0.0f,1.5e5f,2.5e5f,3.5e5f,4.5e5f,5.5e5f};

__constant__ float tungsten_endf_p5_probs[6] = {3.2666e-6,2.8753e-6,1.4773e-6,5.8832e-7,1.4414e-7,1.4904e-8};
__constant__ float tungsten_endf_p5_sum[7] = {0.48999f, 0.77752f, 0.92525f, 0.984082f, 0.998496f, 0.99998644f,1.0f};


typedef float (*CXFunctionPtr)(const float&);
typedef void (*ScatterFunctionPtr)(float&,float&,float,curandState*);

template <typename T>
__device__
int sgn(T val) {
    return (val > T(0)) - (val < T(0));
}


/***************************************************/
/*     Cross Section Function Definitions	   */

__device__
float no_SigmaI(const float& energy)
{
	return 0;
}
// Problem 1
__device__
float pset1_slab1_SigmaA(const float& energy)
{
	return 0.5;
}

__device__
float pset1_slab1_SigmaE(const float& energy)
{
	return 1.0;
}

__device__
float pset1_slab2_SigmaA(const float& energy)
{
	return 1.2;
}

__device__
float pset1_slab2_SigmaE(const float& energy)
{
	return 0.3;
}

/***************************************************/
// Problem 2
__device__
float tungsten_SigmaA(const float& energy)
{
	return energy < 1.0e3 ? 0.063058 : 0.0063058;
}

__device__
float tungsten_SigmaE(const float& energy)
{
	return 0.252233;
}

__device__
float tungsten_SigmaI(const float& energy)
{
	return 0.063058*(1.03*log(energy)-11.99);
}

__device__
float water_SigmaA(const float& energy)
{

	return 6.691263e-5*((3.0-1000.0)/(1.0e6-0.1)*(energy - 0.1)+1000.0);
}

__device__
float water_SigmaE(const float& energy)
{
	return 1.47208;
}

__device__
float He3_SigmaA(const float& energy)
{

	return 0.01;
}

__device__
float He3_SigmaE(const float& energy)
{
	return 0.09;
}

/***************************************************/
/*     Scatter Function Definitions	   */
__device__
void isoscatter_lcs(float& mu, float& energy,float Amass, curandState* randoms)
{
	// The new theta direction
	float theta = pi_const*curand_uniform(randoms);

	// The new phi direction
	float phi = 2.0f*pi_const*curand_uniform(randoms);

	mu =  cos(phi) * sin(theta);

}

__device__
void isoscatter_cmcs(float& mu, float& energy,float Amass, curandState* randoms)
{
	// The new theta direction
	float theta = pi_const*curand_uniform(randoms);

	// The new phi direction
	float phi = 2.0f*pi_const*curand_uniform(randoms);

	float muc =  cos(phi) * sin(theta);

	// Lab scattering angle
	muc = (1.0f+Amass*muc)/sqrt(Amass*Amass+1.0f+2.0f*muc);

	// Change in angle
	float dmu = (mu-muc)/2.0f;

	energy *= (2.0f*dmu*dmu+Amass*Amass-1.0f+2.0f*dmu*sqrt(Amass*Amass+dmu*dmu-1))/
			((Amass+1)*(Amass+1));

}

__device__
void tungsten_iescatter(float& mu, float& energy,float Amass, curandState* randoms)
{
	// The new theta direction
	float theta = pi_const*curand_uniform(randoms);

	// The new phi direction
	float phi = 2.0f*pi_const*curand_uniform(randoms);

	mu =  cos(phi) * sin(theta);

	float random_number = curand_uniform(randoms);

	if((energy >= 110.0e3)&&(energy <= 500.0e3))
	{

		if(energy > 150e3)
		{
			// Use the first energy table

			int i_prob = 0;
			float test_prob = 0;

			while(random_number > tungsten_endf_p5_sum[i_prob])
			{
				i_prob++;
			}

			energy = (random_number-tungsten_endf_p5_sum[i_prob])/
					tungsten_endf_p5_probs[i_prob]+
					tungsten_endf_e[i_prob];

		}

	}
	else if((energy >= 500.0e3)&&(energy <= 700.0e3))
	{


		// Use the second energy table

		int i_prob = 0;
		float test_prob = 0;

		while(random_number > tungsten_endf_p5_sum[i_prob])
		{
			i_prob++;
		}

		energy = (random_number-tungsten_endf_p5_sum[i_prob])/
				tungsten_endf_p5_probs[i_prob]+
				tungsten_endf_e[i_prob];


	}

}

__device__
void no_Iescatter(float& mu, float& energy,float Amass, curandState* randoms)
{
	return;
}



class ProblemSet;

class Domain
{
public:
	CXFunctionPtr SigmaA;
	CXFunctionPtr SigmaE;
	CXFunctionPtr SigmaI;
	ScatterFunctionPtr Escatter;
	ScatterFunctionPtr Iscatter;
	float AtomicNumber;
	float walls[2];

	float* flux_tally;
	float* collisions; // Size of a warp in shared, 32*gridSize in global
	float* absorptions; // Size of a warp in shared, 32*gridSize in global
	int ncells;
	int nblocks;

	bool keep_or_kill; // Do we keep or kill particles in this domain 1 for keep 0 for kill
	bool location; // true for gpu, false for cpu

	__device__
	float get_SigmaE(const float& energy)
	{
		return SigmaE(energy);
	}

	__device__
	float get_SigmaS(const float& energy)
	{
		return SigmaE(energy) + SigmaI(energy);
	}

	__device__
	float get_SigmaA(const float& energy)
	{
		return SigmaA(energy);
	}

	__device__
	float get_SigmaT(const float& energy)
	{
		return SigmaA(energy)+get_SigmaS(energy);
	}

	__device__
	void tally_flux(const float& px, const float& px0,
						    const float& mu, const float& weight)
	{
		if(keep_or_kill)
		{
			float px1 = min(px,px0);
			float px2 = max(px,px0);
			float dxdc = abs((walls[1]-walls[0]))/((float)ncells);
			int cell_start = floor((px1-walls[0])/dxdc);
			int cell_end = floor((px2-walls[0])/dxdc);

			//cell_start = min(ncells-1,max(cell_start,0));
			//cell_end = min(ncells-1,max(cell_end,0));

			float dl;

			// First Cell
			dl = dxdc*(cell_start+1)+walls[0] - px1;
			atomicAdd(flux_tally+cell_start,abs(dl*weight/(dxdc)));
			// Last Cell
			dl = px2 - (dxdc*(cell_end)+walls[0]);
			atomicAdd(flux_tally+cell_end,abs(dl*weight/(dxdc)));

			dl = abs(weight);
			for(int i=cell_start+1;i<cell_end;i++)
			{
				atomicAdd(flux_tally+i,(dl));
			}
		}
	}

	__device__
	void tally_collisions(const float& weight,const float& energy)
	{
		if(keep_or_kill)
		{
			// collisions and absorbtions should be in shared memory
			// since threads are grouped by warps we don't need atomics
			int warpIdx = threadIdx.x%32;

			// These should be volatiles so that the compiler doesn't mess this up
			//((volatile float*)collisions)[warpIdx] += weight;
			//((volatile float*)absorptions)[warpIdx] += weight*SigmaA/SigmaT;
			atomicAdd(collisions+warpIdx,weight);
			atomicAdd(absorptions+warpIdx,weight*get_SigmaA(energy)/get_SigmaT(energy));
		}
	}

	__device__
	void scatter(float& weight,float& mu, float& energy, curandState* randoms)
	{
		// Adjust the weight
		weight *= (1.0f - get_SigmaA(energy)/get_SigmaT(energy));

		// Check to see what kind of scatter we have
		if(curand_uniform(randoms) < get_SigmaE(energy)/get_SigmaS(energy))
		{
			// Elastic Scatter
			Escatter(mu,energy,AtomicNumber,randoms);
		}
		else
		{
			// Inelastic Scatter
			Iscatter(mu,energy,AtomicNumber,randoms);
		}
	}

	__host__
	void allocate(int ncells_in,int nblocks_in,bool ilocation)
	{
		ncells = ncells_in;
		nblocks = nblocks_in;
		location = ilocation;

		printf("Allocating Domain\n");

		if(location)
		{
			cudaMalloc((void**)&flux_tally,ncells*sizeof(float));
			cudaMalloc((void**)&collisions,nblocks*32*sizeof(float));
			cudaMalloc((void**)&absorptions,nblocks*32*sizeof(float));

			cudaMemset(flux_tally,0,ncells*sizeof(float));
			cudaMemset(collisions,0,nblocks*32*sizeof(float));
			cudaMemset(absorptions,0,nblocks*32*sizeof(float));
		}
		else
		{
			flux_tally = (float*)malloc(ncells*sizeof(float));
			collisions = (float*)malloc(sizeof(float));
			absorptions = (float*)malloc(sizeof(float));
		}

	}

	__host__
	Domain copy_to_host(void)
	{
		Domain domain_h;
		domain_h = *this;
		domain_h.allocate(ncells,0,0);

		// Copy the flux distribution to the host
		CUDA_SAFE_CALL(cudaMemcpy(domain_h.flux_tally,flux_tally,ncells*sizeof(float),cudaMemcpyDeviceToHost));

		// Reduce the number of collisions and absorptions
		thrust::device_ptr<float> collisions_t(collisions);
		thrust::device_ptr<float> absorptions_t(absorptions);

		domain_h.collisions[0] = thrust::reduce(collisions_t,collisions_t+32*nblocks);
		domain_h.absorptions[0] = thrust::reduce(absorptions_t,absorptions_t+32*nblocks);

		return domain_h;
	}

	__host__
	void Reset(void)
	{
		cudaMemset(flux_tally,0,ncells*sizeof(float));
		cudaMemset(collisions,0,nblocks*32*sizeof(float));
		cudaMemset(absorptions,0,nblocks*32*sizeof(float));
	}

	__host__
	void Domain_free(void)
	{
		if(location)
		{
			cudaFree(flux_tally);
			cudaFree(collisions);
			cudaFree(absorptions);
		}
		else
		{
			free(flux_tally);
			free(collisions);
			free(absorptions);
		}
	}



};

const int NeutronList_nfloats = 6;
const int NeutronList_nints = 3;

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
	float* px; // cm
	float* mu;
	float* energy; // MeV
	float* weight;
	float* time_done;

	float* buffer;

	int* domain;

	int* dead;
	int* finished_time;

	float weight_low;
	float weight_avg;

	int nptcls;
	int nptcls_allocated;

	__host__ __device__
	NeutronList(){;}

	__host__
	NeutronList(int nptcls_in){allocate(nptcls_in);}

	__host__
	void allocate(int nptcls_in)
	{
		nptcls_allocated = nptcls_in;
		nptcls = 0;

		for(int i=0;i<NeutronList_nfloats;i++)
		{
			cudaMalloc((void**)get_float_ptr(i),nptcls_allocated*sizeof(float));
		}

		for(int i=0;i<NeutronList_nints;i++)
		{
			cudaMalloc((void**)get_int_ptr(i),nptcls_allocated*sizeof(float));
		}


	}

	__host__
	void initiialize(float v0_in, float px_in, float E0_in,float time_in, int nptcls_in)
	{
		nptcls = nptcls_in;
		cudaMemset(mu,v0_in,nptcls*sizeof(float));

		cudaMemset(energy,E0_in,nptcls*sizeof(float));

		cudaMemset(px,px_in,nptcls*sizeof(float));

		cudaMemset(weight,1.0,nptcls*sizeof(float));

		cudaMemset(dead,0,nptcls*sizeof(float));
	}

	__device__
	void clear_slot(int slot_index)
	{
		dead[slot_index] = 1;
		weight[slot_index] = 0;
		finished_time[slot_index] = 0;
		time_done[slot_index] = 0;
	}

	__device__
	void split(int slot_index)
	{
		// Divide weight by 2 and copy everything to
		// the slot that is nptcls away.
		if(slot_index+nptcls < nptcls_allocated)
		{
		weight[slot_index] /= 2.0f;
		int new_index = slot_index+nptcls;
		for(int i=0;i<NeutronList_nfloats-1;i++)
		{
			float* data_ptr = *get_float_ptr(i);

			data_ptr[new_index] = data_ptr[slot_index];
		}

		for(int i=0;i<NeutronList_nints;i++)
		{
			int* data_ptr = *get_int_ptr(i);

			data_ptr[new_index] = data_ptr[slot_index];
		}
		}
	}

	__host__ __device__
	float** get_float_ptr(int member)
	{
		// This lets us iterate over the members of this object
		float** result;
		switch(member)
		{
		case 0:
			result = &mu;
			break;
		case 1:
			result = &px;
			break;
		case 2:
			result = &energy;
			break;
		case 3:
			result = &weight;
			break;
		case 4:
			result = &time_done;
			break;
		case 5:
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
			result = &domain;
			break;
		case 1:
			result = &dead;
			break;
		case 2:
			result = &finished_time;
			break;
		default:
			break;
		}

		return result;

	}


	// Pop out neutrons that have condition = 1 and append them to ListOut
	__host__
	void pop_and_append(NeutronList& ListOut, int* condition);

	// Pop out the neutrons that have finished their time step
	// but have been absorbed or lost or killed
	// and append them to the list for the next step
	__host__
	void pop_finished(NeutronList& ListOut);

	__host__
	void condense_alive(void);

	__host__
	void update_average_weight(void);

	__host__
	void NeutronListFree(void)
	{
		for(int i=0;i<NeutronList_nfloats;i++)
		{
			cudaFree(*get_float_ptr(i));
		}

		for(int i=0;i<NeutronList_nints;i++)
		{
			cudaFree(*get_int_ptr(i));
		}
	}
};

struct Neutron_data
{
	float fltdata[NeutronList_nfloats-1];
	int intdata[NeutronList_nints];
};


class Neutron : public NeutronList
{
public:
	int *parentID;
	curandState* random_state;

	__device__
	Neutron(Neutron_data& storage,int* parentID_in)
	{

		parentID = parentID_in;

		for(int i=0;i<NeutronList_nfloats-1;i++)
		{
			*get_float_ptr(i) = storage.fltdata+i;
		}

		for(int i=0;i<NeutronList_nints;i++)
		{
			*get_int_ptr(i) = storage.intdata+i;
		}


	}

	__device__
	Neutron& operator=(NeutronList& parents)
	{
		weight_avg = parents.weight_avg;
		nptcls = parents.nptcls;
		for(int i=0;i<NeutronList_nfloats-1;i++)
		{
			*(*get_float_ptr(i)) = parents.get_float_ptr(i)[0][*parentID];
		}

		for(int i=0;i<NeutronList_nints;i++)
		{
			*(*get_int_ptr(i)) = parents.get_int_ptr(i)[0][*parentID];
		}

		return *this;
	}

	__device__
	void merge(NeutronList& parents)
	{
		for(int i=0;i<NeutronList_nfloats-1;i++)
		{
			parents.get_float_ptr(i)[0][*parentID] = get_float_ptr(i)[0][0];
		}

		for(int i=0;i<NeutronList_nints;i++)
		{
			parents.get_int_ptr(i)[0][*parentID] = get_int_ptr(i)[0][0];
		}
	}

	__device__
	float random(void){return curand_uniform(random_state);}

	__device__
	void advance(Domain* materials,const float& max_time)
	{
		// Save some values for later
		float px0 = *px;
		float weight0 = *weight;
		float mu0 = *mu;
		float energy0 = *energy;
		int domain0 = *domain;


		float dl = -log(random())/materials[*domain].get_SigmaT(*energy);


		int left_or_right = floor(0.5*(sgn(*mu)+3))-1;

		float dwall = materials[*domain].walls[left_or_right] - *px;

		if((sgn(*mu)*dwall)<(dl*abs(*mu)))
		{
			// We run into a wall before we collide
			// Advance to the wall and return
			//printf("px = %f dl*mu = %f, lorR = %i\n",*px,dl*(*mu),left_or_right);
			*px += dwall;
			*domain += sgn(*mu);

		}
		else
		{
			*px += dl*(*mu);
			// Collision
			materials[*domain].scatter(*weight,*mu,*energy,random_state);
		}

		*domain = max(0,min(n_domains-1,*domain));

		// Increment the time done
		float velocity = sqrt(2.0f*(energy0)/mass_neutron); // cm/s
		*time_done += abs((*px-px0)/(mu0*velocity));

		*finished_time = *time_done >= max_time;

		// Contribute to Tallies
		materials[domain0].tally_flux(*px,px0,mu0,weight0/(weight_avg*nptcls));

		// Update particles alive or dead status
		check_domain(materials);

	}

	__device__
	void check_domain(Domain* material)
	{
		// Sets the particle as dead if outside the computational domain
		*dead = ((material[*domain].keep_or_kill) != 1);
	}

	__device__
	bool check_absorb(Domain& material)
	{
		// Returns true if the neutron was absorbed
		return (random() <= material.get_SigmaA(*energy)/material.get_SigmaT(*energy));
	}

	__device__
	int check_subexit(void)
	{
		// return 1 if the particle needs to exit the subcycle loop
		return (*dead)||(*finished_time);
	}

	__device__
	void isoscatter(void)
	{
		// The new theta direction
		float theta = pi_const*random();

		// The new phi direction
		float phi = 2.0f*pi_const*random();

		*mu =  cos(phi) * sin(theta);

	}

	__device__
	void russian_roulette(void)
	{
		float weight_new;
		int weight_is_low = (*weight <= weight_low);
		int survived = (random() < ((*weight)/weight_avg));

		// Avoid Branching by using the fact that true/false are 1 and 0.

		weight_new = (1-weight_is_low)*(*weight);
		weight_new += (weight_is_low)*survived*(*weight)/weight_avg;

		*dead = 1-survived;
		// If the neutron is dead, weight is 0
		*weight = weight_new*((1-(*dead)));
		*finished_time = (*finished_time)&&((1-*dead));
	}

};


class ProblemSet
{
public:
	Domain materials[n_domains];
	int ncells;


};










