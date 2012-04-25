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


#define n_domains 7

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

static __inline__ __device__
float no_SigmaI(const float& energy)
{
	return 0.0;
}

static __inline__ __device__
float infinite_absorber(const float& energy)
{
	return 1.0e6;
}


// Problem 1
static __inline__ __device__
float pset1_slab1_SigmaA(const float& energy)
{
	return 0.5;
}

static __inline__ __device__
float pset1_slab1_SigmaE(const float& energy)
{
	return 0.5;
}

__device__
float pset1_slab2_SigmaA(const float& energy)
{
	return 1.2;
}

static __inline__ __device__
float pset1_slab2_SigmaE(const float& energy)
{
	return 0.3;
}

/***************************************************/
// Problem 2
static __inline__ __device__
float tungsten_SigmaA(const float& energy)
{
	return energy < 1.0e3 ? 0.063058 : 0.0063058;
}

static __inline__ __device__
float tungsten_SigmaE(const float& energy)
{
	return 0.252233;
}

static __inline__ __device__
float tungsten_SigmaI(const float& energy)
{
	return 0.063058*(1.03*log(energy)-11.99);
}

static __inline__ __device__
float water_SigmaA(const float& energy)
{

	return 6.691263e-5*((3.0-1000.0)/(1.0e6-0.1)*(energy - 0.1)+1000.0);
}

static __inline__ __device__
float water_SigmaE(const float& energy)
{
	return 1.47208;
}

static __inline__ __device__
float He3_SigmaA(const float& energy)
{

	return 0.01;
}

static __inline__ __device__
float He3_SigmaE(const float& energy)
{
	return 0.09;
}

/***************************************************/
/*     Scatter Function Definitions	   */
static __inline__ __device__
void isoscatter_lcs(float& mu, float& energy,float Amass, curandState* randoms)
{

	mu =  2.0*curand_uniform(randoms)-1.0;

}

static __inline__ __device__
void isoscatter_cmcs(float& mu, float& energy,float Amass, curandState* randoms)
{


	float muc =  2.0*curand_uniform(randoms)-1.0;



	energy = sqrt(energy)/(Amass+1.0)*(muc+sqrt(Amass*Amass-1.0+muc*muc));
	energy = energy*energy;

	// Lab scattering angle
	muc = (1.0f+Amass*muc)/sqrt(Amass*Amass+1.0f+2.0f*muc);

	//muc = min(1.0f,max(-1.0f,muc));
	//mu = min(1.0,max(-1.0,mu));
	// Change in angle
	//float dmu = cos(acos(muc)+acos(mu));

	mu = muc;

}

static __inline__ __device__
void tungsten_iescatter(float& mu, float& energy,float Amass, curandState* randoms)
{
	// The new theta direction
	float theta = pi_const*curand_uniform(randoms);

	// The new phi direction
	float phi = 2.0f*pi_const*curand_uniform(randoms);

	mu =  2.0f*curand_uniform(randoms)-1.0;

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

static __inline__ __device__
void no_Iescatter(float& mu, float& energy,float Amass, curandState* randoms)
{
	return;
}

__constant__ CXFunctionPtr infinite_absorber_ptr = &infinite_absorber;
__constant__ CXFunctionPtr no_SigmaI_ptr = &no_SigmaI;

__constant__ CXFunctionPtr pset1_slab1_SigmaA_ptr = &pset1_slab1_SigmaA;
__constant__ CXFunctionPtr pset1_slab1_SigmaE_ptr = &pset1_slab1_SigmaE;
__constant__ CXFunctionPtr pset1_slab2_SigmaA_ptr = &pset1_slab2_SigmaA;
__constant__ CXFunctionPtr pset1_slab2_SigmaE_ptr = &pset1_slab2_SigmaE;

__constant__ CXFunctionPtr tungsten_SigmaA_ptr = &tungsten_SigmaA;
__constant__ CXFunctionPtr tungsten_SigmaE_ptr = &tungsten_SigmaE;
__constant__ CXFunctionPtr tungsten_SigmaI_ptr = &tungsten_SigmaI;

__constant__ CXFunctionPtr water_SigmaA_ptr = &water_SigmaA;
__constant__ CXFunctionPtr water_SigmaE_ptr = &water_SigmaE;

__constant__ CXFunctionPtr He3_SigmaA_ptr = &He3_SigmaA;
__constant__ CXFunctionPtr He3_SigmaE_ptr = &He3_SigmaE;

__constant__ ScatterFunctionPtr isoscatter_lcs_ptr = &isoscatter_lcs;
__constant__ ScatterFunctionPtr no_Iescatter_ptr = &no_Iescatter;
__constant__ ScatterFunctionPtr isoscatter_cmcs_ptr = &isoscatter_cmcs;
__constant__ ScatterFunctionPtr tungsten_iescatter_ptr = &tungsten_iescatter;


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
		return get_SigmaA(energy)+get_SigmaS(energy);
	}

	__device__
	void tally_flux(const float& px, const float& px0,
						    const float& mu, const float& weight)
	{
		if(keep_or_kill == 1)
		{
			float px1 = abs(px) < abs(px0) ? px : px0;
			float px2 = abs(px) > abs(px0) ? px : px0;
			float dxdc = abs((walls[1]-walls[0]))/((float)ncells);
			int cell_start = floor((px1-walls[0])/dxdc);
			int cell_end = floor((px2-walls[0])/dxdc);

			cell_start = min(ncells,max(cell_start,0));
			cell_end = min(ncells,max(cell_end,0));

			float dl;

			// First Cell
			dl = (dxdc*(cell_start+1))+walls[0] - px1;
			atomicAdd(flux_tally+cell_start,abs(weight*dl/(dxdc)));
			// Last Cell
			dl = px2 - (dxdc*(cell_end)+walls[0]);
			atomicAdd(flux_tally+cell_end,abs(dl*weight/(dxdc)));

			dl = abs(weight/mu);
			for(int i=cell_start+1;i<cell_end;i++)
			{
				atomicAdd(flux_tally+i,(dl));
			}
		}
	}

	__device__
	void tally_collisions(const float& weight,const float& energy)
	{
		if(keep_or_kill == 1)
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
		if(keep_or_kill == 1)
		{



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

			tally_collisions(weight,energy);

			// Adjust the weight
			weight *= min(1.0,max((1.0f - get_SigmaA(energy)/get_SigmaT(energy)),0.0));

		}
	}

	__host__
	void allocate(int ncells_in,int nblocks_in,bool ilocation)
	{
		ncells = ncells_in;
		nblocks = nblocks_in;
		location = ilocation;

		AtomicNumber = 1;

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
		domain[slot_index] = 0;
		px[slot_index] = 0;
		mu[slot_index] = 0;
	}

	__device__
	void split(int slot_index)
	{
		// Divide weight by 2 and copy everything to
		// the slot that is nptcls away.
		if(slot_index+nptcls < nptcls_allocated)
		{
			if(dead[slot_index] == 0)
			{
				weight[slot_index] /= 2.0;
				int new_index = slot_index+nptcls;
				for(int i=0;i<NeutronList_nfloats-1;i++)
				{
					float* data_ptr = *get_float_ptr(i);

					data_ptr[new_index] = data_ptr[slot_index];
				}

				for(int i=0;i<NeutronList_nints-1;i++)
				{
					int* data_ptr = *get_int_ptr(i);

					data_ptr[new_index] = data_ptr[slot_index];
				}
			}
			else
			{
				dead[slot_index+nptcls] = 1;
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
		weight_low = parents.weight_low;
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

		check_domain(materials);
		if(*dead == 1) return;

		// Save some values for later
		float px0 = *px;
		float weight0 = *weight;
		float mu0 = *mu;
		float energy0 = *energy;
		int domain0 = *domain;


		float dl = -log(random())/materials[*domain].get_SigmaT(*energy);


		int left_or_right = min(1,max((int)floor(0.5*(sgn(*mu)+3))-1,0));

		float dwall = materials[*domain].walls[left_or_right] - *px;

		if((sgn(*mu)*dwall)<(dl*abs(*mu)))
		{
			// We run into a wall before we collide
			// Advance to the wall and return
			//printf("px = %f dl*mu = %f, lorR = %i\n",*px,dl*(*mu),left_or_right);

			*weight *= (1.0f - abs(dwall/dl)*materials[*domain].get_SigmaA(*energy)/materials[*domain].get_SigmaT(*energy));
			*px += dwall;
			*domain += sgn(*mu);




		}
		else
		{
			*px += dl*(*mu);
			// Collision
			materials[*domain].scatter(*weight,*mu,*energy,random_state);
		}



		// Increment the time done
		float velocity = sqrt(2.0f*(energy0)/mass_neutron); // cm/s
		*time_done += abs((*px-px0)/(velocity*mu0));

		*finished_time = (*time_done >= max_time) ? 1:0;

		// Contribute to Tallies
		materials[domain0].tally_flux(*px,px0,mu0,weight0/(weight_avg*nptcls));
/*
		if(abs(*mu) >= 1.0f)
		{
			printf("Energy too high %f\n",*mu);
		}
		//*energy = max(1.0,min(*energy,1.0e6));
*/
		// Update particles alive or dead status
		check_domain(materials);

		//if((*px > 12.0f)||(*px < 0.0f)) *dead = 1;

	}

	__device__
	void check_domain(Domain* material)
	{
		// Sets the particle as dead if outside the computational domain
		*dead = (*domain >= n_domains-1)||(*domain < 1)||(*dead == 1);
		*dead = ((material[*domain].keep_or_kill) != 1)||(*dead == 1);
		*domain = min(n_domains-1,max(0,*domain));
		*weight *= (1.0 - *dead);
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
		return (*dead != 0)||(*finished_time != 0) ? 1:0;
	}

	__device__
	void isoscatter(void)
	{
		*mu =  2.0*curand_uniform(random_state)-1.0;

	}

	__device__
	void russian_roulette(void)
	{
		float weight_new = 0;
		int weight_is_low = (*weight <= weight_low) ? 1:0;
		int survived = (random() < ((*weight)/weight_avg)) ? 1:0;

		// Avoid Branching by using the fact that true/false are 1 and 0.

		weight_new = (1 - weight_is_low)*(*weight);
		weight_new += (weight_is_low)*survived*(*weight)/weight_avg;

		*dead = (1-survived)||(*dead == 1);
		// If the neutron is dead, weight is 0
		*weight = weight_new*((1 - (*dead)));

		/*
		if(*weight >= 1.0001)
		{
			printf("Warning Weight > 1.0 avg weight = %f\n",(*weight)/weight_avg);
		}
		*/
		*finished_time = ((*finished_time == 1)&&((*dead == 0)));
	}

};




class ProblemSet
{
public:
	Domain materials[n_domains];
	int ncells;


};


/***************************************************/
/*           		 Problem 1 						   */
class Problem1 : public ProblemSet
{
public:
	__host__
	Problem1()
	{
		for(int i=0;i<n_domains;i++)
		{
			materials[i].ncells = 0;
			materials[i].keep_or_kill = 0;
			materials[i].allocate(256,84,1);
		}
	}

	__device__
	void init()
	{
		for(int i=0;i<n_domains;i++)
		{
			materials[i].SigmaA = infinite_absorber_ptr;
			materials[i].SigmaE = no_SigmaI_ptr;
			materials[i].SigmaI = no_SigmaI_ptr;
			materials[i].Escatter = isoscatter_lcs_ptr;
			materials[i].Iscatter = no_Iescatter_ptr;
		}

		// Vacuum domain
		materials[0].walls[0] = 0;
		materials[0].walls[1] = 0;
		materials[0].keep_or_kill = 0;


		// Slab 1
		materials[1].walls[0] = 0;
		materials[1].walls[1] = 2;
		materials[1].SigmaA = pset1_slab1_SigmaA_ptr;
		materials[1].SigmaE = pset1_slab1_SigmaE_ptr;
		materials[1].SigmaI = no_SigmaI_ptr;
		materials[1].Escatter = isoscatter_lcs_ptr;
		materials[1].Iscatter = no_Iescatter_ptr;
		materials[1].keep_or_kill = 1;



		// Slab 2
		materials[2].walls[0] = 2;
		materials[2].walls[1] = 6;
		materials[2].SigmaA = pset1_slab2_SigmaA_ptr;
		materials[2].SigmaE = pset1_slab2_SigmaE_ptr;
		materials[2].SigmaI = no_SigmaI_ptr;
		materials[2].Escatter = isoscatter_lcs_ptr;
		materials[2].Iscatter = no_Iescatter_ptr;
		materials[2].keep_or_kill = 1;

		// Vacuum domain
		materials[3].walls[0] = 6;
		materials[3].walls[1] = 6;
		materials[3].keep_or_kill = 0;


	}

	__device__
	Domain* get_domain(const int& idomain)
	{
		return materials+idomain;
	}

	__device__
	float4 operator()(curandState* random_state)
	{
		float4 result;
		result.x = 2.0*curand_uniform(random_state);


		result.y =  2.0*curand_uniform(random_state)-1.0;

		result.z = 1.0e6f;

		result.w = 1.5f;

		return result;
	}
};

/***************************************************/
/*           		 Problem 2 						   */
class Problem2 : public ProblemSet
{
public:
	__host__
	Problem2()
	{

		for(int i=0;i<n_domains;i++)
		{
			materials[i].ncells = 0;
			materials[i].keep_or_kill = 0;
			if(i==1)
			{
				materials[i].allocate(3*48,84,1);
			}
			else if(i == 2)
			{
				materials[i].allocate(8*48,84,1);
			}
			else if(i == 2)
			{
				materials[i].allocate(1*48,84,1);
			}
			else
			{
				materials[i].allocate(1*48,84,1);
			}
		}
	}

	__device__
	void init()
	{
		for(int i=0;i<n_domains;i++)
		{
			materials[i].SigmaA = infinite_absorber_ptr;
			materials[i].SigmaE = no_SigmaI_ptr;
			materials[i].SigmaI = no_SigmaI_ptr;
			materials[i].Escatter = isoscatter_cmcs_ptr;
			materials[i].Iscatter = no_Iescatter_ptr;
			materials[i].AtomicNumber = 999999999;
		}
		int j;
		// Vacuum domain
		j = 0;
		materials[j].walls[0] = -1.0;
		materials[j].walls[1] = 0.0;
		materials[j].keep_or_kill = 0;


		// Tungsten
		j = 1;
		materials[j].walls[0] = 0.0;
		materials[j].walls[1] = 3.0;
		materials[j].SigmaA = tungsten_SigmaA_ptr;
		materials[j].SigmaE = tungsten_SigmaE_ptr;
		materials[j].SigmaI = tungsten_SigmaI_ptr;
		//materials[j].Iscatter = tungsten_iescatter_ptr;
		materials[j].keep_or_kill = 1;
		materials[j].AtomicNumber = 183.84;



		// Water
		j = 2;
		materials[j].walls[0] = 3.0;
		materials[j].walls[1] = 11.0;
		materials[j].SigmaA = water_SigmaA_ptr;
		materials[j].SigmaE = water_SigmaE_ptr;
		materials[j].keep_or_kill = 1;
		materials[j].AtomicNumber = (18.0*4.0+40.0*1.0)/(40.0+4.0);

		// He3
		j = 3;
		materials[j].walls[0] = 11.0;
		materials[j].walls[1] = 12.0;
		materials[j].SigmaA = He3_SigmaA_ptr;
		materials[j].SigmaE = He3_SigmaE_ptr;
		materials[j].keep_or_kill = 1;
		materials[j].AtomicNumber = 3.0;

		// Vacuum domain
		j = 4;
		materials[j].walls[0] = 12.0;
		materials[j].walls[1] = 13.0;
		materials[j].keep_or_kill = 0;


	}

	__device__
	Domain* get_domain(const int& idomain)
	{
		return materials+idomain;
	}

	__device__
	float4 operator()(curandState* random_state)
	{
		float4 result;

		// Source at x=0;
		result.x = 0.0f;

		// Mono-Directional Source
		result.y =  1.0;

		// 1MeV source
		result.z = 1.0e6;

		// Starts in domain 1
		result.w = 1.5f;

		return result;
	}
};




// Unfortunately due to issues with function pointers we need to do some setup on the gpu
template<class O>
__global__
void setup_problem(O* problem)
{
	int gidx = threadIdx.x+blockIdx.x*blockDim.x;

	if(gidx == 0)
	{
		problem->init();
	}
}

template<class O>
void problem_init(O* problem_h)
{
	O* problem_d;
	cudaMalloc((void**)&problem_d,sizeof(O));

	cudaMemcpy(problem_d,problem_h,sizeof(O),cudaMemcpyHostToDevice);

	CUDA_SAFE_KERNEL((setup_problem<<<1,1>>>(problem_d)));

	cudaMemcpy(problem_h,problem_d,sizeof(O),cudaMemcpyDeviceToHost);

}







__global__
void random_init(curandState* random_states,int seed, int nstates)
{
	int gidx = threadIdx.x+blockDim.x*blockIdx.x;

	if(gidx < nstates)
	{
		curand_init(seed,gidx,gidx,(random_states+gidx));
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

		neutrons.px[gidx+offset] = source_values.x;
		neutrons.mu[gidx+offset] = source_values.y;
		neutrons.energy[gidx+offset] = source_values.z;
		neutrons.domain[gidx+offset] = floor(source_values.w);

		neutrons.weight[gidx+offset] = new_weight;
		neutrons.time_done[gidx+offset] = 0;
		neutrons.dead[gidx+offset] = 0;
		neutrons.finished_time[gidx+offset] = 0;


		gidx += blockDim.x*gridDim.x;

	}

	random_state_g[idx+blockDim.x*blockIdx.x] = random_state;

}

__global__
void neutron_advance(NeutronList neutrons, Domain* materials,curandState* random_state_g,
									 int itally, int iter_in,float time_max,int niter_subcycle)
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

		if((neutrons.dead[gidx]==0)&&(neutrons.time_done[gidx]<time_max))
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
				iter += 1+my_neutron.check_subexit()*(niter_subcycle+1);

			}





			// Write the local data back to global memory
			my_neutron.merge(neutrons);
		}

		gidx += blockDim.x*gridDim.x;

	}


	// Need to save the random state
	random_state_g[threadIdx.x+blockDim.x*blockIdx.x] = random_state;

	__syncthreads();

	if(iter_in >= itally)
	{
	// Write tallies back to global memory
	idx = 1;
	while(idx < 4)
	{
		int idy = threadIdx.x;
		while(idy <= materials_s[idx].ncells)
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
		if(nptcls_new+ListOut.nptcls < ListOut.nptcls_allocated)
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

	int* condition = finished_time;

	cudaMemcpy(finished_time,dead,nptcls_allocated*sizeof(float),cudaMemcpyDeviceToDevice);

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
		for(int i=0;i<NeutronList_nints-1;i++)
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
	//CUDA_SAFE_CALL(cudaMemset(dead,0,nptcls_allocated*sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(finished_time,0,nptcls_allocated*sizeof(int)));


}

__global__
void clear_slots(NeutronList neutrons,int* condition, int value)
{
	int idx = threadIdx.x;
	int gidx = idx+blockDim.x*blockIdx.x;

	while(gidx < neutrons.nptcls_allocated)
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

	weight_avg = weight_total/nptcls;
	//weight_avg = max(weight_avg,weight_low);

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
							float time_max,int itally,int iter_in,
							int niter_subcycle, int nptcls_min)
{
	int cudaGridSize = 42;
	int cudaBlockSize = 256;

	int iter = 0;
	while(neutrons_d.nptcls > 0)
	{
		//printf("Executing super step %i, nptcls left = %i\n",iter,neutrons_d.nptcls);
	/***********************************************/
	/* Advance the current set of neutrons */

	CUDA_SAFE_KERNEL((neutron_advance<<<cudaGridSize,cudaBlockSize>>>
								 (neutrons_d,materials,random_states_d,
								  itally,iter_in,time_max,niter_subcycle)));

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

	//neutrons_d.update_average_weight();


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

		int nptcls_new = min((2*(neutrons_d.nptcls)),(neutrons_d.nptcls_allocated));
		//printf("Doubling the number of particles from %i to %i\n",neutrons_d.nptcls,nptcls_new);
		neutrons_d.nptcls = nptcls_new;
	}
	/***********************************************/



		iter++;

		if(iter > 20)
			break;

	}



}

float2 variance_and_mean(float* array_in,int nelements)
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
	result.y = variance/((float)nelements-1.0);

	return result;
}


template<class Pset>
void MC1D(Pset problem,float* flux_profile,float* x_profile,
		float time_max,int nptcls,int seed)
{


	int nbatches = 40;

	int nptcls_min = nptcls/1.5;
	int niter_subcycle = 5;

	int cudaBlockSize = 256;
	int cudaGridSize = 56;

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

	for(int i=0;i<n_domains;i++)
	{
		domains_h[i] = problem.materials[i];
	}

	//printf("Copying domains to device\n");
	CUDA_SAFE_KERNEL((cudaMemcpy(domains_d,domains_h,n_domains*sizeof(Domain),cudaMemcpyHostToDevice)));

	/***********************************************/

	// Run the simulation
	uint timer;
	cutCreateTimer(&timer);

	float run_time;

	int nsteps = 5;
	int itally = 0;

	float weight = 1.0;

	for(int i=0;i<nsteps;i++)
	{
		cutStartTimer(timer);

		/***********************************************/
		/* Initialize the current set of neutrons */


		//printf("Initializing Neutrons\n");
		cudaMemset(neutrons.buffer,0,nptcls*sizeof(float));
		cudaMemset(neutrons.time_done,0,nptcls*sizeof(float));
		cudaMemset(neutrons.finished_time,0,nptcls*sizeof(float));
		cudaMemset(neutrons.dead,0,nptcls*sizeof(float));
		CUDA_SAFE_KERNEL((neutron_init<<<cudaGridSize,cudaBlockSize>>>
									 (neutrons,problem,random_states,weight,neutrons.nptcls)));


		cudaMemset(neutrons_next.weight,0,nptcls*sizeof(float));
		cudaMemset(neutrons_next.buffer,0,nptcls*sizeof(float));
		cudaMemset(neutrons_next.time_done,0,nptcls*sizeof(float));
		cudaMemset(neutrons_next.finished_time,0,nptcls*sizeof(float));
		cudaMemset(neutrons_next.dead,0,nptcls*sizeof(float));



		neutrons.weight_avg = 1.0;
		neutrons_next.weight_avg = 1.0;
		neutrons.nptcls = neutrons.nptcls_allocated;
		//neutrons.update_average_weight();

		//printf("Average Neutron weight = %f\n", neutrons.weight_avg);
		neutrons_next.nptcls = 0;

		/***********************************************/

		/***********************************************/
		/* Run the superstep */

		super_cycle(neutrons,domains_d,neutrons_next,random_states,time_max,
				i,itally,niter_subcycle,nptcls_min);

		/***********************************************/

		/***********************************************/
		/* Swap the lists */

		if(neutrons_next.nptcls != 0)
		{
			neutrons_next.update_average_weight();

			weight = (1.0*nptcls - neutrons_next.weight_avg*neutrons_next.nptcls)/((float)(nptcls - neutrons_next.nptcls));
			//printf("new_weight = %f\n", weight);
		}
		NeutronList neutrons_swap = neutrons;
		neutrons = neutrons_next;
		neutrons_next = neutrons_swap;

		/***********************************************/




		cutStopTimer(timer);

	}

	run_time = cutGetTimerValue(timer);


printf("Total Run Time was %f ms\n",run_time);





	neutrons.NeutronListFree();
	neutrons_next.NeutronListFree();

	cudaFree(random_states);
	cudaFree(domains_d);

	return;
}


void do_problem1(float* flux_profile,float* x_profile,float& absorptions, float& collisions,int seed)
{
	int nptcls = 1.0e6;

	float time_max = 1.0e-10;
	Problem1 problem;


	problem_init(&problem);

	MC1D(problem,flux_profile,x_profile,time_max,nptcls,seed);

	Domain slab1 = problem.materials[1].copy_to_host();
	Domain slab2 =problem.materials[2].copy_to_host();

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

	absorptions = (slab1.absorptions[0]+slab2.absorptions[0]);
	collisions = (slab2.collisions[0]+slab1.collisions[0]);

	memcpy(flux_profile,slab1.flux_tally,slab1.ncells*sizeof(float));
	memcpy(flux_profile+slab1.ncells,slab2.flux_tally,slab2.ncells*sizeof(float));
}

void do_problem2(float* flux_profile,float* x_profile,float& absorptions, float& collisions,int seed)
{
	int nptcls = 1.0e7;
	float time_max = 1000;
	Problem2 problem;

	problem_init(&problem);

	MC1D(problem,flux_profile,x_profile,time_max,nptcls,seed);

	Domain slab1 = problem.materials[1].copy_to_host();
	Domain slab2 =problem.materials[2].copy_to_host();
	Domain slab3 = problem.materials[3].copy_to_host();

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

	for(int i=0;i<slab3.ncells;i++)
	{
		float dxdc = abs((slab3.walls[1]-slab3.walls[0]))/((float)slab3.ncells);
		x_profile[i+slab1.ncells+slab2.ncells] = i*dxdc + slab3.walls[0];
	}

	absorptions = slab3.absorptions[0]/(nptcls);
	collisions = slab3.collisions[0]/(nptcls);


	memcpy(flux_profile,slab1.flux_tally,(slab1.ncells)*sizeof(float));
	memcpy(flux_profile+slab1.ncells,slab2.flux_tally,(slab2.ncells)*sizeof(float));
	memcpy(flux_profile+slab1.ncells+slab2.ncells,slab3.flux_tally,(slab3.ncells-2)*sizeof(float));
}


int main(void)
{
	cudaSetDevice(1);

	// Allocate some space for the flux profile and x values used in plotting
	float* flux_profile = (float*)malloc(n_domains*512*sizeof(float));
	float* x_profile = (float*)malloc(n_domains*512*sizeof(float));

	int nruns = 30;
	float collisions[nruns];
	float absorptions[nruns];

	gnuplot_ctrl* plot1d;

	plot1d = gnuplot_init();

	int seed = time(NULL);

	srand(seed);

	gnuplot_setstyle(plot1d, "lines");


	for(int i = 0;i<nruns;i++)
	{



	seed = rand()%10928357;

	printf("seed = %i\n",seed);

	// Do problem 1 or problem 2
	//do_problem2(flux_profile,x_profile,absorptions,collisions,seed);



	do_problem2(flux_profile,x_profile,absorptions[i],collisions[i],seed);
	for(int k=0;k<n_domains*512;k++)
	{
		//flux_profile[k] /= 5.0;
	}


	gnuplot_plot_xy(plot1d,x_profile,flux_profile,500,"");
	}

	gnuplot_cmd(plot1d,"set term pdf");
	gnuplot_cmd(plot1d,"set output \"FluxTally2.pdf\"");
	gnuplot_plot_xy(plot1d,x_profile,flux_profile,500,"Flux Tally");

	float2 average_absorb = variance_and_mean(absorptions,nruns);
	float2 average_collide = variance_and_mean(collisions,nruns);

	printf("Average Fraction of Neutrons absorbed: %g +/- %g\n",average_absorb.x,sqrt(average_absorb.y));
	printf("Average Collisions Per Neutron: %g +/- %g\n",average_collide.x,sqrt(average_collide.y));

	printf("Absorption CX / Total Cx for the Detector: %g +/- %g\n",average_absorb.x/average_collide.x,
			sqrt(average_absorb.y)/(average_collide.x+sqrt(average_collide.y)));




	printf("Press 'Enter' to continue\n");
	getchar();
	gnuplot_close(plot1d);
	return 0;
}



























