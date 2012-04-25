#ifndef GPUMCNP_INC
#define GPUMCNP_INC
//#define __cplusplus
//#define __CUDACC__
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <ctime>
#include <cstring>
#include "cuda.h"
#include <cuda_runtime.h>
#include "common_functions.h"
#include "sm_20_intrinsics.h"
#include "host_defines.h"
#include <thrust/sort.h>
#include <thrust/scan.h>
#include "cudamatrix_types.cuh"
#include "vector_functions.h"
#include "curand.h"
#include "curand_kernel.h"
#include "/home/josh/CUDA/gnuplot_c/src/gnuplot_i.h"


#  define CUDA_SAFE_KERNEL(call) {                                         \
	call;																					\
	cudaDeviceSynchronize();														\
	cudaError err = cudaGetLastError();										\
    if ( cudaSuccess != err) {                                               \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
                exit(EXIT_FAILURE);                                                  \
    } }


extern "C" class NeutronList;
extern "C" class SimulationData;
extern "C" class MCNeutron;
extern "C" class NeutronSource;
extern "C" class PointSource;

__constant__ const float Mass_n =  1.04541e-12; //ev cm^-2 s^2
__constant__ float pi_const = 3.14159265;

template <typename T>
__host__ __device__ static __inline__
int sgn(T val) {
    return (val > T(0)) - (val < T(0));

}

static __inline__ __device__
float LineRectangle(
	const float2&			origin, // x0, y0
	const float2&			delta, // (x1 - x0), (y1 - y0)
	const float2&			R00,
	const float2&			R11)
{
	float p1,p2,u1,u2;
	u1 = 1;
	u2 = 1;
    p1 = (origin.x - R00.x)/(-delta.x);
    p2 = (R11.x - origin.x)/delta.x;

    u1 = max(p1,p2);

    // Guard against deltax = 0;
    u1 = (abs(delta.x) >= 1.0e-9f) ? u1:1;

    p1 = (origin.y - R00.y)/(-delta.y);
    p2 = (R11.y - origin.y)/delta.y;

    u2 = max(p1,p2);

    // Guard against deltay = 0;
    u2 = (abs(delta.y) >= 1.0e-9f) ? u2:1;

	// gaurd against negative u's
	u1 = (u1 >= 0) ? u1:1.0f;
	u2 = (u2 >= 0) ? u2:1.0f;

    return min(min(u1,u2),1.0f);

}

static __inline__ __device__
float LineCircle(
	const float2&			origin, // x0, y0
	const float2&			delta, // (x1 - x0), (y1 - y0)
	const float2&			center, // Center of circle
	const float&			radius) // Radius of circle
{

	float a = (delta.x*delta.x+delta.y*delta.y);

	float b = 2*(delta.x*(origin.x-center.x)+delta.y*(origin.y-center.y));
	float c = center.x*center.x
			+ center.y*center.y
			+ origin.x*origin.x
			+ origin.y*origin.y
			- 2*(center.x*origin.x+center.y*origin.y)
			- radius*radius;

	float discrim = b*b - 4*a*c;

	float u = 1;
	if(discrim > 0)
	{
		discrim = sqrt(discrim);

		float up = (-b+discrim)/(2*a);
		float um = (-b-discrim)/(2*a);

		// gaurd against negative u's
		up = (up >= 0) ? up:1;
		um = (um >= 0) ? um:1;

		u = min(up,um);
	}

	return min(u,1.0f);

}

template<class SourceObject>
void Populate_NeutronList(
	SimulationData*			simulation,
	NeutronList*				neutrons,
	curandState*				random_states,
	SourceObject			source
);


/****************************************************/
/* int GlobalTally_Move(
 *
 * 		SimulationData*			simulation,
 * 		(input/output) Provides cross sections
 * 		and domain dimensions. Returns
 * 		tallies.
 *
 *		NeutronList*				neutrons,
 *		(input) Provides neutron position,
 *		velocity, and weight
 *
 *		NeutronList*				neutrons_next,
 *		(output) Returns new neutron position,
 *		velocity, and weight
 *
 * 		int							qRefillList,
 * 		(input) Tells the GPU kernel to only
 * 		run 1 substep if true, otherwise GPU
 * 		move kernel continues to run until
 * 		all particles have been absorbed,
 * 		killed, or finished.
 *
 *		int							qScattering
 *		(input) Flag to enable scattering collisions.
 *		);
 */
/*****************************************************/
void GlobalTally_Move(
	SimulationData*			simulation,
	NeutronList*				neutrons,
	NeutronList*				neutrons_next,
	curandState*				random_states,
	int							qRefillList,
	int							qScattering
	);


/****************************************************/
/* int SharedTally_Move(
 *
 * 		SimulationData*			simulation,
 * 		(input/output) Provides cross sections
 * 		and domain dimensions. Returns
 * 		tallies.
 *
 *		NeutronList*				neutrons,
 *		(input) Provides neutron position,
 *		velocity, and weight
 *
 *		NeutronList*				neutrons_next,
 *		(output) Returns new neutron position,
 *		velocity, and weight
 *
 *		int							qScattering
 *		(input) Flag to enable scattering collisions.
 *		);
 */
/*****************************************************/
void SharedTally_Move(
	SimulationData*			simulation,
	NeutronList*				neutrons,
	NeutronList*				neutrons_next,
	curandState*				random_states,
	int							qScattering
	);

void SharedTally(
	SimulationData*			simulation,
	NeutronList*				neutrons,
	NeutronList*				neutrons_next
	);



void Refill_ParticleList(
	NeutronList*				neutrons
	);

void gpumcnp_super_step(
	SimulationData* 		simulation,
	NeutronList* 				neutrons,
	NeutronList* 				neutrons_next,
	curandState*				random_states,
	int 							qGlobalTallies,
	int							qRefillList,
	int							qScattering);














#endif
