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
#include <float.h>
#include "cuda.h"
#include <cuda_runtime.h>
#include "common_functions.h"
#include "sm_20_intrinsics.h"
#include "host_defines.h"
#include <thrust/sort.h>
#include <thrust/scan.h>
#include "cudamatrix_types.cuh"
#include "vector_functions.h"

#include "include/gnuplot_i.h"


#  define CUDA_SAFE_KERNEL(call) {                                         \
	call;																					\
	cudaDeviceSynchronize();														\
	cudaError err = cudaGetLastError();										\
    if ( cudaSuccess != err) {                                               \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
                exit(EXIT_FAILURE);                                                  \
    } }

#if !defined(CURAND_KERNEL_H_)
extern "C" struct curandState;
#endif

extern "C" class NeutronList;
extern "C" class SimulationData;
extern "C" class MCNeutron;
extern "C" class NeutronSource;
extern "C" class PointSource;
extern "C" class Particlebin;

__constant__ const float Mass_n =  1.04541e-12; //ev cm^-2 s^2
__constant__ float pi_const = 3.14159265;

extern "C" int myid_g;

extern "C" int imethod_step;
extern "C" float step_weights[3][101];
extern "C" float step_nptcls[3][101];

template <typename T>
__host__ __device__ static __inline__
int sgn(T val) {
    return (val > T(0)) - (val < T(0));

}

__device__ static __inline__
bool AlmostEqualRelative(float A, float B)
{
    // Calculate the difference.
    float diff = fabs(A - B);
    A = fabs(A);
    B = fabs(B);
    // Find the largest
    float largest = (B > A) ? B : A;

    if (diff <= largest * 2.0f*FLT_EPSILON)
        return true;
    return false;
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
    u1 = (abs(delta.x) >= 1.0e-9) ? u1:1.0;

    p1 = (origin.y - R00.y)/(-delta.y);
    p2 = (R11.y - origin.y)/delta.y;

    u2 = max(p1,p2);

    // Guard against deltay = 0;
    u2 = (abs(delta.y) >= 1.0e-9) ? u2:1.0;

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

	float b = 2.0f*(delta.x*(origin.x-center.x)+delta.y*(origin.y-center.y));
	float c = center.x*center.x
			+ center.y*center.y
			+ origin.x*origin.x
			+ origin.y*origin.y
			- 2.0f*(center.x*origin.x+center.y*origin.y)
			- radius*radius;

	float discrim = b*b - 4.0f*a*c;

	float u = 1;
	if(discrim >= 0.0f)
	{
		discrim = sqrt(discrim);

		float up = (-b+discrim)/(2.0f*a);
		float um = (-b-discrim)/(2.0f*a);

		// gaurd against negative u's
		up = (up >= 0) ? up:1.0;
		um = (um >= 0) ? um:1.0;

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
extern "C" void GlobalTally_Move(
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
extern "C" void SharedTally_Move(
	SimulationData*			simulation,
	NeutronList*				neutrons,
	NeutronList*				neutrons_next,
	curandState*				random_states,
	int							qScattering
	);

extern "C" void SharedTally(
	SimulationData*			simulation,
	NeutronList*				neutrons,
	NeutronList*				neutrons_next,
	curandState* 				random_states);


extern "C" void Refill_ParticleList(
	NeutronList*				neutrons,
	int							istep
	);

extern "C" void gpumcnp_super_step(
	SimulationData* 		simulation,
	NeutronList* 				neutrons,
	NeutronList* 				neutrons_next,
	curandState*				random_states,
	int 							qGlobalTallies,
	int							qRefillList,
	int							qScattering);

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
	int							seed);

extern "C" void SetGPU(int id);

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
		int							myid);













#endif
