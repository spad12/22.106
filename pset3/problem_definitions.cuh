#include "monte_carlo1d.cuh"

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
		// Vacuum domain
		materials[0].walls[0] = 0;
		materials[0].walls[1] = 0;
		materials[0].keep_or_kill = 0;


		// Slab 1
		materials[1].walls[0] = 0;
		materials[1].walls[1] = 2;
		materials[1].SigmaA = &pset1_slab1_SigmaA;
		materials[1].SigmaE = &pset1_slab1_SigmaE;
		materials[1].SigmaI = &no_SigmaI;
		materials[1].Escatter = &isoscatter_lcs;
		materials[1].Iscatter = &no_Iescatter;
		materials[1].keep_or_kill = 1;



		// Slab 2
		materials[2].walls[0] = 2;
		materials[2].walls[1] = 6;
		materials[2].SigmaA = &pset1_slab1_SigmaA;
		materials[2].SigmaE = &pset1_slab1_SigmaE;
		materials[2].SigmaI = &no_SigmaI;
		materials[2].Escatter = &isoscatter_lcs;
		materials[2].Iscatter = &no_Iescatter;
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

		// The new theta direction
		float theta = pi_const*curand_uniform(random_state);

		// The new phi direction
		float phi = 2.0f*pi_const*curand_uniform(random_state);

		result.y =  cos(phi) * sin(theta);

		result.z = 1000000.0f;

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
		int j;
		for(int i=0;i<n_domains;i++)
		{
			materials[i].ncells = 0;
			materials[i].keep_or_kill = 0;
			materials[i].allocate(256,84,1);
		}
		// Vacuum domain
		j = 0;
		materials[j].walls[0] = 0;
		materials[j].walls[1] = 0;
		materials[j].keep_or_kill = 0;


		// Tungsten
		materials[j].walls[0] = 0;
		materials[j].walls[1] = 2;
		materials[j].SigmaA = &tungsten_SigmaA;
		materials[j].SigmaE = &pset1_slab1_SigmaE;
		materials[j].SigmaI = &no_SigmaI;
		materials[j].keep_or_kill = 1;



		// Water
		materials[j].walls[0] = 2;
		materials[j].walls[1] = 6;
		materials[j].SigmaA = &pset1_slab1_SigmaA;
		materials[j].SigmaE = &pset1_slab1_SigmaE;
		materials[j].SigmaI = &no_SigmaI;
		materials[j].keep_or_kill = 1;

		// He3
		materials[j].walls[0] = 2;
		materials[j].walls[1] = 6;
		materials[j].SigmaA = &pset1_slab1_SigmaA;
		materials[j].SigmaE = &pset1_slab1_SigmaE;
		materials[j].SigmaI = &no_SigmaI;
		materials[j].keep_or_kill = 1;

		// Vacuum domain
		materials[j].walls[0] = 6;
		materials[j].walls[1] = 6;
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
		result.x = 2.0*curand_uniform(random_state);

		// The new theta direction
		float theta = pi_const*curand_uniform(random_state);

		// The new phi direction
		float phi = 2.0f*pi_const*curand_uniform(random_state);

		result.y =  cos(phi) * sin(theta);

		result.z = 1000000.0f;

		result.w = 1.5f;

		return result;
	}
};
