#ifndef __MCNEUTRON_RANDOM__
#define __MCNEUTRON_RANDOM__
/*
 * This is so that we don't have to compile the
 * random number generator stuff for every
 * kernel
 */
#include "cuda.h"

#include "curand.h"
#include "curand_kernel.h"

#include "neutron_list.cuh"
#include "domains.cuh"

__inline__ __device__
float MCNeutron::random(){return curand_uniform(random_state);}


__inline__ __device__
MCNeutron MCNeutron::Advance(
		SimulationData& 	simulation,
		float&					distance,
		const int&			qScattering)
{
	MCNeutron neutron_next = *this;
	float vmag = sqrt(vx*vx
			  +vy*vy
			  +vz*vz);

	float mu;

	mu = sqrt(vx*vx+vy*vy)/vmag;

	float energy = 0.5*Mass_n*vmag*vmag; // eV
	float u = 1;
	float Amass;

	float sigmaS;
	float2 delta;

	//if(!qScattering)
	if(dsince_collide>=(dcollide))
	{
		sigmaS = simulation.SigmaST(energy,mDomain);
		neutron_next.weight = max(weight-weight*abs(simulation.SigmaAT(energy,mDomain)
				/(simulation.SigmaT(energy,mDomain))),0.0);

		weight = neutron_next.weight;


		float muc = max(min(2.0*random() - 1.0f,1.0f),-1.0f);

		float phi = 2.0f*pi_const*random();

		Amass = simulation.AtomicMass[mDomain-1];

		energy *= (Amass*Amass+2.0f*Amass*muc+1.0f)/((Amass+1.0f)*(Amass+1.0f));

		mu = (1.0f+Amass*muc)/sqrt(Amass*Amass+2.0f*Amass*muc+1.0f);

		float su = sqrt(1.0f-mu*mu);

		/*
		vx = vmag*cos(phi);
		vy = vmag*sin(phi);
		vz = 0;

		neutron_next.vx = vx;
		neutron_next.vy = vy;
		neutron_next.vz = vz;

*/
		vx = vx/vmag;
		vy = vy/vmag;
		vz = vz/vmag;

		vmag = sqrt(2.0f*energy/Mass_n);

		neutron_next.vx = (vx*mu+su*(vx*vz*cos(phi)-vy*sin(phi))/sqrt(1.0f-vz*vz))*vmag;
		neutron_next.vy = (vy*mu+su*(vy*vz*cos(phi)+vx*sin(phi))/sqrt(1.0f-vz*vz))*vmag;
		neutron_next.vz = (vz*mu+su*sqrt(1.0f-vz*vz)*cos(phi))*vmag;

		vx = vx*vmag;
		vy = vy*vmag;
		vz = vz*vmag;

		// Scatter, and find the next scatter point
		distance = -log(random())/sigmaS;
		dcollide = distance;
		neutron_next.dcollide = distance;
		neutron_next.dsince_collide = 0.0f;






	}
	else
	{

		distance = neutron_next.dcollide-neutron_next.dsince_collide;
	}

		//mu = sqrt((vx*vx+vy*vy)/(vx*vx+vy*vy+vz*vz));

		distance = abs(distance);

		delta.x = distance*vx/vmag;
		delta.y = distance*vy/vmag;


		neutron_next.px = px + delta.x;
		neutron_next.py = py + delta.y;


		// Guard against going over the time step
		//u = min(1.0f,abs((simulation.TimeStep-time_done)*vmag/distance));

		u = 1.0-FLT_EPSILON;

		u = min(u,simulation.minDistanceCluster(make_float2(px,py),delta,binid));

		u = (u>=0) ? u:1.0;
		u = min(u,simulation.minDistanceDomain(make_float2(px,py),delta,mDomain));

		u = (u>=0) ? u:1.0;
		//printf("delta.x = %f\n",u);

		// The final position,
		neutron_next.px = px + u*delta.x;
		neutron_next.py = py + u*delta.y;
	//}

	neutron_next.time_done += abs(u*distance/(vmag));



	return neutron_next;

}



__inline__ __device__
void MCNeutron::check_domain(
	SimulationData& 					simulation,
	MCNeutron& 							neutron_old)
{

	float sigmaS;
	if(mDomain != neutron_old.mDomain)
	{

		float energy = 0.5*Mass_n*(vx*vx+vy*vy+vz*vz); // eV
		sigmaS = simulation.SigmaST(energy,neutron_old.mDomain);
		weight = max(weight
			-weight*abs((neutron_old.dsince_collide*simulation.SigmaAT(energy,mDomain)
					/(dcollide*simulation.SigmaT(energy,mDomain)))),0.0);






		// Get distance to next scatter
		sigmaS = simulation.SigmaST(energy,mDomain);
		dcollide = -log(random())/sigmaS;
		dcollide = dcollide;//*sqrt((vx*vx+vy*vy)/(vx*vx+vy*vy+vz*vz));
		dsince_collide = 0;

	}

	if(weight <= 0.0f)
	{
		weight = 0.0f;
		dead = 1;
	}
}



__inline__ __device__
void MCNeutron::RussianRoulette(
	const float&		weight_avg,
	const float&		weight_low)
{
	// Russian Roulette


	float weight_new = 0;
	int weight_is_low = (weight <= weight_low) ? 1:0;
	int survived = (random() <= (weight/(weight_avg))) ? 1:0;
	// Avoid Branching by using the fact that true/false are 1 and 0.

	weight_new = (1 - weight_is_low)*(weight);
	weight_new += (weight_is_low)*(survived)*(weight)/(weight_avg+weight);

	dead = (((survived == 0)&&weight_is_low)||(dead != 0)) ? 1:0;
	// If the neutron is dead, weight is 0
	weight = (dead==0) ? weight_new:0.0;

	/*
	if(*weight >= 1.0001)
	{
		printf("Warning Weight > 1.0 avg weight = %f\n",(*weight)/weight_avg);
	}
	*/

}



__inline__ __device__
void PointSource::SourceFunction(
	SimulationData& 					simulation,
	MCNeutron&							neutron)
{
	float theta = 2.0*neutron.random()-1.0;
	float phi = 2.0*pi_const*neutron.random()-pi_const;

	float velocity = sqrt(2.0*energy/Mass_n);

	neutron.vx = velocity*cos(phi)*sqrt(1.0-theta*theta);
	neutron.vy = velocity*sin(phi)*sqrt(1.0-theta*theta);
	neutron.vz = velocity*theta;

	float r = 1.0*simulation.PinRadius*(sqrt(1.0f-(0.999*neutron.random())));
	phi = 2.0*pi_const*neutron.random()-pi_const;
	neutron.px = r*cos(phi);
	neutron.py = r*sin(phi);

	//neutron.px = 0.5*neutron.random()*(simulation.xmax-simulation.xmin)+simulation.xmin;
	//neutron.py =0.5*neutron.random()*(simulation.ymax-simulation.ymin);

	neutron.time_done  = 0;
	neutron.dead = 0;

	neutron.mDomain = mDomain;
	simulation.PeriodicBoundary(neutron.px,neutron.py,
			neutron.mDomain,
			neutron.binid);

	//printf("mdomain = %i\n",neutron.mDomain);

	neutron.binid = simulation.calc_clusterID(neutron.px,neutron.py);

	neutron.weight = 1.0f;

	float energy = 0.5*Mass_n*(neutron.vx*neutron.vx
			+neutron.vy*neutron.vy
			+neutron.vz*neutron.vz); // eV

	// Get distance to next scatter
	float sigmaS = simulation.SigmaST(energy,neutron.mDomain);
	neutron.dcollide = (-log(neutron.random())/sigmaS);
	neutron.dsince_collide = 0;


	float3 cellf;
	int3 icell;
	simulation.ptomesh(neutron.px,neutron.py,energy,icell,cellf);
	//float data_out = -abs(neutron.weight);

	 //write out the tally to gloabal memory
	//simulation.AtomicAdd_to_mesh(simulation.flux_tally,icell,cellf,data_out);



}
#endif
