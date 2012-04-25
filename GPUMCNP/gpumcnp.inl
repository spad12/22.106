#ifndef __GPUMCNP_INL__
#define  __GPUMCNP_INL__
#include "neutron_list.cuh"
#include "domains.cuh"


__inline__ __device__
MCNeutron& MCNeutron::operator=(NeutronList& parents)
{
	for(int i=0;i<NeutronList_nfloats-1;i++)
	{
		(*get_float_ptr(i)) = parents.get_float_ptr(i)[0][*thid];
	}

	for(int i=0;i<NeutronList_nints;i++)
	{
		(*get_int_ptr(i)) = parents.get_int_ptr(i)[0][*thid];
	}

	binid = parents.binid[*thid];

	return *this;
}

__inline__ __device__
NeutronList& NeutronList::operator=(MCNeutron& child)
{
	int thid_out = *(child.thid);
	for(int i=0;i<NeutronList_nfloats-1;i++)
	{
		(*get_float_ptr(i))[thid_out] = child.get_float_ptr(i)[0];
	}

	for(int i=0;i<NeutronList_nints;i++)
	{
		(*get_int_ptr(i))[thid_out] = child.get_int_ptr(i)[0];
	}

	binid[thid_out] = child.binid;

	return *this;
}



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

	float energy = 0.5*Mass_n*vmag*vmag; // eV
	float u = 1;

	float sigmaA;
	float2 delta;

	//if(!qScattering)
	//{// Absorption only
		sigmaA = simulation.SigmaAT(energy,mDomain);
		distance = -log(random())/sigmaA;

		neutron_next.px = px + distance*vx/vmag;
		neutron_next.py = py + distance*vy/vmag;


		delta.x =neutron_next.px-px;
		delta.y = neutron_next.py-py;

		// Guard against going over the time step
		u = min(1.0f,abs((simulation.TimeStep-time_done)*vmag/distance));

		u = min(u,simulation.minDistanceCluster(position(),delta,binid));

		u = min(u,simulation.minDistanceDomain(position(),delta,mDomain));

		//printf("delta.x = %f\n",u);

		// The final position,
		neutron_next.px = px + u*delta.x*(1.000001f);
		neutron_next.py = py + u*delta.y*(1.000001f);
	//}

	neutron_next.time_done += abs((neutron_next.px-px)/vx);

	return neutron_next;

}

__inline__ __device__
void MCNeutron::Tally(
	SimulationData&	simulation,
	MCNeutron&			neutrons_old,
	float&					distance)
{
	float2 delta;
	int3 icell;
	float3 cellf;
	float l,dl,dx,dy;
	int nsteps_l;

	float weight0 = weight;

	float energy = 0.5*Mass_n*
			(vx*vx + vy*vy + vz*vz); // eV
	simulation.ptomesh(px,py,energy,icell,cellf);

	// Set up for tallies
	delta.x = px-neutrons_old.px;
	delta.y = py-neutrons_old.py;

	// figure out how many steps to
	// divide the path into
	l = sqrt(delta.x*delta.x+delta.y*delta.y);
	nsteps_l = floor(2.0f*l/sqrt((pow(simulation.dxdc,2)+pow(simulation.dydc,2))))+1;

	dl = l/((float)nsteps_l);
	dx = delta.x/((float)nsteps_l);
	dy = delta.y/((float)nsteps_l);

	px = neutrons_old.px;
	py = neutrons_old.py;

	float data_out;

	// Do the tallies
	for(int i=0;i<nsteps_l;i++)
	{
		px += 0.5f*dx;
		py += 0.5f*dy;
		simulation.ptomesh_simple(px,py,icell,cellf);

		data_out = weight*dl;

		// write out the tally to gloabal memory
		simulation.AtomicAdd_to_mesh(simulation.flux_tally,icell,cellf,data_out);

		px += 0.5f*dx;
		py += 0.5f*dy;

		//weight *= 1.0f - dl/distance;

	}

	// Set the position back to where it started;
	px = neutrons_old.px + delta.x;
	py = neutrons_old.py + delta.y;

	weight = weight0*(1.0f - l/distance);
	if(weight <= 0.0f)
	{
		weight = 0.0f;
		dead = 1;
	}



	if(time_done >= simulation.TimeStep)
	{
		// Dump all the particle weight at current location
		// So that it gets picked up as a source for the
		// next iteration

		simulation.ptomesh_simple(px,py,icell,cellf);

		// write out the tally to gloabal memory
		simulation.AtomicAdd_to_mesh(simulation.FinishedTally,icell,cellf,weight);

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
	int survived = (random() < ((weight)/weight_avg)) ? 1:0;
	// Avoid Branching by using the fact that true/false are 1 and 0.

	weight_new = (1 - weight_is_low)*(weight);
	weight_new += (weight_is_low)*survived*(weight)/weight_avg;

	dead = (1-survived)||(dead == 1);
	// If the neutron is dead, weight is 0
	weight = weight_new*((1 - (dead)));

	/*
	if(*weight >= 1.0001)
	{
		printf("Warning Weight > 1.0 avg weight = %f\n",(*weight)/weight_avg);
	}
	*/

}

__inline__ __device__
int MCNeutron::check_subexit(SimulationData* simulation)
{

}



__inline__ __device__
void PointSource::SourceFunction(
	SimulationData& 					simulation,
	MCNeutron&							neutron)
{
	float theta = neutron.random();
	float phi = 2.0*pi_const*neutron.random();

	float velocity = sqrt(2.0f*energy/Mass_n);

	neutron.vx = velocity*cos(phi)*(theta);
	neutron.vy = velocity*sin(phi)*(theta);
	neutron.vz = velocity*(1.0-theta);

	float r = 0.8*simulation.PinRadius*(sqrt(1.0f-(neutron.random())));
	phi = 2*pi_const*neutron.random();
	neutron.px = r*cos(phi);
	neutron.py = r*sin(phi);

	neutron.time_done  = 0;
	neutron.dead = 0;

	neutron.binid = simulation.calc_clusterID(neutron.px,neutron.py);

	neutron.mDomain = mDomain;
	neutron.weight = 1.0f;

}

#endif /* __GPUMCNP_INL__ */
