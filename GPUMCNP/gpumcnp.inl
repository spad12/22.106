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

	for(int i=0;i<NeutronList_nints-1;i++)
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

	for(int i=0;i<NeutronList_nints-1;i++)
	{
		(*get_int_ptr(i))[thid_out] = child.get_int_ptr(i)[0];
	}

	binid[thid_out] = child.binid;

	return *this;
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
	float dl,dr,dtdl,delt,t,data_out;
	int nsteps_l,istep;

	float weight0 = abs(weight);

	float vmag = sqrt(vx*vx+vy*vy+vz*vz);

	float mu = sqrt((vx*vx+vy*vy)/(vx*vx+vy*vy+vz*vz));

	float energy = 0.5*Mass_n*
			(vx*vx + vy*vy + vz*vz); // eV
	simulation.ptomesh(px,py,energy,icell,cellf);

	// Set up for tallies
	delta.x = px-neutrons_old.px;
	delta.y = py-neutrons_old.py;

	px = neutrons_old.px;
	py = neutrons_old.py;

	float sigmaS = simulation.SigmaST(energy,mDomain);

	float invdistance = simulation.SigmaAT(energy,mDomain)/(dcollide*simulation.SigmaT(energy,mDomain));




	//weight = weight0*(1.0f - abs(dsince_collide*invdistance));


	dr = sqrt(delta.x*delta.x+delta.y*delta.y);

	// Handle the x crossings first

	// Move to the first x surface
	simulation.ptomesh_simple(px,py,icell,cellf);
	dtdl = 1.0/(delta.x);
	dl = (delta.x <= 0) ? (1.0-cellf.x):cellf.x;
	dl *= simulation.dxdc;
	delt = min(abs(dtdl*dl),1.0-FLT_EPSILON);
	t = delt;

	delt = min(abs(dtdl*simulation.dxdc),1.0-FLT_EPSILON);

	istep = 0;
	nsteps_l = 2*ceil(simulation.dcdx*abs(delta.x))+4;

	//t += delt/2.0;

	while(t < (1.0-FLT_EPSILON))
	{
		if(istep > nsteps_l)
			break;

		dl = min(t,1.0-FLT_EPSILON);
		px = (dl+0.5*(dl-(t-delt)))*delta.x + neutrons_old.px;
		py = (dl)*delta.y + neutrons_old.py;
		simulation.ptomesh_simple(px,py,icell,cellf);

		weight = weight0 - weight0*abs((abs((dl)*dr/mu)+dsince_collide)*invdistance);
		data_out = abs(weight*dcollide);
		//data_out = weight;
		simulation.AtomicAdd_to_xsurf(simulation.flux_tally,icell,cellf,data_out);
		simulation.AtomicAdd_to_xsurf(simulation.flux_tally2,icell,cellf,data_out*data_out);

		t += delt;
		istep++;
	}

	weight = weight0;



	px = neutrons_old.px;
	py = neutrons_old.py;

	// Handle the y crossings

	// Move to the first y surface
	simulation.ptomesh_simple(px,py,icell,cellf);
	dtdl = 1.0/(delta.y);
	dl = (delta.y <= 0) ? (1.0-cellf.y):cellf.y;
	dl *= simulation.dydc;
	delt = min(abs(dtdl*dl),1.0-FLT_EPSILON);
	t = delt;

	delt = min(abs(dtdl*simulation.dydc),1.0-FLT_EPSILON);

	istep = 0;
	nsteps_l = 2*ceil(simulation.dcdy*abs(delta.y))+4;

	//t += delt/2.0;

	while(t < (1.0-FLT_EPSILON))
	{
		if(istep > nsteps_l)
			break;

		dl = min(t,1.0-FLT_EPSILON);
		px = (dl)*delta.x + neutrons_old.px;
		py = (dl+0.5*(dl-(t-delt)))*delta.y + neutrons_old.py;
		simulation.ptomesh_simple(px,py,icell,cellf);

		weight = weight0 - weight0*abs((abs((dl)*dr/mu)+dsince_collide)*invdistance);
		data_out =  abs(weight*dcollide);

		//data_out = weight;
		simulation.AtomicAdd_to_ysurf(simulation.flux_tally,icell,cellf,data_out);
		simulation.AtomicAdd_to_ysurf(simulation.flux_tally2,icell,cellf,data_out*data_out);



		t += delt;
		istep++;
	}

	// Set the position back to where it started;
	px = neutrons_old.px + delta.x;
	py = neutrons_old.py + delta.y;
	weight = weight0;

	dsince_collide += abs(dr/mu);

//	if(dsince_collide >= (2*FLT_EPSILON))
	//weight *= (1.0f-abs(dr*simulation.SigmaAT(energy,mDomain)/(dcollide*simulation.SigmaT(energy,mDomain))));

	//weight = weight0 - weight0*abs(dr*invdistance);

/*



	// figure out how many steps to
	// divide the path into
	float l = sqrt(delta.x*delta.x+delta.y*delta.y);
	nsteps_l = floor(2.0f*l/sqrt((pow(simulation.dxdc,2)+pow(simulation.dydc,2))))+1;

	nsteps_l = min(nsteps_l,simulation.CellspClusterx);
	dl = l/((float)nsteps_l);
	float dx = delta.x/((float)nsteps_l);
	float dy = delta.y/((float)nsteps_l);

	px = neutrons_old.px;
	py = neutrons_old.py;



	// Do the tallies
	for(int i=0;i<nsteps_l;i++)
	{
		px += 0.5f*dx;
		py += 0.5f*dy;
		simulation.ptomesh_simple(px,py,icell,cellf);

		//weight = weight0*(1.0f - abs(((i+0.5)*dl)*invdistance));

		data_out = weight*dl*dl/(mu*dcollide);

		// write out the tally to gloabal memory
		simulation.AtomicAdd_to_mesh(simulation.flux_tally,icell,cellf,data_out);

		px += 0.5f*dx;
		py += 0.5f*dy;

		//weight *= 1.0f - dl/distance;

	}


	// Set the position back to where it started;
	px = neutrons_old.px + delta.x;
	py = neutrons_old.py + delta.y;
	weight = weight0;

	weight *= (1.0f-abs(l*simulation.SigmaAT(energy,mDomain)/(dcollide*simulation.SigmaT(energy,mDomain))));

	dsince_collide += l;

	//weight = weight0*(1.0f - abs(l*invdistance));

*/
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

		//simulation.ptomesh_simple(px,py,icell,cellf);

		// write out the tally to gloabal memory
		//simulation.AtomicAdd_to_mesh(simulation.FinishedTally,icell,cellf,weight);


		dead = 1;
		weight = 0;


	}

}

__inline__ __device__
void MCNeutron::STally(
	SimulationData&	simulation,
	MCNeutron&			neutrons_old,
	float*					s1,
	float*					s2,
	float&					distance)
{
	float2 delta;
	int3 icell;
	float3 cellf;
	int2 cluster_offset;
	float dl,dr,dtdl,delt,t,data_out;
	int nsteps_l,istep;

	float weight0 = abs(weight);

	float vmag = sqrt(vx*vx+vy*vy+vz*vz);
	float mu = sqrt((vx*vx+vy*vy)/(vx*vx+vy*vy+vz*vz));

	float energy = 0.5*Mass_n*
			(vx*vx + vy*vy + vz*vz); // eV
	simulation.ptomesh(px,py,energy,icell,cellf);

	// Set up for tallies
	delta.x = px-neutrons_old.px;
	delta.y = py-neutrons_old.py;

	px = neutrons_old.px;
	py = neutrons_old.py;

	float sigmaS = simulation.SigmaST(energy,mDomain);

	float invdistance = simulation.SigmaAT(energy,mDomain)/(dcollide*simulation.SigmaT(energy,mDomain));


	cluster_offset = simulation.cluster_Offsets(binid);

	//weight = weight0*(1.0f - abs(dsince_collide*invdistance));


	dr = sqrt(delta.x*delta.x+delta.y*delta.y);

	// Handle the x crossings first

	// Move to the first x surface
	simulation.ptomesh_simple(px,py,icell,cellf);
	dtdl = 1.0/(delta.x);
	dl = (delta.x <= 0) ? (1.0-cellf.x):cellf.x;
	dl *= simulation.dxdc;
	delt = min(abs(dtdl*dl),1.0-FLT_EPSILON);
	t = delt;

	delt = min(abs(dtdl*simulation.dxdc),1.0-FLT_EPSILON);

	istep = 0;
	nsteps_l = 2*ceil(simulation.dcdx*abs(delta.x))+4;

	//t += delt/2.0;

	while(t < (1.0-FLT_EPSILON))
	{
		if(istep > nsteps_l)
			break;

		dl = min(t,1.0-FLT_EPSILON);
		px = (dl+0.5*(dl-(t-delt)))*delta.x + neutrons_old.px;
		py = (dl)*delta.y + neutrons_old.py;
		simulation.ptomesh_simple(px,py,icell,cellf);

		icell.x -= cluster_offset.x;
		icell.y -= cluster_offset.y;

//		if((icell.x < 0)||(icell.y < 0 )||(icell.x > 36)||(icell.y > 36 ))
//			printf("warning negative cell value %i, %i  in cluster %i %i\n",icell.x,icell.y,cluster_offset.x,cluster_offset.y);

		weight = weight0 - weight0*abs((abs((dl)*dr/mu)+dsince_collide)*invdistance);
		data_out = abs(weight*dcollide);
		//data_out = weight;
		simulation.AtomicAdd_to_Sxsurf(s1,icell,cellf,data_out);
		simulation.AtomicAdd_to_Sxsurf(s2,icell,cellf,data_out*data_out);

		t += delt;
		istep++;
	}

	weight = weight0;



	px = neutrons_old.px;
	py = neutrons_old.py;

	// Handle the y crossings

	// Move to the first y surface
	simulation.ptomesh_simple(px,py,icell,cellf);
	dtdl = 1.0/(delta.y);
	dl = (delta.y <= 0) ? (1.0-cellf.y):cellf.y;
	dl *= simulation.dydc;
	delt = min(abs(dtdl*dl),1.0-FLT_EPSILON);
	t = delt;

	delt = min(abs(dtdl*simulation.dydc),1.0-FLT_EPSILON);

	istep = 0;
	nsteps_l = 2*ceil(simulation.dcdy*abs(delta.y))+4;

	//t += delt/2.0;

	while(t < (1.0-FLT_EPSILON))
	{
		if(istep > nsteps_l)
			break;

		dl = min(t,1.0-FLT_EPSILON);
		px = (dl)*delta.x + neutrons_old.px;
		py = (dl+0.5*(dl-(t-delt)))*delta.y + neutrons_old.py;
		simulation.ptomesh_simple(px,py,icell,cellf);

		icell.x -= cluster_offset.x;
		icell.y -= cluster_offset.y;

//		if((icell.x < 0)||(icell.y < 0 )||(icell.x > 36)||(icell.y > 36 ))
//			printf("warning negative cell value %i, %i  in cluster %i %i\n",icell.x,icell.y,cluster_offset.x,cluster_offset.y);

		weight = weight0 - weight0*abs((abs((dl)*dr/mu)+dsince_collide)*invdistance);
		data_out =  abs(weight*dcollide);

		//data_out = weight;
		simulation.AtomicAdd_to_Sysurf(s1,icell,cellf,data_out);
		simulation.AtomicAdd_to_Sysurf(s2,icell,cellf,data_out*data_out);



		t += delt;
		istep++;
	}

	// Set the position back to where it started;
	px = neutrons_old.px + delta.x;
	py = neutrons_old.py + delta.y;
	weight = weight0;

	dsince_collide += abs(dr/mu);

//	if(dsince_collide >= (2*FLT_EPSILON))
	//weight *= (1.0f-abs(dr*simulation.SigmaAT(energy,mDomain)/(dcollide*simulation.SigmaT(energy,mDomain))));

	//weight = weight0 - weight0*abs(dr*invdistance);

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

		//simulation.ptomesh_simple(px,py,icell,cellf);

		// write out the tally to gloabal memory
		//simulation.AtomicAdd_to_mesh(simulation.FinishedTally,icell,cellf,weight);

		dead = 1;
		weight = 0;

	}

}




#endif /* __GPUMCNP_INL__ */
