#ifndef __DOMAINS_H__
#define __DOMAINS_H__
#include "gpumcnp.h"

// Base class for material properties
class MaterialData
{
public:
	__device__
	float SigmaA(float energy)
	{
		return 0;
	}

	__device__
	float SigmaF(float energy)
	{
		return 0;
	}

	__device__
	float SigmaES(float energy)
	{
		return 0;
	}

	__device__
	float SigmaIS(float energy)
	{
		return 0;
	}

	__device__
	float SigmaT(float energy)
	{
		return 0;
	}
};

class Water_domain : public MaterialData
{
public:
	__device__
	float SigmaA(float energy)
	{
		return 0.001;
	}

	__device__
	float SigmaF(float energy)
	{
		return 0;
	}

};

class Fuel_domain : public MaterialData
{
public:
	__device__
	float SigmaA(float energy)
	{
		return 0.001;
	}

	__device__
	float SigmaF(float energy)
	{
		return 0.001;
	}

};

class Water_domain2 : public MaterialData
{
public:
	__device__
	float SigmaA(float energy)
	{
		return 1.0;
	}

	__device__
	float SigmaF(float energy)
	{
		return 0;
	}

};



class SimulationData
{
public:
	int nx, ny, ne;
	float xmin,xmax,dcdx,dxdc; // cm
	float ymin,ymax,dcdy,dydc; // cm
	float Emin,didE; // E_i = Emin*10^(i/didE)
	float TimeStep; // Time step size in sec
	int nclusters_x, nclusters_y;

	int CellspClusterx;
	int CellspClustery;

	float PinCenter_x,PinCenter_y,PinRadius;

	float epsilon;

	cudaMatrixf flux_tally;
	cudaMatrixf FinishedTally;

	/********************************/
	/* Source Array Definitions */
	cudaMatrixf ScatterSource;
	cudaMatrixf FissionSource;
	cudaMatrixf FixedSource;
	cudaMatrixf TotalSource; // 1D array to make Prefix Scan easy
	float* invSourceCDF; // Map [0,1] -> [0,nx*ny*ne] using Zorder curve

	/********************************/
	/* Main domain definitions
	 *
	 * 3 - - - 4
	 * |    5 	 |
	 * 1 - - - 2
	 *
	 */
	Water_domain2 domain1;
	Water_domain domain2;
	Water_domain domain3;
	Water_domain2 domain4;
	Fuel_domain domain5;
	/********************************/

	__host__
	void setup(
		float 	xmax_in,
		float 	ymax_in,
		float	PinRadius_in,
		float 	Emin_in,
		float 	Emax_in,
		float 	TimeStep_in)
	{
		xmax = xmax_in;
		ymax = ymax_in;
		xmin = -xmax_in;
		ymin = -ymax_in;



		PinCenter_x = 0.5f*(xmax-xmin) + xmin;
		PinCenter_y = 0.5f*(ymax-ymin) + ymin;
		PinRadius = PinRadius_in;

		Emin = Emin_in;
		didE = ne/log10(Emax_in/Emin);

		dcdx = nx/(xmax-xmin);
		dcdy = ny/(ymax-ymin);
		dxdc = 1.0/dcdx;
		dydc = 1.0/dcdy;

		epsilon = 1.0e-5*sqrt(pow(1.0/dcdx,2)+pow(1.0/dcdy,2));

		TimeStep = TimeStep_in;
		CellspClusterx = 32;
		CellspClustery = 32;
		nclusters_x = (nx+CellspClusterx-1)/CellspClusterx;
		nclusters_y = (ny+CellspClustery-1)/CellspClustery;


	}

	__host__
	void allocate(int nx_in, int ny_in, int nE_in)
	{
		nx = nx_in;
		ny = ny_in;
		ne = nE_in;

		flux_tally.cudaMatrix_allocate(nx+1,ny+1,ne+1);
		FinishedTally.cudaMatrix_allocate(nx+1,ny+1,ne+1);
		ScatterSource.cudaMatrix_allocate(nx+1,ny+1,ne+1);
		FissionSource.cudaMatrix_allocate(nx+1,ny+1,ne+1);
		FixedSource.cudaMatrix_allocate(nx+1,ny+1,ne+1);
		TotalSource.cudaMatrix_allocate(nx+1,ny+1,ne+1);

		CUDA_SAFE_CALL(cudaMalloc((void**)&invSourceCDF,nx*ny*ne*sizeof(float)));
	}


	__device__
	void ptomesh(float x, float y, float energy, int3& icell, float3& cellf)
	{
		cellf.x = (x-xmin)*dcdx;
		cellf.y = (y-ymin)*dcdy;
		cellf.z = log10(energy/Emin)*didE;

		icell.x = floor(cellf.x);
		icell.y = floor(cellf.y);
		icell.z = floor(cellf.z);

		cellf.x -= icell.x;
		cellf.y -= icell.y;
		cellf.z -= icell.z;

	}

	__device__
	void ptomesh_simple(float x, float y, int3& icell, float3& cellf)
	{
		cellf.x = (x-xmin)*dcdx;
		cellf.y = (y-ymin)*dcdy;

		icell.x = floor(cellf.x);
		icell.y = floor(cellf.y);

		cellf.x -= icell.x;
		cellf.y -= icell.y;

	}

	__device__
	void ClusterID2coord(
		float2& 				p00, // (Output) bottom left coordinate
		float2& 				p11, // (Output) top right coordinate
		const short int& 		clusterID // (Input) 1D cluster index
		)
	{
		// Return the x and y coordinates of
		// the clusters bottom left and top
		// right corners
		// result.x = x(0,0)
		// result.y = y(0,0)
		// result.z = x(1,1)
		// result.w = y(1,1)
		int clsidy = clusterID/(nclusters_x);
		int clsidx = clusterID%(nclusters_x);

		float dxdcls = CellspClusterx*(xmax-xmin)/((float)nx);
		float dydcls = CellspClustery*(ymax-ymin)/((float)ny);


		p00.x = (clsidx)*dxdcls+xmin;
		p00.y = (clsidy)*dydcls+ymin;
		p11.x = (clsidx+1)*dxdcls+xmin;
		p11.y = (clsidy+1)*dydcls+ymin;

		// Adjust for ghost cells
		float nghost = 0.5*CellspClusterx+0.5;
		p00.x -= nghost*dxdc;
		p00.y -= nghost*dydc;
		p11.x += nghost*dxdc;
		p11.y += nghost*dydc;


	}

	__device__
	void mDomainID2coord(
		float2& 				p00, // (Output) bottom left coordinate
		float2& 				p11, // (Output) top right coordinate
		const int& 		mDomain // (Input) 1D domain index
		)
	{
		// This only works for domains 1,2,3, and 4


		/*** Note ***/
		/* Should probably convert this
		 * entire method into an array
		 * but I'm lazy and don't feel like
		 * doing that right now, perhaps
		 * I'll get to it later
		 */

		switch(mDomain)
		{
		case 1:
			p00.x = xmin-0.5f*dxdc;
			p00.y = ymin-0.5f*dydc;
			p11.x = xmin+0.5f*(xmax-xmin);
			p11.y = ymin+0.5f*(ymax-ymin);
			break;
		case 2:
			p00.x = xmin+0.5f*(xmax-xmin);
			p00.y = ymin-0.5f*dydc;
			p11.x = xmax+0.5f*dxdc;
			p11.y = ymin+0.5f*(ymax-ymin);
			break;
		case 3:
			p00.x = xmin-0.5f*dxdc;
			p00.y = ymin+0.5f*(ymax-ymin);
			p11.x = xmin+0.5f*(xmax-xmin);
			p11.y = ymax+0.5f*dydc;
			break;
		case 4:
			p00.x = xmin+0.5f*(xmax-xmin);
			p00.y = ymin+0.5f*(ymax-ymin);
			p11.x = xmax+0.5f*dxdc;
			p11.y = ymax+0.5f*dydc;
			break;
		default:
			break;
		}

	}

	__device__
	void updateDomain(
		float2&					origin,
		int&						mDomain)
	{
		int ix,iy,mDomain_temp;
		ix = floor(2*(origin.x-xmin)/(xmax-xmin));
		iy = floor(2*(origin.y-ymin)/(ymax-ymin));
		float r = sqrt(pow(origin.x-PinCenter_x,2)+pow(origin.y-PinCenter_y,2));

		mDomain_temp = ix+iy*2+1;

		if(mDomain == 5)
		{

			if(r >= PinRadius)
			{ // Moving to one of the outer domains
				//int left_or_right = min(1,max((int)floor(0.5*(sgn(origin.x-PinCenter_x)+3))-1,0));
				//int up_or_down = min(1,max((int)floor(0.5*(sgn(origin.y-PinCenter_y)+3))-1,0));

				//mDomain = 2*up_or_down + left_or_right +1;

				mDomain = mDomain_temp;

			}
		}
		else
		{

			if(r <= PinRadius)
			{	// Moving to center domain
				mDomain = 5;

			}
			else
			{
				/*
				float2 p00;
				float2 p11;

				mDomainID2coord(p00,p11,mDomain);

				int imDy = (mDomain-1)/2;
				int imDx = mDomain -1 - 2*imDy;



				if(abs(origin.x - p00.x) <= (p11.x-p00.x)*1.0e-9f)
				{
					imDx -= 1;
				}
				else if(abs(origin.x - p11.x) <= (p11.x-p00.x)*1.0e-9f)
				{
					imDx += 1;
				}

				if(abs(origin.y - p00.y) <= (p11.y-p00.y)*1.0e-9f)
				{
					imDy -= 1;
				}
				else if(abs(origin.y - p11.y) <= (p11.y-p00.y)*1.0e-9f)
				{
					imDy += 1;
				}

				// Handle periodic boundary condition
				imDx = (imDx < 2) ? imDx:(imDx-2);
				imDx = (imDx > 0) ? imDx:(imDx+2);
				imDy = (imDy < 2) ? imDy:(imDy-2);
				imDy = (imDy > 0) ? imDy:(imDy+2);

				mDomain = 2*imDy+imDx+1;
				*/
				mDomain = mDomain_temp;
			}
		}
	}

	__device__
	float SigmaA(float& energy, int& mDomain)
	{
		float result;
		switch(mDomain)
		{
		case 1:
			result = domain1.SigmaA(energy);
			break;
		case 2:
			result = domain2.SigmaA(energy);
			break;
		case 3:
			result = domain3.SigmaA(energy);
			break;
		case 4:
			result = domain4.SigmaA(energy);
			break;
		case 5:
			result = domain5.SigmaA(energy);
			break;
		default:
			break;
		}

		return result;
	}
	__device__
	float SigmaF(float& energy, int& mDomain)
	{
		float result;
		switch(mDomain)
		{
		case 1:
			result = domain1.SigmaF(energy);
			break;
		case 2:
			result = domain2.SigmaF(energy);
			break;
		case 3:
			result = domain3.SigmaF(energy);
			break;
		case 4:
			result = domain4.SigmaF(energy);
			break;
		case 5:
			result = domain5.SigmaF(energy);
			break;
		default:
			break;
		}

		return result;
	}
	__device__
	float SigmaES(float& energy, int& mDomain)
	{
		float result;
		switch(mDomain)
		{
		case 1:
			result = domain1.SigmaES(energy);
			break;
		case 2:
			result = domain2.SigmaES(energy);
			break;
		case 3:
			result = domain3.SigmaES(energy);
			break;
		case 4:
			result = domain4.SigmaES(energy);
			break;
		case 5:
			result = domain5.SigmaES(energy);
			break;
		default:
			break;
		}

		return result;
	}
	__device__
	float SigmaIS(float& energy, int& mDomain)
	{
		float result;
		switch(mDomain)
		{
		case 1:
			result = domain1.SigmaIS(energy);
			break;
		case 2:
			result = domain2.SigmaIS(energy);
			break;
		case 3:
			result = domain3.SigmaIS(energy);
			break;
		case 4:
			result = domain4.SigmaIS(energy);
			break;
		case 5:
			result = domain5.SigmaIS(energy);
			break;
		default:
			break;
		}

		return result;
	}

	__device__
	float SigmaAT(float& energy, int& mDomain)
	{
		return SigmaA(energy,mDomain)+SigmaF(energy,mDomain);
	}

	__device__
	float SigmaST(float& energy, int& mDomain)
	{
		return SigmaIS(energy,mDomain)+SigmaES(energy,mDomain);
	}

	__device__
	float SigmaT(float& energy, int& mDomain)
	{
		return SigmaST(energy,mDomain)+SigmaAT(energy,mDomain);
	}

	__device__
	void AtomicAdd_to_mesh(cudaMatrixf mesh,int3 icell,float3 cellf,float data_in)
	{
		int idx,idy,idz;
		for(int i=0;i<2;i++)
		{
			idx = icell.x+i;

			if(idx > nx)
				idx -= nx+1;
			else if(idy < 0)
				idx += nx+1;

			for(int j=0;j<2;j++)
			{
				idy = icell.y+j;

				if(idy > ny)
					idy -= ny+1;
				else if(idy < 0)
					idy += ny+1;

				for(int k=0;k<1;k++)
				{
					idz = icell.z;
					float weighted_data = (((1.0f-i)+(2*i-1)*cellf.x)*((1.0f-j)+(2*j-1)*cellf.y)*((1.0f-k)+(2*k-1)*cellf.z))*data_in;

					atomicAdd(&(mesh(idx,idy,idz)),weighted_data);
				}
			}
		}
	}


	__device__
	float minDistanceCluster(
		const float2&				origin, // x0, y0
		const float2&				delta, // (x1 - x0), (y1 - y0)
		const int&					clusterID)
	{
		float2 clusterP00;
		float2 clusterP11;

		// Find the positions of the bottom left and top right corners
		ClusterID2coord(clusterP00,clusterP11,clusterID);

		// Return the fraction of the original line where it intersects the
		// cluster boundary.
		return LineRectangle(origin,delta,clusterP00,clusterP11);

	}

	__device__
	float minDistanceDomain(
		const float2&							origin, // x0, y0
		const float2& 							delta, // (x1 - x0), (y1 - y0)
		const int& 								mDomain)
	{
		float u = 1;

		// Every domain has some circular component to it, so might as well test all
		// paths against the pin
		u = LineCircle(origin,delta,make_float2(PinCenter_x,PinCenter_y),PinRadius);

		if(mDomain < 5)
		{
			// We have to figure out whether it intersects the pin or another boundary
			// This can be accomplished by testing both a rectangular boundary
			// and a circular boundary.

			float2 domainP00;
			float2 domainP11;

			// Find the positions of the bottom left and top right corners
			mDomainID2coord(domainP00,domainP11,mDomain);

			u = min(u,LineRectangle(origin,delta,domainP00,domainP11));
		}

		return min(u,1.0f);

	}

	__device__
	void PeriodicBoundary(
		float2&								position,
		int&									mDomain,
		short int&									clusterID)
	{

		// First update the material domain
		updateDomain(position,mDomain);

		if(position.x <= xmin)
			position.x += xmax-xmin;
		else if(position.x >= xmax)
			position.x -= xmax-xmin;

		if(position.y <= ymin)
			position.y += ymax-ymin;
		else if(position.y >= ymax)
			position.y -= ymax-ymin;

		// recalculate the clusterID
		clusterID = calc_clusterID(position.x,position.y);

	}

	__device__
	short int calc_clusterID(float& px,float& py)
	{
		int3 icell;
		float3 cellf;

		ptomesh_simple(px,py,icell,cellf);


		icell.x = (icell.x)/(CellspClusterx);
		icell.y = (icell.y)/(CellspClustery);

		return (icell.x + (nclusters_x)*(icell.y));
	}


};


class NeutronSource
{
public:

	__device__
	void SourceFunction(
		SimulationData& 					simulation,
		MCNeutron&							neutron);
};

// Isotropic Point Source
class PointSource : public NeutronSource
{
public:
	float px;
	float py;
	float energy;
	int mDomain;

	__host__ __device__
	PointSource(){;}

	__host__ __device__
	PointSource(float px_in,float py_in,float energy_in)
	{
		px = px_in;
		py = py_in;
		energy= energy_in;
		mDomain = 5;
	}

	__device__
	void SourceFunction(
		SimulationData& 					simulation,
		MCNeutron&							neutron);
};


























#endif /* !defined(__DOMAINS_H__) */
