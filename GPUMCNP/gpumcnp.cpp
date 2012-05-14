#include <stdlib.h>
#include "mpi.h"
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <ctime>
#include <cstring>
#include <float.h>
#include "math.h"

#include "/home/josh/CUDA/gnuplot_c/src/gnuplot_i.h"

extern "C" int imethod_step;
extern "C" float step_weights[3][101];
extern "C" float step_nptcls[3][101];

int imethod_step;
float step_weights[3][101];
float step_nptcls[3][101];

extern "C" class NeutronList;
extern "C"  class SimulationData;
extern "C"  class MCNeutron;
extern "C"  class NeutronSource;
extern "C"  class PointSource;
extern "C"  class Particlebin;

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

int main(int argc, char *argv[])
{
	int myid;
	int numprocs;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);

	SetGPU(myid%2);
	int qGlobalTallies[3] = {1,1,0};
	int qScattering[3] = {0,0,0};
	int qRefillList[3] = {0,1,1};

	int nptcls = (15.0e6+numprocs-1)/numprocs;
	int nx = 512;
	int ny = 512;
	int nE = 4;

	int seed0 = 209384756*(1+0.5*myid);

	float xdim = 2.0; // cm
	float ydim = 2.0; // cm
	float radius = 1.0; // cm
	float emin = 1.0; // ev
	float emax = 1.0e4; // ev
	float TimeStep = 10; // seconds

	float weight_avg = 1.0;
	float weight_low = 0.1;

	float dV = 2.0*xdim/((float)nx)*2.0*ydim/((float)ny);

	srand(time(NULL)*(myid+1)+seed0);

	int seed = rand()%seed0;

	/********************************/
	/********************************/
	/* Setup the plotting stuff */

	gnuplot_ctrl* plot[3];
	gnuplot_ctrl* weight_plots;
	gnuplot_ctrl* nptcl_plots;
	float* xvals;
	float* yvals;

	float* plotvals = (float*)malloc((nx+1)*(ny+1)*(nE+1)*sizeof(float));
	float* plotvals2 = (float*)malloc((nx+1)*(ny+1)*(nE+1)*sizeof(float));

	float* plotvals_main = (float*)malloc((nx+1)*(ny+1)*(nE+1)*sizeof(float));
	float* plotvals2_main = (float*)malloc((nx+1)*(ny+1)*(nE+1)*sizeof(float));

	float step_weights_main[3][101];
	float step_nptcls_main[3][101];

	for(int i=0;i<3;i++)
	{
		for(int j=0;j<101;j++)
		{
			step_weights[i][j] = 0;
			step_nptcls[i][j] = 0;
		}
	}

	if(myid == 0){



		weight_plots = gnuplot_init();
		nptcl_plots = gnuplot_init();
		gnuplot_cmd(weight_plots,"set xlabel \"istep\"");
		gnuplot_cmd(nptcl_plots,"set xlabel \"istep\"");
		gnuplot_cmd(weight_plots,"set ylabel \"Average Weight\"");
		gnuplot_cmd(nptcl_plots,"set ylabel \"\% of original particles\"");

		gnuplot_cmd(weight_plots,"set xrange [0:100.0]");
		gnuplot_cmd(nptcl_plots,"set xrange [0:100.0]");

		//gnuplot_cmd(weight_plots,"set logscale y");
		//gnuplot_cmd(nptcl_plots,"set logscale y");

		for(int i=0;i<3;i++)
		{
			plot[i] = gnuplot_init();

			gnuplot_cmd(plot[i],"set xlabel \"X\"");
			gnuplot_cmd(plot[i],"set ylabel \"Y\"");

			gnuplot_cmd(plot[i],"set xrange [-2.0:2.0]");
			gnuplot_cmd(plot[i],"set yrange [-2.0:2.0]");

			//gnuplot_cmd(plot[i],"set cbrange [0:0.04]");
		}


		xvals = (float*)malloc((nx+1)*(ny+1)*sizeof(float));
		yvals = (float*)malloc((nx+1)*(ny+1)*sizeof(float));


		for(int i=0;i<(nx+1);i++)
		{
			for(int j=0;j<ny+1;j++)
			{
				xvals[i] = -xdim + (2*xdim)/((float)nx)*i;
				yvals[j] = -ydim + (2*ydim)/((float)ny)*j;
			}
		}
	}


	/********************************/
	NeutronList* neutrons;
	NeutronList* neutrons_next;
	SimulationData* simulation;

	gpumcnp_setup(
			&simulation,&neutrons,&neutrons_next,xdim,ydim,radius,emin,emax,TimeStep,
									weight_avg,weight_low,nptcls,	nx,ny,nE,myid);



	float variances[3]= {0,00};
	float means[3]= {0,00};
	float runtime[3]= {0,00};
	float FOM[3] = {1,1,1};

	char method_names[3][30] = {"Niave","Refill+Niave Tallies","Shared Tallies"};





	for(int i=0;i<3;i++)
	{
		seed = rand()%seed0;
		MPI_Barrier(MPI_COMM_WORLD);

		imethod_step = i;

		gpumcnp_run(
			simulation,
			neutrons,
			neutrons_next,
			plotvals,
			plotvals2,
			runtime+i,
			qGlobalTallies[i],
			qRefillList[i],
			qScattering[i],
			seed);

		MPI_Barrier(MPI_COMM_WORLD);

		MPI_Reduce(plotvals,plotvals_main,(nx+1)*(ny+1)*(nE+1),MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(plotvals2,plotvals2_main,(nx+1)*(ny+1)*(nE+1),MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(step_weights[i],step_weights_main[i],101,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(step_nptcls[i],step_nptcls_main[i],101,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);

		if(myid == 0){

			float mean = 0.0;
			float s2 = 0.0;
			float tvariance = 0.0;
			float tmean = 0.0;

			for(int i=0;i<(nx+1);i++)
			{
				for(int j=0;j<ny+1;j++)
				{
					plotvals_main[i+(nx+1)*j] /= dV*1.0*numprocs*nptcls;
					plotvals2_main[i+(nx+1)*j] /= pow(1.0*numprocs*nptcls,2.0);
					plotvals2_main[i+(nx+1)*j] /= dV*dV;

					mean = plotvals_main[i+(nx+1)*j]/(1.0*numprocs*nptcls);
					s2 = plotvals2_main[i+(nx+1)*j];
					tvariance += (s2/(1.0*nptcls)-mean*mean)/(1.0*numprocs*nptcls-1.0);
					tmean += mean;


				}
			}

			means[i] = tmean;
			variances[i] = tvariance;

			printf("Mean, Variance for step %i = %g, %g\n",i,means[i],variances[i]);

			gnuplot_plot_xyz(plot[i],xvals,yvals,plotvals_main,(nx+1),(ny+1),"");

			gnuplot_plot_x(weight_plots,step_weights[i],100,method_names[i]);
			gnuplot_plot_x(nptcl_plots,step_nptcls[i],100,method_names[i]);
		}

	}


	float mean_total = 0.0;
	float mean_time = 0.0;
	if(myid==0)
	{
		for(int i=0;i<3;i++)
		{
			mean_total += means[i]/3.0;
			mean_time += runtime[i]/3.0;
		}


		for(int i=0;i<3;i++)
		{

			FOM[i] = 1.0/FOM[0];
			float RE = variances[i]*means[i]*means[i]/(mean_total*mean_total);
			FOM[i] *= 1.0/(RE*runtime[i]);

		}

		FOM[0] = 1.0;
		printf("Method FOM: ");
		for(int i=0;i<3;i++)
			printf(" %f ",FOM[i]);

	}

	MPI_Barrier(MPI_COMM_WORLD);

	printf("Press 'Enter' to continue\n");
	getchar();

	if(myid == 0){

		for(int i=0;i<3;i++)
		{
			//gnuplot_cmd(plot[i],"set size square");
			//gnuplot_cmd(plot[i],"set pm3d map");
			//gnuplot_cmd(plot[i],"set samples 50; set isosamples 100");

			char name[30];
			sprintf(name,"flux_profile%i",i);
			gnuplot_save_pdf(plot[i],name);
			gnuplot_close(plot[i]);


		}

		gnuplot_save_pdf(weight_plots,"weight_plots");
		gnuplot_save_pdf(nptcl_plots,"nptcls_plots");
		gnuplot_close(weight_plots);
		gnuplot_close(nptcl_plots);
	}

	MPI_Barrier(MPI_COMM_WORLD);


	MPI_Finalize();

	return 0;

}
