#include "../include/gnuplot_i.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>



class RanduState
{
private:
	int seed;
	int g;
	int m;
	int c;

	int last_number;

public:

	RanduState(int seed_in){
		seed=seed_in;
		last_number = seed;
		g = 65539;
		m = 31;
		c = 0;
	}

	float uniform(void)
	{
		int sp = (last_number*g+c)%(1<<m);
		last_number = sp;

		return sp/((float)(1<<m));
	}
};


int main(void)
{
	int np3d = rint(1.0e4);
	int np1d = np3d;
	int np2d = np3d;

	printf("Generating 1D, 2D, and 3D plots with %i pts\n",np3d);


	int seed = 1;

	float* x1d = (float*)malloc(np1d*sizeof(float));

	float* x2d= (float*)malloc(np2d*sizeof(float));
	float* y2d= (float*)malloc(np2d*sizeof(float));

	float* x3d = (float*)malloc(np3d*sizeof(float));
	float* y3d = (float*)malloc(np3d*sizeof(float));
	float* z3d = (float*)malloc(np3d*sizeof(float));


	// Initialize the random number generator state
	RanduState randoms0(seed);
	RanduState randoms(seed);

	printf("Generating 1d numbers\n");
	for(int i=0;i<np1d;i++)
	{
		x1d[i] = randoms.uniform();
	}

	randoms = randoms0;
	printf("Generating 2d numbers\n");
	for(int i=0;i<np2d;i++)
	{
		x2d[i] = randoms.uniform();
		y2d[i] = randoms.uniform();
	}

	randoms = randoms0;
	printf("Generating 3d numbers\n");
	for(int i=0;i<np3d;i++)
	{
		x3d[i] = randoms.uniform();
		y3d[i] = randoms.uniform();
		z3d[i] = randoms.uniform();
	}


	gnuplot_ctrl* plot1d;
	gnuplot_ctrl* plot2d;
	gnuplot_ctrl* plot3d;

	plot1d = gnuplot_init();
	plot2d = gnuplot_init();
	plot3d = gnuplot_init();


	gnuplot_setstyle(plot1d, "points");
	gnuplot_setstyle(plot2d, "points");
	gnuplot_setstyle(plot3d, "points");

	gnuplot_cmd(plot1d,"set term pdf");
	gnuplot_cmd(plot1d,"set output \"randu1D.pdf\"");
	gnuplot_cmd(plot2d,"set term pdf");
	gnuplot_cmd(plot2d,"set output \"randu2D.pdf\"");
	gnuplot_cmd(plot3d,"set term pdf");
	gnuplot_cmd(plot3d,"set output \"randu3D.pdf\"");

	// Set the view on the 3d plot to show the hyperplanes
	gnuplot_cmd(plot3d,"set view 61.0, 149.0, 1.0, 1.0");

	gnuplot_plot_x(plot1d,x1d,np1d,"1-D Random Numbers");

	gnuplot_plot_xy(plot2d,x2d,y2d,np2d,"2-D Random Numbers");

	gnuplot_plot_xyz(plot3d,x3d,y3d,z3d,np3d,"3-D Random Numbers");

	printf("Press 'Enter' to continue\n");
	getchar();
	gnuplot_close(plot1d);
	gnuplot_close(plot2d);
	gnuplot_close(plot3d);


}












