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




class PsetProblem
{
public:
	int ndomains;
	float* dimensions;
	__device__
	float4 operator()(curandState* random_state);
};

