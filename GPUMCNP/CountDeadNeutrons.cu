#include "neutron_list.cuh"



__host__
int NeutronList::CountDeadNeutrons(int istep)
{
	int neutrons_dead;

	// Use the buffer array as temporary storage
	CUDA_SAFE_CALL(cudaMemcpy(buffer,dead,nptcls_allocated*sizeof(int),cudaMemcpyDeviceToDevice));

	// Use thrust reduce to count the number of dead neutrons
	thrust::device_ptr<int> reduce_t((int*)buffer);

	neutrons_dead = thrust::reduce(reduce_t,reduce_t+nptcls_allocated);


	thrust::device_ptr<float> weight_t(buffer);

	// for debugging purposes, calculate the current total weight
	CUDA_SAFE_CALL(cudaMemcpy(buffer,weight,nptcls_allocated*sizeof(float),cudaMemcpyDeviceToDevice));
	float weight_temp = thrust::reduce(weight_t,weight_t+nptcls_allocated);

	step_weights[imethod_step][istep] = weight_temp/((float)(nptcls_allocated));
	step_nptcls[imethod_step][istep] = 1.0*(nptcls_allocated-neutrons_dead)/((float)nptcls_allocated);

	//printf("weight = %f\n",weight_temp/((float)nptcls_allocated));

	//weight_avg = weight_temp/((float)(nptcls_allocated-neutrons_dead));

	return neutrons_dead;

}
