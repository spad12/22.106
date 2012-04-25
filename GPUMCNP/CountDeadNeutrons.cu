#include "neutron_list.cuh"



__host__
int NeutronList::CountDeadNeutrons(void)
{
	int neutrons_dead;

	// Use the buffer array as temporary storage
	CUDA_SAFE_CALL(cudaMemcpy(buffer,dead,nptcls*sizeof(int),cudaMemcpyDeviceToDevice));

	// Use thrust reduce to count the number of dead neutrons
	thrust::device_ptr<int> reduce_t((int*)buffer);

	neutrons_dead = thrust::reduce(reduce_t,reduce_t+nptcls);

	return neutrons_dead;

}
