NVCC		:= /usr/local/cuda/bin/nvcc
#cdLD_LIBRARY_PATH	:= /usr/local/cuda/lib64
CUDA_INCLUDE_PATH	:= /home/josh/lib,/home/josh/lib/cudpp,../cutil/common/inc
CUDAFORTRAN_FLAGS :=  -L/usr/local/cuda/lib64 -lcuda -lcudart -I$(CUDA_INCLUDE_PATH) -L../cutil/lib -lcutil_x86_64
PGPLOT_FLAGS := -L/usr/local/pgplot -lcpgplot -lpgplot -lX11 -lgcc -lm
PGPLOT_DIR = /usr/local/pgplot/
NVCCFLAGS	:=  -m64 -O3 -use_fast_math -ftz=true -prec-div=false -prec-sqrt=false -Xptxas -O3 -Xptxas -maxrregcount=40 -gencode arch=compute_20,code=sm_20 --ptxas-options=-v -ccbin /opt/intel/Compiler/11.1/073/bin/intel64/icc -I$(CUDA_INCLUDE_PATH) 
CC := icc
CPP := icpc

all: randu water_slab
	
water_slab : ../include/gnuplot_i.o water_slab.o
	$(CPP) gnuplot_i.o water_slab.o  $(CUDAFORTRAN_FLAGS) -o water_slab 
	
randu: ../include/gnuplot_i.o randu.o
	$(CPP) gnuplot_i.o randu.o  $(CUDAFORTRAN_FLAGS) -o randu 
	
%.o : %.c
	$(CPP) -c $*.c
	
%.o : %.cpp
	$(CPP) -c $*.cpp
	
%.o : %.cu
	$(NVCC) -c $*.cu $(NVCCFLAGS)
	
clean:
	rm *.o
