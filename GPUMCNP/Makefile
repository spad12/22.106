NVCC		:= /usr/local/cuda/bin/nvcc
#cdLD_LIBRARY_PATH	:= /usr/local/cuda/lib64
CUDA_INCLUDE_PATH	:= /home/josh/lib,/home/josh/lib/cudpp,../cutil/common/inc
CUDAFORTRAN_FLAGS :=  -L/usr/local/cuda/lib64 -lcuda -lcudart -I$(CUDA_INCLUDE_PATH) -L../cutil/lib -lcutil_x86_64
PGPLOT_FLAGS := -L/usr/local/pgplot -lcpgplot -lpgplot -lX11 -lgcc -lm
PGPLOT_DIR = /usr/local/pgplot/
NVCCFLAGS	:=  -m64 -O -use_fast_math -ftz=true -prec-div=false -prec-sqrt=false -gencode arch=compute_20,code=sm_20 --ptxas-options=-v -ccbin /opt/intel/Compiler/11.1/073/bin/intel64/icc -I$(CUDA_INCLUDE_PATH) 
CC := icc
CPP := icpc

OBJ := Populate_NeutronList.o \
				GlobalTally_Move.o \
				CountDeadNeutrons.o \
				Refill_ParticleList.o \
				SharedTally_Move.o \
				SharedTally.o \
				gnuplot_i.o \
				gpumcnp.o \
				NeutronList_Sort.o
				

all: gpumcnp
	
gpumcnp : $(OBJ)
	$(CPP) $(OBJ) $(CUDAFORTRAN_FLAGS) -o gpumcnp 
	
%.o : %.c
	$(CPP) -c $*.c
	
%.o : %.cpp
	$(CPP) -c $*.cpp
	
%.o : %.cu
	$(NVCC) -c $*.cu $(NVCCFLAGS)
	
gnuplot_i.o : /home/josh/CUDA/gnuplot_c/src/gnuplot_i.c
	$(CPP) -c /home/josh/CUDA/gnuplot_c/src/gnuplot_i.c

clean:
	rm *.o
	