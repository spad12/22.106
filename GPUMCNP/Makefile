NVCC		:= /usr/local/cuda/bin/nvcc
#cdLD_LIBRARY_PATH	:= /usr/local/cuda/lib64
CUDA_INCLUDE_PATH	:= ./include,./cutil/common/inc
CUDAFORTRAN_FLAGS :=  -L/usr/local/cuda/lib64 -lcuda -lcudart -I$(CUDA_INCLUDE_PATH) -L./cutil/lib -lcutil_x86_64
PGPLOT_FLAGS := -L/usr/local/pgplot -lcpgplot -lpgplot -lX11 -lgcc -lm
PGPLOT_DIR = /usr/local/pgplot/
#NVCCFLAGS	:=  -m64 -O3 -use_fast_math -ftz=true -prec-div=false -prec-sqrt=false -gencode arch=compute_20,code=sm_20 --ptxas-options=-v -ccbin /opt/intel/Compiler/11.1/073/bin/intel64/icc -I$(CUDA_INCLUDE_PATH) 

NVCCFLAGS	:=  -m64 -O3 -use_fast_math -ftz=true -prec-div=false -prec-sqrt=false -gencode arch=compute_20,code=sm_20 --ptxas-options=-v -ccbin /opt/intel/Compiler/11.1/073/bin/intel64/icc -I$(CUDA_INCLUDE_PATH) 

CC := mpiicc -O3
CPP := mpiicpc -O3

OBJ := Populate_NeutronList.o \
				GlobalTally_Move.o \
				CountDeadNeutrons.o \
				Refill_ParticleList.o \
				SharedTally_Move.o \
				SharedTally.o \
				gnuplot_i.o \
				gpumcnp_step.o \
				gpumcnp.o \
				NeutronList_Sort.o
				

all: gpumcnp
	
gpumcnp : $(OBJ) ./cutil/lib/libcutil_x86_64.a 
	$(CC) $(OBJ) $(CUDAFORTRAN_FLAGS) -o gpumcnp 
	
	
%.o : %.c
	$(CPP) -c $*.c 
	
%.o : %.cpp
	$(CPP) -c $*.cpp
	
%.o : %.cu
	$(NVCC) -c $*.cu $(NVCCFLAGS)
	
gnuplot_i.o : include/gnuplot_i.c
	$(CPP) -c include/gnuplot_i.c
	
./cutil/lib/libcutil_x86_64.a : ./cutil/common/Makefile
	make -C cutil/common CC=icc

clean:
	rm *.o
	
cleanall: clean cleancutil
cleancutil :
	make -C cutil/common clean
	