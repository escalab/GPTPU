include ../../common/make.config
CUDA_DIR=/usr/local/cuda-10.2
CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = 3D.cu

EXE = 3D 

OUTPUT = *.out

FLAGS = --std=c++11 -O3  -g -G #-arch sm_20 --ptxas-options=-v
release: $(SRC)
	$(CC) $(KERNEL_DIM) $(FLAGS) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lgptpu -ldl -I/opt/OpenBLAS/include -L/opt/OpenBLAS/lib -lopenblas 

enum: $(SRC)
	$(CC) $(KERNEL_DIM) $(FLAGS) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lgptpu -ldl -I/opt/OpenBLAS/include -L/opt/OpenBLAS/lib -lopenblas

debug: $(SRC)
	$(CC) $(KERNEL_DIM) $(FLAGS) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lgptpu -ldl -I/opt/OpenBLAS/include -L/opt/OpenBLAS/lib -lopenblas

debugenum: $(SRC)
	$(CC) $(KERNEL_DIM) $(FLAGS) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lgptpu -ldl -I/opt/OpenBLAS/include -L/opt/OpenBLAS/lib -lopenblas

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo $(OUTPUT) 
