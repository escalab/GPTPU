CC       = g++
CXXFLAGS = --std=c++11 -O3 -g
CFLAGS   = -shared -fPIC -I/usr/include/python3.5m/ -fopenmp -lpthread
CFLAGS  += -DCURRENT_DIR=\"$(shell pwd)\"
LINK     = -L/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu -L~/build -Lcnpy
LD       = -lgptpu -ldl -I/opt/OpenBLAS/include -L/opt/OpenBLAS/lib -lopenblas -lpthread
SRC      = $(wildcard ./src/gptpu.cc ./src/make_temp.cc ./src/offset.cc ./src/make_model.cc ./src/utils.cc ./src/fifo.cc)
OBJ      = $(subst ./src, ., $(SRC:.cc=.o))
uname_m:=$(shell uname -m)
all: main app

DIR = /mnt/ramdisk
TMP_DIR = /usr/local/gptpu

main: 
	mkdir -p ./obj ./data
	mkdir -p ./data/conv_model_tflite
app: hotspot3D blackscholes lud gauss pagerank backprop openctpu_demo simple

hotspot3D:
	$(MAKE) -s -C ./app/rodinia_3.1/cuda/hotspot3D

blackscholes:
	$(MAKE) -s -C ./app/pkgs/apps/blackscholes/src

lud:
	$(MAKE) -s -C ./app/rodinia_3.1/openmp/lud/omp

gauss:
	$(MAKE) -s -C ./app/gaussian-elimination-pthreads-openmp/

pagerank: pagerank.o
	$(CC) run_a_pagerank.o $(LD) -o ./obj/run_a_pagerank
pagerank.o:
	$(CC) $(CXXFLAGS) -c ./src/run_a_pagerank.cc 

backprop:
	$(MAKE) -s -C ./app/rodinia_3.1/openmp/backprop

openctpu_demo: openctpu.o
	$(CC) openctpu.o $(LD) -o ./obj/openctpu
openctpu.o: ./src/openctpu.cc
	$(CC) $(CXXFLAGS) -c ./src/openctpu.cc
simple: simple.o
	$(CC) run_a_model.o $(LD) -o ./obj/run_a_model
simple.o:
	$(CC) $(CXXFLAGS) -c ./src/run_a_model.cc

run:
	./app/pkgs/apps/blackscholes/src/blackscholes 1 ./app/pkgs/apps/blackscholes/src/input_1024x1024.txt ./app/pkgs/apps/blackscholes/src/out.txt	
	./app/gaussian-elimination-pthreads-openmp/openmp.out 16 1 1
	./app/rodinia_3.1/openmp/lud/omp/lud_omp -s 1024
	./app/rodinia_3.1/openmp/backprop/backprop 1024
	./obj/openctpu
	./app/rodinia_3.1/cuda/hotspot3D/3D 256 1 1 ./app/rodinia_3.1/data/hotspot3D/power_8192x8 ./app/rodinia_3.1/data/hotspot3D/temp_8192x8 output.out
	./obj/run_a_pagerank ./src/pagerank_1K_iter1_edgetpu.tflite 1 1024

clean:
	rm -f ./*.o ./obj/* ./libdense.so ./libgptpu.so ./libdense_arm.so
	rm -rf ./data ./edgetpu/bazel-* ./edgetpu/build ./edgetpu/dist ./edgetpu/edgetpu/swig/*.so ./edgetpu/edgetpu/swig/edgetpu_cpp_wrapper.py
#	sudo rm -rf $(TMP_DIR)/*
#	sudo rm -rf $(DIR)/*
	rm -rf ./edgetpu/out 

