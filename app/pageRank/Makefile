CC = g++
CXXFLAGS = --std=c++11 -O3 -g
LD = -lgptpu -ldl -I/opt/OpenBLAS/include -L/opt/OpenBLAS/lib -lopenblas -fopenmp
SRC = ./pagerank.cc
EXE = ./pagerank
SRC_TEST= ./get_iteration.cc
EXE_TEST= ./get_iteration

all: main test
	
main:
	$(CC) $(CXXFLAGS) $(SRC) $(LD) -o $(EXE)
test:
	$(CC) $(CXXFLAGS) $(SRC_TEST) $(LD) -o $(EXE_TEST)
clean:
	rm -rf $(EXE)

