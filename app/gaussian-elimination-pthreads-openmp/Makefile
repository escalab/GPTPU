CC:=g++ --std=c++11 -O3 -g -lgptpu
LD=-lgptpu -ldl -I/opt/OpenBLAS/include -L/opt/OpenBLAS/lib -lopenblas -lpthread

program: gauss.c threads_gauss.c chunk_threads_gauss.c openmp_gauss.c chunk_thpool_gauss.c thpool_gauss.c thpool.c

#	 $(CC) gauss.c -o gauss.out

#	 $(CC) threads_gauss.c -pthread -o threads.out

#	 $(CC) chunk_threads_gauss.c -pthread -o chunkthreads.out

	 $(CC) openmp_gauss.c -fopenmp $(LD)  -o openmp.out

#	 $(CC) chunk_thpool_gauss.c thpool.c -pthread -o chunkpool.out

#	 $(CC) thpool_gauss.c thpool.c -pthread -o pool.out


clean:
	rm -rf *.out
