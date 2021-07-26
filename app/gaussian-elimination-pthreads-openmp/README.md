# Gaussian Elimination

This program is a C implementation to improve performance of a Gaussian Elimination Code, using OpenMP, Pthreads and [Pool of Threads](https://github.com/Pithikos/C-Thread-Pool)

You can compile everything using the Makefile and run the objects created.
```sh
$ make
```
There is a script with alias for running the programs. You can type:
```sh
$ source script.sh
```
Then, type:
```sh
$ help
```
You will see all the alias to run the program. It should be run after `make` and should be with `source`.
You should be able to see:
```
gauss - serial version
threads_gauss - pthread implementation
chunk_threads - pthread implementation with chunks
openmp - OpenMP implementation
pool_threads - Pool of threads implementation
chunk_pool_thread - Pool of threads implementation with chunks
```

Otherwise you can run, one by one using the following commands:

For compile and run the serial version. you can type:

```sh
$ gcc gauss.c -o gauss.out

$ ./gauss.out <matrix_dimensions> [random seed]
```

For compile and run the pthread implementation. you can type:

```sh
$ gcc threads_gauss.c -pthread -o threads.out

$ ./threads.out <matrix_dimensions> [random seed]
```

For compile and run the pthread implementation with chunks. you can type:

```sh
$ gcc chunk_threads_gauss.c -pthread -o chunkthreads.out

$ ./chunkthreads.out <matrix_dimensions> [number of threads] [random seed]
```

For compile and run the OpenMP implementation. you can type:

```sh
$ gcc openmp_gauss.c -fopenmp -o openmp.out

$ ./openmp.out <matrix_dimensions> [random seed]
```

For compile and run the Pool of threads implementation. you can type:

```sh
$ gcc thpool_gauss.c thpool.c -o pool.out

$ ./pool.out <matrix_dimensions> [number of threads] [random seed]
```

For compile and run the Pool of threads implementation with chunks. you can type:

```sh
$ gcc chunk_thpool_gauss.c thpool.c -o chunkpool.out

$ ./chunkpool.out <matrix_dimensions> [number of threads] [random seed]
```
