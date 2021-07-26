#!/bin/bash

gauss()
{
  ./gauss.out $1 $2
}

threads_gauss()
{
  ./threads.out $1 $2
}

chunk_threads()
{
  ./chunkthreads.out $1 $2 $3
}

openmp()
{
  ./openmp.out $1 $2
}

pool_threads()
{
  ./pool.out $1 $2 $3
}

chunk_pool_threads()
{
  ./chunkpool.out $1 $2 $3
}

help()
{
    echo "        "
    echo "        "
    echo "        gauss - serial version"
    echo "        threads_gauss - pthread implementation"
    echo "        chunk_threads - pthread implementation with chunks"
    echo "        openmp - OpenMP implementation"
    echo "        pool_threads - Pool of threads implementation"
    echo "        chunk_pool_thread - Pool of threads implementation with chunks"
    echo "        "
    echo "        "
}
