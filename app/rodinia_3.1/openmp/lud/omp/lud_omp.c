#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <chrono>
#include <gptpu.h>

extern int omp_num_threads;
inline int next_p2(int n){
  int i = 0;
  for(--n ; n > 0 ; n >>= 1){
    i++;
  }
  return 1<< i;
}
#define AA(_i,_j) a[offset*size+_i*size+_j+offset]
#define BB(_i,_j) a[_i*size+_j]

#ifdef OMP_OFFLOAD
#pragma offload_attribute(push, target(mic))
#endif

typedef std::chrono::time_point<std::chrono::high_resolution_clock> timing;
typedef std::chrono::high_resolution_clock clk;


void lud_diagonal_omp (int* a, int size, int offset, int BS)
{
    int i, j, k;
    for (i = 0; i < BS; i++) {

        for (j = i; j < BS; j++) {
            for (k = 0; k < i ; k++) {
                AA(i,j) = AA(i,j) - AA(i,k) * AA(k,j);
            }
        }
   
        int temp = 1.f/AA(i,i);
        for (j = i+1; j < BS; j++) {
            for (k = 0; k < i ; k++) {
                AA(j,i) = AA(j,i) - AA(j,k) * AA(k,i);
            }
            AA(j,i) = AA(j,i)*temp;
        }
    }

}

#ifdef OMP_OFFLOAD
#pragma offload_attribute(pop)
#endif


void lud_diagonal_gptpu (int* a, int size, int offset, int BS)
{
    set_exact_mode(1);
    int i, j, k;
    int* temp1 = (int*)malloc(BS*BS*sizeof(int));
    int* temp2 = (int*)malloc(BS*BS*sizeof(int));
    int sum = 0;
    int crop_cnt = 0, mac_cnt = 0;
    for (i = 0; i < BS; i++) {
       gptpu_crop(a, temp1, size, size, 1, i, i, 0, false);
//        gptpu_crop(a, temp2, size, size, i, BS-i, 0, i, true);
       for (j = i; j < BS; j++) {
            gptpu_crop(a, temp2, size, size, i, 1, 0, j, false);
            std::cout << "i: " << i << ", j: " << j << ", sum: " << sum << std::endl;
            gptpu_mac(temp1, /*&temp2[0]*/temp2, sum, i);
            AA(i,j) -= sum;
//            for (k = 0; k < i ; k++) {
//                AA(i,j) = AA(i,j) - AA(i,k) * AA(k,j);
//            }
        }
        int temp = 1.f/AA(i,i);
        for (j = i+1; j < BS; j++) {
            for (k = 0; k < i ; k++) {
                AA(j,i) = AA(j,i) - AA(j,k) * AA(k,i);
            }
            AA(j,i) = AA(j,i)*temp;
        }
    }

}


// implements block LU factorization 
void lud_omp_int(int *a, int size, int BS)
{
    int offset, chunk_idx, size_inter, chunks_in_inter_row, chunks_per_inter;
int idx = 0;
//    int* temp = (int*)malloc(BS*BS*sizeof(int));
//    int* temp_top  = (int*)malloc(BS*BS*sizeof(int));
//    int* temp_left = (int*)malloc(BS*BS*sizeof(int));
//    int* sum       = (int*)malloc(BS*sizeof(int));
#ifdef OMP_OFFLOAD
#pragma omp target map(to: size) map(a[0:size*size])
#endif

#ifdef OMP_OFFLOAD
{
    omp_set_num_threads(224);
#else
    printf("running OMP on host\n");
    omp_set_num_threads(omp_num_threads);
#endif
    timing mms, mme, dia_s, dia_e, prep_s, prep_e, top_s, top_e, left_s, left_e, blk_s, blk_e;
    double mm_us = 0, dia_us = 0, prep_us = 0, top_us = 0, left_us = 0, blk_us = 0;
    for (offset = 0; offset < size - BS ; offset += BS)
    {
        // lu factorization of left-top corner block diagonal matrix 
        //
        dia_s = clk::now();
        lud_diagonal_omp(a, size, offset, BS);
        dia_e = clk::now();
        dia_us += std::chrono::duration_cast<std::chrono::nanoseconds>(dia_e - dia_s).count()/1000.0;
            
        size_inter = size - offset -  BS;
        chunks_in_inter_row  = size_inter/BS;
        
        // calculate perimeter block matrices
        // 
#pragma omp parallel for default(none) \
          private(chunk_idx) shared(size, BS, chunks_per_inter, chunks_in_inter_row, offset, a, prep_s, prep_e, prep_us, top_s, top_e, top_us, left_s, left_e, left_us) 
        for ( chunk_idx = 0; chunk_idx < chunks_in_inter_row; chunk_idx++)
        {
            int i, j, k, i_global, j_global, i_here, j_here;
            int sum;           
            int temp[BS*BS] __attribute__ ((aligned (64)));
            prep_s = clk::now();
            for (i = 0; i < BS; i++) {
                #pragma omp simd
                for (j =0; j < BS; j++){
                    temp[i*BS + j] = a[size*(i + offset) + offset + j ];
                }
            }
            prep_e = clk::now();
            prep_us += std::chrono::duration_cast<std::chrono::nanoseconds>(prep_e - prep_s).count()/1000.0;
            i_global = offset;
            j_global = offset;
            
            // processing top perimeter
            //
            j_global += BS * (chunk_idx+1);
            top_s = clk::now();
            for (j = 0; j < BS; j++) {
                for (i = 0; i < BS; i++) {
                    sum = 0.f;
                    for (k=0; k < i; k++) {
                        sum += temp[BS*i +k] * BB((i_global+k),(j_global+j));
                    }
                    i_here = i_global + i;
                    j_here = j_global + j;
                    BB(i_here, j_here) = BB(i_here,j_here) - sum;
                }
            }
            top_e = clk::now();
            top_us += std::chrono::duration_cast<std::chrono::nanoseconds>(top_e - top_s).count()/1000.0;

            // processing left perimeter
            //
            j_global = offset;
            i_global += BS * (chunk_idx + 1);
            left_s = clk::now();
            for (i = 0; i < BS; i++) {
                for (j = 0; j < BS; j++) {
                    sum = 0.f;
                    for (k=0; k < j; k++) {
                        sum += BB((i_global+i),(j_global+k)) * temp[BS*k + j];
                    }
                    i_here = i_global + i;
                    j_here = j_global + j;
//printf("a{(%d+%d), (%d+%d)}: %d\n", offset, j, offset, j, a[size*(offset+j)+offset+j]);
                    a[size*i_here + j_here] = ( a[size*i_here+j_here] - sum ) / a[size*(offset+j) + offset+j];
                }
            }
            left_e = clk::now();
            left_us += std::chrono::duration_cast<std::chrono::nanoseconds>(left_e - left_s).count()/1000.0;

        }
        // update interior block matrices
        //
        chunks_per_inter = chunks_in_inter_row*chunks_in_inter_row;

        blk_s = clk::now();
#pragma omp parallel for schedule(auto) default(none) \
         private(chunk_idx ) shared(idx, size, BS, chunks_per_inter, chunks_in_inter_row, offset, a, blk_s, blk_e, blk_us, mm_us, mms, mme) 
        for  (chunk_idx =0; chunk_idx < chunks_per_inter; chunk_idx++)
        {
//printf("idx: %d\n", idx);
idx++;
            int i, j, k, i_global, j_global;
            int temp_top[BS*BS] __attribute__ ((aligned (64)));
            int temp_left[BS*BS] __attribute__ ((aligned (64)));
            int sum[BS] __attribute__ ((aligned (64))) = {0};
            
            i_global = offset + BS * (1 +  chunk_idx/chunks_in_inter_row);
            j_global = offset + BS * (1 + chunk_idx%chunks_in_inter_row);

            for (i = 0; i < BS; i++) {
#pragma omp simd
                for (j =0; j < BS; j++){
                    temp_top[i*BS + j]  = a[size*(i + offset) + j + j_global ];
                    temp_left[i*BS + j] = a[size*(i + i_global) + offset + j];
                }
            }
            for (i = 0; i < BS; i++)
            {
                mms = clk::now();
                for (k=0; k < BS; k++) {
#pragma omp simd 
                    for (j = 0; j < BS; j++) {
                        sum[j] += temp_left[BS*i + k] * temp_top[BS*k + j];
                    }
                }
                mme = clk::now();
                mm_us += std::chrono::duration_cast<std::chrono::nanoseconds>(mme - mms).count()/1000.0;
#pragma omp simd 
                for (j = 0; j < BS; j++) {
                    BB((i+i_global),(j+j_global)) -= sum[j];
                    sum[j] = 0.f;
                }
            }
        }
        blk_e = clk::now();
        blk_us += std::chrono::duration_cast<std::chrono::nanoseconds>(blk_e - blk_s).count()/1000.0;
    }
    dia_s = clk::now();
    lud_diagonal_omp(a, size, offset, BS);
    dia_e = clk::now();
    dia_us += std::chrono::duration_cast<std::chrono::nanoseconds>(dia_e - dia_s).count()/1000.0;
#ifdef OMP_OFFLOAD
}
#endif
  printf("dia  time: %12.3f (us) \n", dia_us);
  printf("prep time: %12.3f (us) \n", prep_us);
  printf("top  time: %12.3f (us) \n", top_us);
  printf("left time: %12.3f (us) \n", left_us);
  printf("blk  time: %12.3f (us) \n", blk_us);
  printf("mm   time: %12.3f (us) ~\n", mm_us);

}


// implements block LU factorization 
void lud_gptpu(int *a, int size, int BS)
{
    printf("start to run lud_gptpu\n");
    int* temp      = (int*)malloc(BS*BS*sizeof(int));
    int* temp_BB   = (int*)malloc(BS*BS*sizeof(int));
    int* temp_temp = (int*)malloc(BS*sizeof(int));

    int* temp_top  = (int*)malloc(BS*BS*sizeof(int));
    int* temp_left = (int*)malloc(BS*BS*sizeof(int));
    int* Sum       = (int*)malloc(BS*BS*sizeof(int));

    timing crop_s, crop_e, mac_s, mac_e, mm_s, mm_e;
    double crop_us = 0, mac_us = 0, mm_us = 0;
    int crop_cnt = 0, mac_cnt = 0, mm_cnt = 0;
    int offset, chunk_idx, size_inter, chunks_in_inter_row, chunks_per_inter;
    set_dev_cnt(1);
 //   set_block(1, 1024); 
#ifdef OMP_OFFLOAD
#pragma omp target map(to: size) map(a[0:size*size])
#endif

#ifdef OMP_OFFLOAD
{
    omp_set_num_threads(224);
#else
    printf("running OMP on host\n");
    omp_set_num_threads(omp_num_threads);
#endif
    std::cout << "size: " << size << ", BS : " << BS << std::endl;
    timing dia_s, dia_e, top_s, top_e, left_s, left_e, blk_s, blk_e;
    double dia_us = 0, top_us = 0, left_us = 0, blk_us = 0;
    for (offset = 0; offset < size - BS ; offset += BS)
    {
        // lu factorization of left-top corner block diagonal matrix 
        //
        dia_s = clk::now();
        lud_diagonal_omp/*gptpu*/(a, size, offset, BS);
        dia_e = clk::now();
        dia_us += std::chrono::duration_cast<std::chrono::nanoseconds>(dia_e - dia_s).count()/1000.0;
            
        size_inter = size - offset -  BS;
        chunks_in_inter_row  = size_inter/BS;
        
        // calculate perimeter block matrices
        // 
//        #pragma omp parallel for default(none) \
          private(chunk_idx) shared(size, chunks_per_inter, chunks_in_inter_row, offset, a) 
        for ( chunk_idx = 0; chunk_idx < chunks_in_inter_row; chunk_idx++)
        {
            int i, j, k, i_global, j_global, i_here, j_here;
            int sum;           
            //int temp[BS*BS] __attribute__ ((aligned (64)));
            crop_s = clk::now();
            gptpu_crop(a, temp, size, size, BS, BS, offset, offset, false);
            gptpu_crop(a, temp, size, size, BS, 1 , offset, offset, false);
            crop_e = clk::now();
            crop_cnt+=4;
            crop_us += 2*std::chrono::duration_cast<std::chrono::nanoseconds>(crop_e - crop_s).count()/1000.0;
//            for (i = 0; i < BS; i++) {
//                #pragma omp simd
//                for (j =0; j < BS; j++){
//                    temp[i*BS + j] = a[size*(i + offset) + offset + j ];
//                }
//            }
            i_global = offset;
            j_global = offset;
            // processing top perimeter
            //
            j_global += BS * (chunk_idx+1);
            top_s = clk::now();
            for (j = 0; j < BS; j++) {
                for (i = 0; i < BS; i++) {
                    sum = 0;
//                    set_block(1, next_p2(i));
//                    gptpu_crop(a, temp_BB, size, size, i, 1, i_global, j_global+j, false);
 //                   gptpu_mac(temp+(BS*i), temp_BB, sum, i);
                    for (k=0; k < i ; k++) {
                        sum += temp[BS*i +k] * BB((i_global+k),(j_global+j));
                    }
                    i_here = i_global + i;
                    j_here = j_global + j;
                    BB(i_here, j_here) = BB(i_here,j_here) - sum;
                }
            }
/*
            //ideally, the loops above can be viewd as:
            for(j = 0 ; j < BS; j++){
              gptpu_mv(A, B, C); where A(BS, BS): temp, B(BS, 1): BB(i_global:i_global+BS, j_global), C(1,BS): for BB(i_global+i, j_global:j_global+BS)  
            }
*/
            std::cout << "top, BS: " << BS << "chunk_idx:" << chunk_idx << std::endl;

            top_e = clk::now();
            top_us += std::chrono::duration_cast<std::chrono::nanoseconds>(top_e - top_s).count()/1000.0;
            // processing left perimeter
            //
            j_global = offset;
            i_global += BS * (chunk_idx + 1);
            left_s = clk::now();
            for (i = 0; i < BS; i++) {
                for (j = 0; j < BS; j++) {
                    sum = 0;
//                    set_block(1, next_p2(i));
//                    gptpu_crop(a, temp_BB, size, size, 1, j, i_global, j_global, false);
//                    gptpu_crop(temp, temp_temp, BS, BS, j, 1, 0, j, false);
//                    gptpu_mac(temp_BB, temp_temp, sum, j);
                    for (k=0; k < j; k++) {
                        sum += BB((i_global+i),(j_global+k)) * temp[BS*k + j];
                    }
                    i_here = i_global + i;
                    j_here = j_global + j;
//                    BB(i_here, j_here) = (BB(i_here, j_here) - sum) / AA(j, j);
                    a[size*i_here + j_here] = ( a[size*i_here+j_here] - sum ) / a[size*(offset+j) + offset+j];
                }
            }
/*
            ideally, the alove loops can be viewd as:
	    for(i = 0 ; i < BS ; i++){
              gptpu_mv(A, B, C); where A(1, BS): BB(i_global+i,j_global:j_global+BS), B(BS, BS): temp(BS, BS), C(1, BS): (i_global+i, BS)
            }
*/
            std::cout << "left, BS: " << BS << "chunk_idx:" << chunk_idx << std::endl;
    
            left_e = clk::now();
            left_us += std::chrono::duration_cast<std::chrono::nanoseconds>(left_e - left_s).count()/1000.0;

        }
        
        // update interior block matrices
        //
        chunks_per_inter = chunks_in_inter_row*chunks_in_inter_row;

//#pragma omp parallel for schedule(auto) default(none) \
         private(chunk_idx ) shared(size, chunks_per_inter, chunks_in_inter_row, offset, a) 
        blk_s = clk::now();
        for  (chunk_idx =0; chunk_idx < chunks_per_inter; chunk_idx++)
        {
            int i, j, k, i_global, j_global;
//            int temp_top[BS*BS] __attribute__ ((aligned (64)));
//            int temp_left[BS*BS] __attribute__ ((aligned (64)));
//            int Sum[BS*BS] __attribute__ ((aligned (64))) = {0};
            
            i_global = offset + BS * (1 +  chunk_idx/chunks_in_inter_row);
            j_global = offset + BS * (1 + chunk_idx%chunks_in_inter_row);

//            for (i = 0; i < BS; i++) {
//#pragma omp simd
//                for (j =0; j < BS; j++){
//                    temp_top[i*BS + j]  = a[size*(i + offset) + j + j_global ];
//                    temp_left[i*BS + j] = a[size*(i + i_global) + offset + j];
//                }
//            }
            crop_s = clk::now();
            gptpu_crop(a, temp_top,  size, size, BS, BS, offset, j_global, false);
            gptpu_crop(a, temp_left, size, size, BS, BS, i_global, offset, false);
            crop_e = clk::now();
            crop_cnt+=2; 
            crop_us += std::chrono::duration_cast<std::chrono::nanoseconds>(crop_e - crop_s).count()/1000.0;

//            for (i = 0; i < BS; i++)
//            {
//                for (k=0; k < BS; k++) {
//#pragma omp simd 
//                    for (j = 0; j < BS; j++) {
//                        Sum[j] += temp_left[BS*i + k] * temp_top[BS*k + j];
//                    }
//                }
//#pragma omp simd 
//                for (j = 0; j < BS; j++) {
//                    BB((i+i_global),(j+j_global)) -= Sum[j];
//                    Sum[j] = 0;
//                }
//            }
           mm_s = clk::now();
  printf("===============for this iter of mm: chunk_idx: %d, BS: %d\n", chunk_idx, BS);
           gptpu_mm(temp_left, temp_top, Sum, BS, BS, BS, false);
           mm_e = clk::now();
           mm_cnt++;
           mm_us += std::chrono::duration_cast<std::chrono::nanoseconds>(mm_e - mm_s).count()/1000.0;
           for(i = 0 ; i < BS ; i++){
              for(j = 0 ; j < BS; j++){
                BB((i+i_global),(j+j_global)) -= Sum[i*BS+j];
              }
           }
        }
        blk_e = clk::now();
        blk_us += std::chrono::duration_cast<std::chrono::nanoseconds>(blk_e - blk_s).count()/1000.0;
    }
    dia_s = clk::now();    
    lud_diagonal_omp/*gptpu*/(a, size, offset, BS);
    dia_e = clk::now();    
    dia_us += std::chrono::duration_cast<std::chrono::nanoseconds>(dia_e - dia_s).count()/1000.0;
#ifdef OMP_OFFLOAD
}
#endif
  printf("dia  time: %12.3f (us) \n", dia_us);
  printf("top  time: %12.3f (us) \n", top_us);
  printf("left time: %12.3f (us) \n", left_us);
  printf("blk  time: %12.3f (us) \n", blk_us);
  printf("crop time: %12.3f (us) | cnt = %d\n", crop_us, crop_cnt);
  printf("mm   time: %12.3f (us) | cnt = %d\n", mm_us, mm_cnt);
}
