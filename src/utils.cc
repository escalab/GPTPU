#include <float.h>
#include <sstream>
#include <chrono>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <limits.h>
#include <cassert>
#include <fcntl.h>
#include <cstdlib>
#include <iomanip>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <glob.h>
#include "gptpu.h"
#include "utils.h"
#include "fifo.h"
#include <algorithm>
#include <math.h>
#include <cblas.h>
#include <fstream>
#ifndef UTILS_H
#define UTILS_H

extern int MM_BLK_ROW;
extern int MM_BLK_COL;
extern int VERBOSE;
extern std::string data_dir;
extern std::string out_path;
extern int** partial_c;
extern int ITER;
extern int ramdisk;
extern bool DEV_OPENED;
extern int dev_cnt;
extern unsigned long long int model_ns;
extern unsigned long long int itpr_ns;
extern unsigned long long int input_ns;
extern unsigned long long int mem_ns;
extern unsigned long long int run_ns;
extern unsigned long long int pop_ns;
extern unsigned long long int check_ns;
extern unsigned long long int itpr_init_ns;

pthread_mutex_t run_mtx;
extern pthread_mutex_t pmtx;
extern pthread_cond_t in_CV;
extern pthread_cond_t out_CV;
extern pthread_cond_t end_CV;
extern int qin;
extern int qout;
extern int queue_cnt;
extern bool done_enqueue;
//extern struct Task_OP_queue Q[];
extern OP_node Q[];
extern bool stop_all;
extern int ack_cnt;
extern unsigned long long int itpr_ns;
extern std::string data_type;
extern int min_op_or_dev;
extern int exact_mode;

// ===== use fifo =====
extern struct fifo *SPMC_fifo;
//=====================
unsigned int *tpu_id;

typedef std::chrono::time_point<std::chrono::high_resolution_clock> timing;
typedef std::chrono::high_resolution_clock clk;

int get_chunk_size(){
  return dense_get_chunk_size();
}

void set_chunk_size(int size){
  dense_set_chunk_size(size);
}

struct tid_args{
  unsigned int tid;
};

#define STOP_ALL_HANDLE(){\
  stop_all_handle(ack_cnt, *tid, invoke_cnt, dev_mem_ns, dev_run_ns, dev_pop_ns, dev_cnt);\
  break; \
}

void inline stop_all_handle(int& ack_cnt, int tid, int invoke_cnt, long long int dev_mem_ns, long long int dev_run_ns, long long int dev_pop_ns, int dev_cnt){
  ack_cnt++;
  std::cout << "consumer " << tid << " is terminated, ack_cnt: " << ack_cnt << std::endl;
  printf("tid: %d, dev_mem: %f (us), dev_run: %f (us), dev_pop: %f (us), invoke_cnt: %d\n", tid, (double)dev_mem_ns/1000.0, (double)dev_run_ns/1000.0, (double)dev_pop_ns/1000.0, invoke_cnt);
  if(ack_cnt >= dev_cnt){ // the last consumer is informing producer
    pthread_cond_signal(&end_CV);
  }
  pthread_mutex_unlock(&pmtx);
}

int tmp_cnt = 0;
// one consumer for one device, the concept of runtime
void *consumer(void *_i){
  unsigned int* tid = ((unsigned int *)_i);//args->tid;

  OP_node op_node;
  int invoke_cnt = 0;

  int A, B, C, in, idx, xi, yi, model_id, i, j, k, w_chunk_idx, ROW_BLK_CNT, INNER_BLK_CNT, COL_BLK_CNT, blk_A, blk_B, blk_C;
  int *a; int *b; int *c;
  int** partial_c; int** a_feed;
  bool b_major, output_no_chunk;
  std::string op;
  float SCALE;
  unsigned long long int out_size, b_offset, c_offset, offset;
  long long int dev_mem_ns = 0, dev_run_ns = 0, dev_pop_ns = 0;
  long long int mem_ns, run_ns, pop_ns;
  int output_size = 0;
  int chunk_num = CHAR_BIT/get_chunk_size();

  timing deq_s, deq_e;
  double deq_us = 0;
  
  struct OP_node *curr_node;

  while(true){
    if((curr_node = (struct OP_node*)fifo_pop(SPMC_fifo)) != NULL){
      chunk_num = CHAR_BIT/get_chunk_size(); // TODO: remove this by knowing the value on runtime
      if(!curr_node->op.compare("add_model")){
        out_size = curr_node->A * curr_node->B;
        populate_element_wise_input_chunking(curr_node->a+curr_node->offset, curr_node->b+curr_node->offset, out_size, curr_node->w_chunk_idx, curr_node->w_chunk_idx, add_chunk_size, *tid, curr_node->op, mem_ns);
        run_ns = invoke_model(*tid, ITER);
        pthread_mutex_lock(&pmtx);
        pop_ns = populate_element_wise_output_chunking(curr_node->c+curr_node->offset, out_size, *tid/*model_id*/, 0, curr_node->w_chunk_idx, 1/*SCALE*/);
        pthread_mutex_unlock(&pmtx);
        dev_mem_ns += mem_ns;
        dev_run_ns += run_ns;
        dev_pop_ns += pop_ns;
      }else if(!curr_node->op.compare("sub_model")){
        out_size = curr_node->A * curr_node->B;
        populate_element_wise_input_chunking(curr_node->a+curr_node->offset, curr_node->b+curr_node->offset, out_size, curr_node->w_chunk_idx, curr_node->w_chunk_idx, add_chunk_size, *tid, curr_node->op, mem_ns);
        run_ns = invoke_model(*tid, ITER);
        pthread_mutex_lock(&pmtx);
        pop_ns = populate_element_wise_output_chunking(curr_node->c+curr_node->offset, out_size, *tid/*model_id*/, 0, curr_node->w_chunk_idx, 1/*SCALE*/);
        pthread_mutex_unlock(&pmtx);
        dev_mem_ns += mem_ns;
        dev_run_ns += run_ns;
        dev_pop_ns += pop_ns;
      }else if(!curr_node->op.compare("mul_model")){
        out_size = curr_node->A * curr_node->B;
        populate_element_wise_input_chunking(curr_node->a+curr_node->offset, curr_node->b+curr_node->offset, out_size, curr_node->xi, curr_node->yi, (exact_mode==1)?mul_chunk_size:CHAR_BIT, *tid, curr_node->op, mem_ns);
        run_ns = invoke_model(*tid, ITER);
        pthread_mutex_lock(&pmtx);
        pop_ns = populate_element_wise_output_chunking(curr_node->c+curr_node->offset, out_size, *tid/*model_id*/, curr_node->xi, curr_node->yi, 1/*SCALE*/);
        dev_mem_ns += mem_ns;
        dev_run_ns += run_ns;
        dev_pop_ns += pop_ns;
        pthread_mutex_unlock(&pmtx);
      }else if(!curr_node->op.compare("mv_model")){
        itpr_ns += build_interpreter(*tid, curr_node->model_id);
        if(curr_node->b_major == true){
          for(int k = 0 ; k < curr_node->B ; k++){
            for(int in_chunk_idx = 0 ; in_chunk_idx < chunk_num ; in_chunk_idx++){
              b_offset = k * curr_node->in + curr_node->j * MM_BLK_COL;
              c_offset = k * curr_node->in + curr_node->i * MM_BLK_ROW;
              run_modelV2(curr_node->b+b_offset, MM_BLK_COL, ITER, output_size, curr_node->partial_c[j]+c_offset, *tid, curr_node->model_id, data_type, curr_node->w_chunk_idx, in_chunk_idx, VERBOSE, mem_ns, run_ns, pop_ns);
              dev_mem_ns += mem_ns;
              dev_run_ns += run_ns;
              dev_pop_ns += pop_ns;
            }
          }
        }else{
          for(int k = 0 ; k < curr_node->COL_BLK_CNT ; k++){
            for(int l = 0 ; l < MM_BLK_ROW ; l++){
              for(int in_chunk_idx = 0 ; in_chunk_idx < chunk_num ; in_chunk_idx++){
                b_offset = (k*MM_BLK_ROW+l)*curr_node->in+curr_node->j*MM_BLK_COL;
                c_offset = k*(MM_BLK_ROW*MM_BLK_COL*curr_node->COL_BLK_CNT)+curr_node->i*(MM_BLK_ROW)+l*curr_node->A;
                run_modelV2(curr_node->b+b_offset, MM_BLK_COL, ITER, output_size, curr_node->partial_c[j]+c_offset, *tid, curr_node->model_id, data_type, curr_node->w_chunk_idx, in_chunk_idx, VERBOSE, mem_ns, run_ns, pop_ns);
                dev_mem_ns += mem_ns;
                dev_run_ns += run_ns;
                dev_pop_ns += pop_ns;
              }
            }
          }
        }
      }else if(!curr_node->op.compare("mm_model")){
      //std::cout << "model_id: " << curr_node->model_id << ", w_chunk_idx: " << curr_node->w_chunk_idx << ", chunk_num: " << chunk_num << ", start_chunk: " << curr_node->start_chunk << std::endl;
        build_interpreter(*tid, curr_node->model_id); 
        if(curr_node->mm256 == false){ // the default mm2conv design
          for(int i = 0 ; i < curr_node->ROW_BLK_CNT ; i++){
            for(int in_chunk_idx = curr_node->start_chunk ; in_chunk_idx < chunk_num ; in_chunk_idx++){
              //dev_mem_ns += populate_input_chunking(curr_node->a_feed[(i*curr_node->INNER_BLK_CNT*curr_node->COL_BLK_CNT) + (curr_node->j*curr_node->COL_BLK_CNT) + curr_node->k], (curr_node->blk_A)*(curr_node->blk_C), curr_node->model_id, in_chunk_idx, data_type);
              dev_mem_ns += populate_input_chunking(curr_node->a_feed[(i*curr_node->INNER_BLK_CNT) + curr_node->j], (curr_node->blk_A)*(curr_node->blk_B), curr_node->model_id, in_chunk_idx, data_type);
              dev_run_ns += invoke_model(curr_node->model_id, ITER);
              pthread_mutex_lock(&pmtx);
              dev_pop_ns += populate_output_chunking(curr_node->partial_c[curr_node->j], curr_node->A, curr_node->C, curr_node->blk_A, curr_node->blk_C, i, curr_node->k, curr_node->model_id, in_chunk_idx, curr_node->w_chunk_idx, curr_node->SCALE);
              pthread_mutex_unlock(&pmtx);
            }
          }
        }else{ // the mm 256 for exact mode design
          for(int i = 0 ; i < curr_node->ROW_BLK_CNT ; i++){
            //dev_mem_ns += populate_input_exact(curr_node->a_feed[(i*curr_node->INNER_BLK_CNT*curr_node->COL_BLK_CNT) + (curr_node->j*curr_node->COL_BLK_CNT) + curr_node->k], curr_node->blk_A, curr_node->blk_C, curr_node->chunk_num, curr_node->model_id, data_type);
            dev_mem_ns += populate_input_exact(curr_node->a_feed[(i*curr_node->INNER_BLK_CNT) + curr_node->j], curr_node->blk_A, curr_node->blk_B, curr_node->chunk_num, curr_node->model_id, data_type);
            dev_run_ns += invoke_model(curr_node->model_id, ITER);
//            printf("cnt: %4d, accu us: %f\n", ++tmp_cnt, dev_run_ns/1000.0);
            pthread_mutex_lock(&pmtx);
            dev_pop_ns += populate_output_exact(curr_node->partial_c[i*(curr_node->COL_BLK_CNT)+curr_node->k], curr_node->A, curr_node->C, curr_node->blk_A, curr_node->blk_C, i, curr_node->k, curr_node->model_id, curr_node->chunk_num, curr_node->SCALE);
            pthread_mutex_unlock(&pmtx);
          }
        }
      }else if(!curr_node->op.compare("stall")){
        std::cout << "tid: " << *tid << ", op is stall" << std::endl;
      }else{
        std::cout << "consumer " << *tid << " receives unknown op name: " << curr_node->op << ", call pythread_exit()" << std::endl;
        pthread_exit(NULL);
      }
      // ===== end OP handling =====
      pthread_mutex_lock(&pmtx);
      if(stop_all == true){
        STOP_ALL_HANDLE();
      }else if(fifo_empty(SPMC_fifo) && done_enqueue == true){
        stop_all = true; // start to exit consumers
        STOP_ALL_HANDLE();
      }
      pthread_mutex_unlock(&pmtx);
    } // end pop 
  }
  pthread_exit(NULL);
// temp ignore below

//  while(true){
//    pthread_mutex_lock(&pmtx);
//    while(qin == qout){ // condition for empty
//      if(stop_all == true){
//          STOP_ALL_HANDLE();
//      }
//      pthread_cond_wait(&in_CV, &pmtx);
//    }
//    if(stop_all == true){  // check flag "stop_all" before dequeue, otherwise the OP wouldn't be valid
//      STOP_ALL_HANDLE();
//    }
//    queue_cnt--;
//// ===== dequeue =====
//    op = Q[qout].op;
//    if(!op.compare("mul_model")){
//      A           = Q[qout].A;
//      B           = Q[qout].B;
//      a           = Q[qout].a;
//      b           = Q[qout].b;
//      c           = Q[qout].c;
//      i           = Q[qout].i;
//      j           = Q[qout].j;
//      ROW_BLK_CNT = Q[qout].ROW_BLK_CNT;
//      COL_BLK_CNT = Q[qout].COL_BLK_CNT;
//      offset      = Q[qout].offset;
//      output_no_chunk = Q[qout].output_no_chunk;
//      idx         = Q[qout].idx;
//      xi          = Q[qout].xi;
//      yi          = Q[qout].yi;
//      //DEQUEUE_OP();
//      //std::cout << "A: " << A << ", B: " << B << ", i: " << i << ", j: " << j << ", ROW_BLK_CNT: " << ROW_BLK_CNT << ", COL_BLK_CNT: " << COL_BLK_CNT << ", offset: " << offset << ", xi: " << xi << ", yi: " << yi << std::endl;
//      out_size    = A*B;
//    }else if(!op.compare("add_model")){
//      A           = Q[qout].A;
//      B           = Q[qout].B;
//      a           = Q[qout].a;
//      b           = Q[qout].b;
//      c           = Q[qout].c;
//      i           = Q[qout].i;
//      j           = Q[qout].j;
//      ROW_BLK_CNT = Q[qout].ROW_BLK_CNT;
//      COL_BLK_CNT = Q[qout].COL_BLK_CNT;
//      offset      = Q[qout].offset;
//      w_chunk_idx = Q[qout].w_chunk_idx;
//      out_size    = A*B;
//    }else if(!op.compare("mv_model")){
//      model_id      = Q[qout].model_id;
//      b_major       = Q[qout].b_major;
//      A             = Q[qout].A;
//      B             = Q[qout].B;
//      in            = Q[qout].in;
//      ROW_BLK_CNT   = Q[qout].ROW_BLK_CNT;
//      INNER_BLK_CNT = Q[qout].INNER_BLK_CNT;
//      COL_BLK_CNT   = Q[qout].COL_BLK_CNT;
//      b             = Q[qout].b;
//      partial_c     = Q[qout].partial_c;
//      i             = Q[qout].i;
//      j             = Q[qout].j;
//      w_chunk_idx   = Q[qout].w_chunk_idx;
//    }else if(!op.compare("mm_model")){
//      model_id      = Q[qout].model_id;
//      a_feed        = Q[qout].a_feed;
//      partial_c     = Q[qout].partial_c;
//      A             = Q[qout].A;
//      B             = Q[qout].B;
//      C             = Q[qout].C;
//      j             = Q[qout].j;
//      k             = Q[qout].k;
//      w_chunk_idx   = Q[qout].w_chunk_idx;
//      blk_A         = Q[qout].blk_A;
//      blk_B         = Q[qout].blk_B;
//      blk_C         = Q[qout].blk_C;
//      ROW_BLK_CNT   = Q[qout].ROW_BLK_CNT;
//      INNER_BLK_CNT = Q[qout].INNER_BLK_CNT;
//      COL_BLK_CNT   = Q[qout].COL_BLK_CNT;
//      SCALE         = Q[qout].SCALE; 
//    /*}else */if(!op_node.op.compare("stall")){
//      std::cout << "consumer " << tid << " is stalling." << std::endl;
//    }
//    }else{
//      std::cout << "consumer " << tid << "receives unknown op name: " << op << ", call pythread_exit()" << std::endl;
//      pthread_exit(NULL);
//    }
//// ===== done dequeue =====
//    qout = (qout + 1) % queue_size;
////    std::cout << __func__ << ", tid: " << tid << ", op: " << op << ", qin: " << qin << ", qout: " << qout << std::endl;
//    pthread_mutex_unlock(&pmtx);
//// ===== do acutal work =====
//    if(!op.compare("mul_model")){
//      if(output_no_chunk == false){
//        populate_element_wise_input_chunking(a+offset, b+offset, out_size, xi, yi, (exact_mode==1)?mul_chunk_size:CHAR_BIT, *tid, op, mem_ns);
//        run_ns = invoke_model(*tid, ITER);
//        pthread_mutex_lock(&pmtx);
//        pop_ns = populate_element_wise_output_chunking(c+offset, out_size, *tid/*model_id*/, xi, yi, 1/*SCALE*/);
//        pthread_mutex_unlock(&pmtx);
//      }else{
//        invoke_cnt ++;
//        populate_mm2mul_input_chunking(a, A, B, offset, b+offset, idx, i, j, ROW_BLK_CNT, COL_BLK_CNT, out_size, xi, yi, (exact_mode==1)?mul_chunk_size:CHAR_BIT, *tid, op, mem_ns);
//        run_ns = invoke_model(*tid, ITER);
//        //pop_ns = populate_element_wise_output(c+offset, out_size, tid/*model_id*/, 1/*SCALE*/);
//        pthread_mutex_lock(&pmtx);
//        pop_ns = populate_element_wise_output_chunking(c+offset, out_size, *tid/*model_id*/, xi, yi, 1/*SCALE*/);
//        pthread_mutex_unlock(&pmtx);
//      }
//      dev_mem_ns += mem_ns;
//      dev_run_ns += run_ns;
//      dev_pop_ns += pop_ns;
//    }else if(!op.compare("add_model")){
//      populate_element_wise_input_chunking(a+offset, b+offset, out_size, w_chunk_idx, w_chunk_idx, add_chunk_size, *tid, op, mem_ns);
//      run_ns = invoke_model(*tid, ITER);
//      pthread_mutex_lock(&pmtx);
//      pop_ns = populate_element_wise_output_chunking(c+offset, out_size, *tid/*model_id*/, 0, w_chunk_idx, 1/*SCALE*/);
//      pthread_mutex_unlock(&pmtx);
//      dev_mem_ns += mem_ns;
//      dev_run_ns += run_ns;
//      dev_pop_ns += pop_ns;
//    }else if(!op.compare("mv_model")){
//      itpr_ns += build_interpreter(*tid, model_id);
//      if(b_major == true){
//        for(int k = 0 ; k < B ; k++){
//          for(int in_chunk_idx = 0 ; in_chunk_idx < chunk_num ; in_chunk_idx++){
//            b_offset = k * in + j * MM_BLK_COL;
//            c_offset = k * in + i * MM_BLK_ROW;
//            run_modelV2(b+b_offset, MM_BLK_COL, ITER, output_size, partial_c[j]+c_offset, *tid, model_id, data_type, w_chunk_idx, in_chunk_idx, VERBOSE, mem_ns, run_ns, pop_ns);
//            dev_mem_ns += mem_ns;
//            dev_run_ns += run_ns;
//            dev_pop_ns += pop_ns;
//          }
//        }
//      }else{
//        for(int k = 0 ; k < COL_BLK_CNT ; k++){
//          for(int l = 0 ; l < MM_BLK_ROW ; l++){
//            for(int in_chunk_idx = 0 ; in_chunk_idx < chunk_num ; in_chunk_idx++){
//              b_offset = (k*MM_BLK_ROW+l)*in+j*MM_BLK_COL;
//              c_offset = k*(MM_BLK_ROW*MM_BLK_COL*COL_BLK_CNT)+i*(MM_BLK_ROW)+l*A;
//              run_modelV2(b+b_offset, MM_BLK_COL, ITER, output_size, partial_c[j]+c_offset, *tid, model_id, data_type, w_chunk_idx, in_chunk_idx, VERBOSE, mem_ns, run_ns, pop_ns);
//              dev_mem_ns += mem_ns;
//              dev_run_ns += run_ns;
//              dev_pop_ns += pop_ns;
//            }
//          }
//        }
//      }
//    }else if(!op.compare("mm_model")){
//      build_interpreter(*tid, model_id); 
//      for(int i = 0 ; i < ROW_BLK_CNT ; i++){
//        for(int in_chunk_idx = 0 ; in_chunk_idx < chunk_num ; in_chunk_idx++){
//          dev_mem_ns += populate_input_chunking(a_feed[i*INNER_BLK_CNT*COL_BLK_CNT + j*COL_BLK_CNT + k], blk_A*blk_C, model_id, in_chunk_idx, data_type);
//          dev_run_ns += invoke_model(model_id, ITER);
//          pthread_mutex_lock(&pmtx);
//          dev_pop_ns += populate_output_chunking(partial_c[j], A, C, blk_A, blk_C, i, k, model_id, in_chunk_idx, w_chunk_idx, SCALE);
//          pthread_mutex_unlock(&pmtx);
//        }
//      }         
//    }else if(!op.compare("stall")){
//      std::cout << "tid: " << tid << ", op is stall" << std::endl;
//    }
//// ===== done actual work =====
//// ===== check for exit =====
//    pthread_mutex_lock(&pmtx);
//    if(stop_all == true){
//      STOP_ALL_HANDLE();
//    }else if(queue_cnt == 0 && done_enqueue == true){
//      stop_all = true; // start to exit consumers
//      STOP_ALL_HANDLE();
//    }
//    pthread_mutex_unlock(&pmtx);
//// ===== end checking for exit =====
//    pthread_cond_signal(&out_CV); 
//  } // end while true
//  pthread_exit(NULL);
}

void close_devices(){
  for(int i = 0 ; i < dev_cnt ; i++){
    close_device(i, VERBOSE);
  }
}

void open_devices(int opening_order, int wanted_dev_cnt){
  pthread_cond_init(&in_CV, NULL);
  pthread_cond_init(&out_CV, NULL);
  pthread_cond_init(&end_CV, NULL);
  int local_dev_cnt;
  long long int list_ns = 0;
  long long int open_ns = 0;
  set_dev_cnt(wanted_dev_cnt);
  //data_dir = (ramdisk == 1)?("/mnt/ramdisk/"):("./../data/");
  data_dir = (ramdisk == 1)?("/mnt/ramdisk/"):("./");
  if(DEV_OPENED == false){ // once for all during process lifetime
    DEV_OPENED = true;
    list_ns = ListTheDevices(VERBOSE, local_dev_cnt); // once for all
   // set_chunk_size(CHUNK_SIZE);
    if(local_dev_cnt <= 0){
      std::cout << "no any valid edgetpu device found, program exit" << std::endl;
      exit(0);
    }else if(dev_cnt > local_dev_cnt){
      std::cout << "required device count "+std::to_string(dev_cnt)+" is smaller than available device count "+std::to_string(local_dev_cnt) << std::endl;
      exit(0);
    }
// ===== use fifo =====
    SPMC_fifo = fifo_new(queue_size);
    if(! SPMC_fifo){
      std::cout << "fifo_new invalid, exit." << std::endl;
      exit(0);
    }
//=====================
    stop_all = false;
    ack_cnt = 0;
    pthread_t tid[dev_cnt];
    tpu_id = (unsigned int*) malloc(dev_cnt*sizeof(unsigned int));
    srand(time(NULL));
    int offset = (opening_order == 0)?0:((int)(rand()%local_dev_cnt));
    for(int i = 0 ; i < dev_cnt ; i++){
      std::cout << "opening " << std::to_string((i+offset)%local_dev_cnt) << "th device (" << std::to_string(i+1) << " out of " << std::to_string(local_dev_cnt) << ")" << std::endl;
      open_ns += open_device((i+offset)%local_dev_cnt, VERBOSE); //once for all
      tpu_id[i] = (i+offset)%local_dev_cnt;
      pthread_create(&tid[i], NULL, consumer, (void *)&tpu_id[i]);
    }
    std::cout << "device opening elasped: " << open_ns/1000000.0 << "(ms), listed time: " << list_ns/1000000.0 << " (ms)." << std::endl;
  }
  model_ns = itpr_init_ns = itpr_ns = input_ns = mem_ns = run_ns = pop_ns = check_ns = 0;
}

bool crop_check(int A, int B, int blk_row, int blk_col, int start_i, int start_j){
  if(blk_row > A || blk_col > B || start_i > A || start_j > B || (start_i+blk_row) > A || (start_j+blk_col) > B || start_i < 0 || start_j < 0){
    std::cout << "crop parameters checking fails." <<std::endl;
    std::cout << "A: " << A << ", B: " << B << ", blk_row: " << blk_row << ", blk_col: " << blk_col << ", start_i: " << start_i  << ", start_j: " << start_j << std::endl;
    return false;
  }else if(A <= 0 || B <= 0 || blk_row <= 0 || blk_col <= 0){
    return false; // either no input or no output, skip any kernel operation, just return
  }else{
    return true;
  }
}

int print_buffer(char* buf, unsigned long long int size){
  for(unsigned long long int i = 0 ; i < size ; i++){
    std::cout << (unsigned)buf[i];
  }
  std::cout << std::endl;
  return 0;
}

float set_data_array(int* b, char* data_array, unsigned long long int size){
 // TODO: mean_value is assume to be 1 for now, and location not clear
  // assign data
  timing convert_start = clk::now();
  for(unsigned long long int i = 0 ; i < size ; i++){
    data_array[i] = b[i];
  }
  timing convert_end   = clk::now();
//  if(VERBOSE >= 0){
//    std::cout << "+===== set_data_array timing breakdown =====" << std::endl;
//    std::cout << "| data   : " << std::chrono::duration_cast<std::chrono::nanoseconds>(convert_end - convert_start).count()/1000.0 << "\t(us)." << std::endl;
//    std::cout << "+===========================================" << std::endl;
//  }
  float scaling = 1;
  return scaling;
}

void set_block_array(int* a, bool b_major, char* data_array, int row_idx, int col_idx, int A, int B, int ROW_BLK_REM, int COL_BLK_REM, int chunk_idx/*w*/){
 // check idxs are within bound
  int ROW_BLK_CNT   = (A / MM_BLK_ROW) + ((A % MM_BLK_ROW != 0)?1:0);
  int INNER_BLK_CNT = (B / MM_BLK_COL) + ((B % MM_BLK_COL != 0)?1:0);
  int a_offset, d_offset, starting;
  int chunk_size = get_chunk_size();
  int chunk_mask = ~(0xffffffff << chunk_size);
  if(!(row_idx < ROW_BLK_CNT && col_idx < INNER_BLK_CNT)){
    std::cout << "set_block_array: invalid indexing, row_idx: " << row_idx << ", col_idx: " << col_idx << " are out of [" << ROW_BLK_CNT << "x" << INNER_BLK_CNT << "]" << std::endl;
    exit(0);
  }
  starting = row_idx * (MM_BLK_ROW * B) + (col_idx * MM_BLK_COL);
  timing convert_start = clk::now();
  for(int i = 0 ; i < MM_BLK_ROW ; i++){
    for(int j = 0 ; j < MM_BLK_COL ; j++){
      a_offset = starting + ((b_major == true)?( (i*B)+j ):( (j*B)+i) );
      d_offset = i*MM_BLK_COL + j;
      data_array[d_offset] = (COL_BLK_REM != 0 && (col_idx+1 == INNER_BLK_CNT) && j >= COL_BLK_REM)
                             ?(0)
                             :((a[a_offset] >> (chunk_idx * chunk_size)) & chunk_mask);
//      printf("data_array[%d]=%d\n", d_offset, data_array[d_offset]);
    }
  }
  timing convert_end   = clk::now();
  if(VERBOSE >= 0){
    std::cout << "+===== set_block_array timing breakdown =====" << std::endl;
    std::cout << "| data   : " << std::chrono::duration_cast<std::chrono::nanoseconds>(convert_end - convert_start).count()/1000.0 << "\t(us)." << std::endl;
    std::cout << "+============================================" << std::endl;
  }
}

int save_input(const std::string& input_path, int* a, int A){
  int fd = open(input_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, mode);
  char* map;
  struct stat st;
  fstat(fd, &st);
  map = static_cast<char*>(mmap(NULL, A*sizeof(char), PROT_WRITE, MAP_SHARED, fd, 0));
  assert(map != MAP_FAILED);
  if(ftruncate(fd, A*sizeof(char)) != 0){
    std::cout << "input file ftruncate fail." << std::endl;
  }
  char* buf = (char*) malloc(A*sizeof(char));
  for(int i = 0 ; i < A ; i++){
     assert(a[i] >= 0 || a[i] < UCHAR_MAX);
     buf[i] = (char)a[i];
  }
  memcpy(map, buf, A);
  munmap(map, A*sizeof(char));
  close(fd);
  free(buf);
  return 0;
}

int read_output(const std::string& output_path, int* c, int N){
  int fd = open(output_path.c_str(), O_RDONLY);
  char *map;
  struct stat st;
  fstat(fd, &st);
  map = static_cast<char*>(mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0));
  assert(map != MAP_FAILED);
  memcpy(reinterpret_cast<char*>(&c[0]), map, st.st_size);
  munmap(map, st.st_size);
  close(fd);
  if(0){
    std::cout << "read output: ";
    for(int i = 0 ; i < N ; i++){
      std::cout << (unsigned)c[i] << " ";
    }
    std::cout << std::endl;
  }
  return 0;
}

void conv_save_weight(int* f, const std::string& weight_file_name, int A, int B){
  int fd = open(weight_file_name.c_str(), O_RDWR | O_CREAT | O_TRUNC, mode);
  char* map;
  struct stat st;
  fstat(fd, &st);
  map = static_cast<char*>(mmap(NULL, A*B*sizeof(int32_t), PROT_WRITE, MAP_SHARED, fd, 0));
  assert(map != MAP_FAILED);
  if(ftruncate(fd, A*B*sizeof(int32_t)) != 0){
    std::cout << __func__ << "input file ftruncate fail." << std::endl;
    exit(0);
  }
  // data mapping
  memcpy(map, reinterpret_cast<char*>(f), A*B*sizeof(int32_t));
  // end mapping
  munmap(map, A*B*sizeof(int32_t));
  close(fd);
}

void save_weight_uint8(uint8_t* f, const std::string& weight_file_name, int size){
  int fd = open(weight_file_name.c_str(), O_RDWR | O_CREAT | O_TRUNC, mode);
  char* map;
  struct stat st;
  fstat(fd, &st);
  map = static_cast<char*>(mmap(NULL, size*sizeof(uint8_t), PROT_WRITE, MAP_SHARED, fd, 0));
  assert(map != MAP_FAILED);
  if(ftruncate(fd, size*sizeof(uint8_t)) != 0){
    std::cout << __func__ << "file ftruncate fail." << std::endl;
    exit(0);
  }
  // data mapping
  memcpy(map, reinterpret_cast<char*>(f), size*sizeof(uint8_t));
  // end mapping
  munmap(map, size*sizeof(uint8_t));
  close(fd);
}

void partition_conv_input(int* a, int* a_blk, int A, int B, int blk_A, int blk_B, int start_i, int start_j){
  for(int i = 0; i < blk_A ; i++){
    for(int j = 0 ; j < blk_B ; j++){
      //a_blk[i*blk_B+j] = a[((j+(start_j*blk_B))*B)+(i+(start_i*blk_A))];
      a_blk[i*blk_B+j] = a[((i+(start_i*blk_A))*B)+(j+(start_j*blk_B))];
    }
  }
  return;
}

void pad_input(int* a, int* a_blk, int* a_pad, int GA, int GB, int A, int B, int A_pad, int B_pad, int padding, int blk_i ,int blk_j, int ROW_BLK_CNT, int COL_BLK_CNT){
// padding: 1 for same(zero) padding, 2 for replication padding
// TODO: wider padding size > 1
/*    B_pad
+-------------+
|      B      |
|   +-----+   |
|   |     | A | A_pad
|   |     |   |
|   +-----+   |
|             |
+-------------+
*/
  int pad_h = (A_pad - A)/2;
  int pad_w = (B_pad - B)/2;

// a_blk = a[blk_i*A*GB+blk_j*B]
  int offset = blk_i*A*GB+blk_j*B; // the offset where a_blk, as a pointer,  points to in a
// shifting inner data matrix
  for(int i = B-1 ; i >= 0 ; i--){
    memcpy(a_pad+((i+pad_w)*A_pad)+pad_h, a_blk+(i*A), A*sizeof(int));
  }
  // actual same adding
  for(int i = 0 ; i < A_pad ; i++){
    for(int j = 0 ; j < B_pad ; j++){
      if(i < pad_h){                  // upper stride
        if(blk_i > 0 && j >= pad_w && j < (B_pad - pad_w)){ a_pad[i*B_pad+j] = a[offset - GB*pad_h + (j - pad_w)]; }
        else{                                              a_pad[i*B_pad+j] = a_pad[pad_h*B_pad+j]; }
      }else if(i >= (A_pad - pad_h)){ // lower stride
        if(blk_i < (ROW_BLK_CNT-1) && j >= pad_w && j < (B_pad - pad_w)){ a_pad[i*B_pad+j] = a[offset + (A+i-(A_pad-pad_h))*GB + (j - pad_w)]; }
        else{                                                            a_pad[i*B_pad+j] = a_pad[(A_pad - pad_h - 1)*B_pad+j]; }
      }else if(j < pad_w){            // left  stride
        if(blk_j > 0 && i >= pad_h && i < (A_pad - pad_h)){ a_pad[i*B_pad+j] = a[offset - pad_w + (i - pad_h)*GB]; }
        else{                                              a_pad[i*B_pad+j] = a_pad[i*B_pad+pad_w]; }
      }else if(j >= (B_pad - pad_w)){ // right stride
        if(blk_j < (COL_BLK_CNT-1) && i >= pad_h && i < (A_pad - pad_h)){ a_pad[i*B_pad+j] = a[offset + B + (i-pad_h)*GB];}
        else{                                                            a_pad[i*B_pad+j] = a_pad[i*B_pad+(B_pad - pad_w - 1)]; }
      }else{ /* do nothing*/ }
    }
  }
}

void save_partial(int A, int B, int INNER_BLK_CNT){
// save partial_c for checking
  int* fds = (int*) malloc(INNER_BLK_CNT*sizeof(int));
  char* dst[INNER_BLK_CNT];
  std::string out_path;
  for(int i = 0 ; i < INNER_BLK_CNT ; i++){
    out_path = data_dir+"mm_partial_["+std::to_string(i)+"].txt";
    fds[i] = open(out_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0664);
    if(fds[i] < 0){ std::cout << "partial saving file open fail." << out_path << std::endl; }
    dst[i] = static_cast<char*>(mmap(NULL, A*B*sizeof(int), PROT_WRITE, MAP_SHARED, fds[i], 0));
    assert(dst[i] != MAP_FAILED);
    if(ftruncate(fds[i], A*B*sizeof(int)) != 0){ std::cout << "ftruncate file fail." << out_path << std::endl; }
    memcpy(dst[i], partial_c[i], A*B*sizeof(int));
    munmap(dst[i], A*B*sizeof(int));
     //close(fds[i]);
  }
  free(fds);
}

int HPof2(int n){
// get highest power of 2 less than or equal to given number
  int ret = 0;
  for(int i=n ; i>=1 ; i--){
    if((i & (i-1)) == 0){
      ret = i; break;
    }
  }
  return ret;
}

void select_blk_shape(int A, int in, int B, int& MM_BLK_ROW, int& MM_BLK_COL, bool b_major){
  int dim = (b_major == true)?A:B;
  //TODO: the following is default rule, can be improved later
  // MM_BLK_ROW is the largest power of 2 which smaller than dim
  // MM_BLK_COl is the largest power of 2 which smaller than in
  MM_BLK_ROW = HPof2(dim);
  MM_BLK_COL = HPof2(in);
}

void data_mm256conv(int* in/*blk_A*blk_C*/, int *out/*blk_A*blk_C*chunk_num*/, int A, int B, int IN_W, int IN_H, int F_W, int F_H, int IN_C, int chunk_num){
  int IN_W_cnt     = IN_W / F_W ;    //1024  // block count in row direction
  int section_size = F_W * IN_H * IN_C; // 1024  //independent section size (multiple mm rows)
  int block_size   = F_H * IN_C; // 32  // length of continuous data in memory
//TODO:  parallelize it for optimization
  /*
A:    256
B:    256
IN_W: 8192
IN_H: 8
F_W:  8
F_H:  2
IN_C: 16
*/
  for(int IN_W_idx = 0 ; IN_W_idx < IN_W_cnt/*1024*/ ; IN_W_idx++){
    int sid = 0;
    int section_offset = IN_W_idx * section_size/* IN_W_idx * 1024*/;
    for(int i = 0 ; i< F_W/*8*/ ; i++){
      for(int j = 0 ; j < (IN_H/F_H)/*4*/ ; j++){
        sid = i*(IN_H/F_H)+j; // = 4*i+j     // serial number of a block (F_H*IN_C) within section (section_size)
        int i_len = sid / F_W;// = sid/8   // the ith block in row direction within a section
        int jump  = (IN_H/F_H) * (sid % (F_W)); // 4*(sid % 8)
        int in_offset = section_offset + sid * block_size/*block offset*/; // 
//          in_offset = (0,1,2,...,1023)*1024 + sid * 32
     //               = (0,1,2,...,1023*1024) + (0,1,2,...,31)*32
//                    = (0,32,64,...,992,1024,1056,...,1048544) (total len: 1048576)
      //  int chunk_idx = in_offset/(256*256);
         // mm2conv: memcpy(
//            &out[section_offset + (i_len + jump) * block_size/*block offset*/], 
//            &in[section_offset + sid * block_size/*block offset*/], 
              // in = IN_W_idx(0,1,2,...,1023) * 1024 + sid(0,1,2,...,31) * 32
//            block_size*sizeof(int));
//std::cout << "section_offset: " << section_offset << ", i_len: " << i_len << ", jump: " << jump << ", block_size: " << block_size << " | sid: " << sid << "| [" << section_offset + (i_len + jump) * block_size << "] and [" << (section_offset + sid * block_size)%(256*256) << "]" << std::endl;
        memcpy(&out[section_offset + (i_len + jump) * block_size],
               // out = IN_W_idx*1024 + (sid/8 + 4*(sid%8) ) * 32
               //     = (0,1,2,...,1023)*1024 + ((8[0],8[1],...,8[3]) + 4[0,4,8,...,28] ) *32
               &in[(section_offset + sid * block_size)%(256*256)],
               block_size*sizeof(int));
      }
    }
  }
//  for(int i = 0 ; i < 256*256 ; i++){
//    if(in[i] != 0){
//      int x = i/256;
//      int y = i%256;
//      std::cout << __func__ << ": in[" << i << "] is " << in[i] << ", x, y: " << x << ", " << y << std::endl;
//    }
//  }
//  for(int i = 0 ; i < 256*256*8 ; i++){
//    if(out[i] != 0){
//      int x = i%256;
//      int y = (i/(256))%256;
//      int z = (i/(256))/256;
//      std::cout << __func__ << ": out[" << i << "] is " << out[i] << ", x, y, z: " << x << ", " << y << ", " << z << std::endl;
//    }
//  }
}
void data_mm2conv(int* in, int *out, int A, int B, int IN_W, int IN_H, int F_W, int F_H, int IN_C){
  // shape checking
  if(!(A == ((IN_W/F_W)*(IN_H/F_H)) && B == (F_W*F_H*IN_C) && ((IN_W % F_W) == 0) && ((IN_H % F_H) == 0) )){
    std::cout << __func__ << ": sizes mismatch, A: " << A << ", B: " << B << ", IN_H: " << IN_W << ", IN_H: " << IN_H << ", F_W: " << F_W << ", F_H: " << F_H << ", IN_C: " << IN_C << std::endl;
    std::cout << "constructed A: " << ((IN_W/F_W)*(IN_H/F_H)) << ", B: " << F_W*F_H*IN_C << std::endl;
    exit(0);
  }
  int IN_W_cnt     = IN_W / F_W ; // block count in row direction
  int section_size = F_W * IN_H * IN_C; //independent section size (multiple mm rows)
  int block_size   = F_H * IN_C; // length of continuous data in memory
//TODO:  parallelize it for optimization
  for(int IN_W_idx = 0 ; IN_W_idx < IN_W_cnt ; IN_W_idx++){
    int sid = 0;
    int section_offset = IN_W_idx * section_size;
    for(int i = 0 ; i< F_W ; i++){
      for(int j = 0 ; j < (IN_H/F_H) ; j++){
        sid = i*(IN_H/F_H)+j;
        int i_len = sid / F_W;
        int jump  = (IN_H/F_H) * (sid % (F_W));
//        printf("IN_W_idx:%d, i:%d, j:%d, sid:%d, i_len:%d, jump:%d, F_W:%d, (IN_H/F_H):%d, section_offset:%d, block_size:%d, in_offset:%d, out_offset:%d\n", IN_W_idx, i, j, sid, i_len, jump, F_W, (IN_H/F_H), section_offset, block_size, section_offset+sid*block_size, section_offset+(i_len+jump)*block_size);
       memcpy(&out[section_offset + (i_len + jump) * block_size/*block offset*/], &in[section_offset + sid * block_size/*block offset*/], block_size*sizeof(int));
      }
    }
  }
}


void set_mm256conv_array(int* b, bool b_major, char* data_array, int B, int C, int i, int j, int ROW_BLK_CNT, int COL_BLK_CNT, int blk_r, int blk_c, int INN_BLK_REM, int COL_BLK_REM, int chunk_num, int exact_mode){
//  if(b_major == false /*row-major*/){
//    std::cout << __func__ << ": currently mm2conv only support col-major      B matrix, exit" << std::endl;
//    exit(0);
//  }
  int offset = 0;
  int chunk_size = get_chunk_size();
  if(chunk_size != 1){ std::cout << __func__ << ": warning: chunk_size is: " << chunk_size << " for mm256conv exact mode design." << std::endl;}
  int chunk_mask = ~(0xffffffff << chunk_size);
  for(int curr_j = 0 ; curr_j < blk_c ; curr_j++){
    for(int idx = 0 ; idx < blk_r ; idx++){
      for(int chunk_idx = 0 ; chunk_idx < chunk_num ; chunk_idx++){
        //offset = (b_major == false)?(idx*(blk_c*chunk_num)+(chunk_idx*blk_c)+curr_j):((chunk_idx*blk_c+curr_j)*blk_r+idx);
        offset = (b_major == false)?(idx*(blk_c*chunk_num)+(chunk_idx*blk_c)+curr_j):(curr_j*(blk_r*chunk_num)+chunk_idx*blk_r+idx);
        data_array[offset/*curr_j*blk_r+idx*/]  = ( ((i == ROW_BLK_CNT-1) && (idx >= INN_BLK_REM && INN_BLK_REM > 0)) ||
                                                    ((j == COL_BLK_CNT-1) && (curr_j >= COL_BLK_REM && COL_BLK_REM > 0))    )
                                                ?0
                                                :((b[B*(j*blk_c+curr_j)+i*blk_r+idx] >> (chunk_idx * chunk_size)) & chunk_mask)*((exact_mode==0)?0x01:0x01);
//        if(curr_j == 4 && idx == 0){
//          std::cout << __func__ << ": data_array[" << offset << "] = " << (unsigned)data_array[offset] << ", b[]: " << b[B*(j*blk_c+curr_j)+i*blk_r+idx] << ", w_chunk_idx = " << chunk_idx << ", exact_mode: " << exact_mode << ", b_major: " << b_major << ", blk_r: " << blk_r  << ", blk_c: " << blk_c << std::endl;
//        }
//        if(data_array[offset] != 0){
//          std::cout << __func__ << ": (i,j, chunk_idx) = (" << idx << ", " << curr_j << ", " << chunk_idx << ") , offset: " << offset << ", data: " << (unsigned)data_array[offset] << std::endl;
//        }
      }
    }
  }
}
void set_mm2conv_array(int* b, bool b_major, char* data_array, int B, int C, int i, int j, int ROW_BLK_CNT, int COL_BLK_CNT, int blk_r, int blk_c, int INN_BLK_REM, int COL_BLK_REM, int chunk_idx, int exact_mode){
//  if(b_major == false /*row-major*/){
//    std::cout << __func__ << ": currently mm2conv only support col-major      B matrix, exit" << std::endl;
//    exit(0);
//  }
  int offset = 0;
  int chunk_size = get_chunk_size();
  int chunk_mask = ~(0xffffffff << chunk_size);
  for(int curr_j = 0 ; curr_j < blk_c ; curr_j++){
    for(int idx = 0 ; idx < blk_r ; idx++){
      offset = (b_major == false)?(idx*blk_c+curr_j):(curr_j*blk_r+idx);
//std::cout << "curr_j,idx:(" << curr_j << ", " << idx << "), " << "offset: " << offset << ", b's offset: " << B*(j*blk_c+curr_j)+i*blk_r+idx << std::endl;
      data_array[offset/*curr_j*blk_r+idx*/]  = ( ((i == ROW_BLK_CNT-1) && (idx >= INN_BLK_REM && INN_BLK_REM > 0)) ||
                                                  ((j == COL_BLK_CNT-1) && (curr_j >= COL_BLK_REM && COL_BLK_REM > 0))    )
                                              ?0
                                              :((b[B*(j*blk_c+curr_j)+i*blk_r+idx] >> (chunk_idx * chunk_size)) & chunk_mask)*((exact_mode==0)?0x01:0xff);
      //if(offset == 0){
//      std::cout << __func__ << ": data_array[" << offset << "] = " << (unsigned)data_array[offset] << ", b: " << b[B*(j*blk_c+curr_j)+i*blk_r+idx] << ", chunk_idx = " << chunk_idx << ", chunj_size: " << chunk_size << ", chunk_mask: " << chunk_mask << ", exact_mode: " << exact_mode << std::endl;
      //}
    }
  }
}

void mm2conv_save_weight(int* b, bool b_major, const std::string& weight_file_name, int B, int C, int i, int j, int ROW_BLK_CNT, int COL_BLK_CNT, int blk_r, int blk_c, int INN_BLK_REM, int COL_BLK_REM, int chunk_idx){
  if(b_major == false /*row-major*/){
//    std::cout << __func__ << ": currently mm2conv only support col-major      B matrix, exit" << std::endl;
//    exit(0);
//TODO : transpose the b from col-major to row-major internally
    for(int x = 0 ; x < B ; x++){
      for(int y = 0 ; y < C ; y++){
        b[x*C+y] = b[y*B+x];
      }
    }
//      for(int i = 0 ; i < blk_row ; i++){
//        for(int j = 0 ; j < blk_col ; j++){
//          c[i*blk_col+j] = c_result[j*blk_row+i];
//        }
//      }
  }
  int fd = open(weight_file_name.c_str(), O_RDWR | O_CREAT | O_TRUNC, mode);
  char* map;
  struct stat st;
  fstat(fd, &st);
  map = static_cast<char*>(mmap(NULL, blk_r*blk_c*sizeof(int32_t), PROT_WRITE, MAP_SHARED, fd, 0));
  assert(map != MAP_FAILED);
  if(ftruncate(fd, blk_r*blk_c*sizeof(int32_t)) != 0){
    std::cout << "input file ftruncate fail." << std::endl;
    exit(0);
  }
 // data mapping
  int *temp = (int*) malloc(blk_r*sizeof(int32_t));
  for(int curr_j = 0 ; curr_j < blk_c ; curr_j++){
    for(int idx = 0 ; idx < blk_r ; idx++){
      temp[idx] = (((i == ROW_BLK_CNT-1) && (idx >= INN_BLK_REM && INN_BLK_REM > 0)) || ((j == COL_BLK_CNT -1) && (curr_j >= COL_BLK_REM && COL_BLK_REM > 0)))
                  ?0
                 :b[B*(j*blk_c+curr_j)+i*blk_r+idx] >> (chunk_idx * get_chunk_size());
 //        temp[idx] = b[B*(j*blk_c+curr_j)+i*blk_r+idx] >> (chunk_idx * ge     t_chunk_size());
    }
    memcpy(map+(curr_j*blk_r)*sizeof(int), reinterpret_cast<char*>(temp), blk_r*sizeof(int32_t));
  }
 // end mapping
  munmap(map, blk_r*blk_c*sizeof(int32_t));
  close(fd);
  free(temp);
}
void mm256blk_save_weight(int* b, bool b_major, const std::string& weight_file_name, int B, int C, int i, int j, int ROW_BLK_CNT, int COL_BLK_CNT, int blk_r, int blk_c, int INN_BLK_REM, int COL_BLK_REM, int chunk_num){
  std::cout << __func__ << ": B: " << B << ", C: " << C << ", i: " << i << ", j: " << j << ", blk_r: " << blk_r << ", blk_c: " << blk_c << std::endl;
  if(b_major == false /*row-major*/){
    for(int x = 0 ; x < B ; x++){
      for(int y = 0 ; y < C ; y++){
        b[x*C+y] = b[y*B+x]; // do transpose
      }
    }
  }
  int fd = open(weight_file_name.c_str(), O_RDWR | O_CREAT | O_TRUNC, mode);
  char* map;
  struct stat st;
  fstat(fd, &st);
  map = static_cast<char*>(mmap(NULL, blk_r*blk_c*chunk_num*sizeof(int32_t), PROT_WRITE, MAP_SHARED, fd, 0));
  assert(map != MAP_FAILED);
  if(ftruncate(fd, blk_r*blk_c*chunk_num*sizeof(int32_t)) != 0){
    std::cout << "input file ftruncate fail." << std::endl;
    exit(0);
  }
 // data mapping
  int *temp = (int*) malloc(blk_r*sizeof(int32_t));
  for(int w = 0 ; w < chunk_num ; w++){
    for(int curr_j = 0 ; curr_j < blk_c ; curr_j++){
      for(int idx = 0 ; idx < blk_r ; idx++){
        temp[idx] = (((i == ROW_BLK_CNT-1) && (idx >= INN_BLK_REM && INN_BLK_REM > 0)) || ((j == COL_BLK_CNT -1) && (curr_j >= COL_BLK_REM && COL_BLK_REM > 0)))
                    ?0
                   :/*(w==7&&idx==0)?curr_j:0;// testing dummy number*/b[B*(j*blk_c+curr_j)+i*blk_r+idx] >> (w * get_chunk_size());
      }
      memcpy(map+(w*(blk_r*blk_c)+curr_j*blk_r)*sizeof(int), reinterpret_cast<char*>(temp), blk_r*sizeof(int32_t));
    }
  }
 // end mapping
  munmap(map, blk_r*blk_c*chunk_num*sizeof(int32_t));
  close(fd);
  free(temp);
}

void basic_block_mapping(int A, int B, int C, int& AA, int& BB, int& CC){
  std::cout << __func__ << ": A: " << A << ", B: " << B  << ", C: " << C << std::endl;
  if(A == B && B == C){ // for square mm operations
    if(     A <= 64  ){ AA = 64   ; BB = 64   ; CC = 64 ; }
    else if(A <= 128 ){ AA = 128  ; BB = 128  ; CC = 128; }
    else if(A <= 256 ){ AA = 256  ; BB = 256  ; CC = 256; }
    else if(A <= 512 ){ AA = 512  ; BB = 512  ; CC = 512; }
    else if(A <= 1024){ AA = 1024 ; BB = 1024 ; CC = 1024;}
    else{               AA = 2048 ; BB = 2048 ; CC = 2048;} // assume that 2K is optimal
  }else if(A == 1 && B == C){ // for mv operations with square weight matrix
    if(     B <= 1024){ AA = 1; BB = 1024 ; CC = 1024; }
    else if(B <= 2048){ AA = 1; BB = 2048 ; CC = 2048; }
    else if(B <= 4096){ AA = 1; BB = 4096 ; CC = 4096; }
    else{               AA = 1; BB = 8192 ; CC = 8192; }
  }else if(A == B && C == 1){
    if(     B <= 1024){ AA = 1024; BB = 1024 ; CC = 1; }
    else if(B <= 2048){ AA = 2048; BB = 2048 ; CC = 1; }
    else if(B <= 4096){ AA = 4096; BB = 4096 ; CC = 1; }
    else{               AA = 8192; BB = 8192 ; CC = 1; }
  }else if(A > 1 && B > 1 && C > 1){ // non-perfect square mm operation
  }else if(A == 1 && B != C){ // non-square mv operation
    if(     MAX(B, C) <= 1024){ AA = 1; BB = 1024 ; CC = 1024; }
    else if(MAX(B, C) <= 2048){ AA = 1; BB = 2048 ; CC = 2048; }
    else if(MAX(B, C) <= 4096){ AA = 1; BB = 4096 ; CC = 4096; }
    else{                       AA = 1; BB = 8192 ; CC = 8192; }
  }else{
    std::cout << __func__ << ": input size is not designed yet, A:" << A << ", B:" << B << ", C:" << C << std::endl;
    exit(0);
  }
}

void mm2conv_shape_mapping(int A, int B, int C, int& AA, int& BB, int& CC, int exact_mode, int& IN_W, int& IN_H, int& IN_C, int& F_W, int& F_H, int& S_W, int& S_H, int& OUT_C){
  /*
  mm2conv formula (for (A*B) x (B, C) MM operation):
        A: (IN_W/F_W) * (IN_H/F_H)
        B: F_W * F_H * IN_C
        C: OUT_C
  */
  //map any shape of inputs to supported shapes
  if(exact_mode == 0){  basic_block_mapping(A, B, C, AA, BB, CC);  }
  else{               AA = BB = CC = 256; }

//  AA = A;
//  BB = B;
//  CC = C;

  // list all supported shape
  if(     AA == BB && BB == CC && CC == 64 ){ IN_W = 4 ;  IN_H = 64;  IN_C = 16;  F_W = S_W = 2; F_H = S_H = 2; OUT_C = 64;  }
  else if(AA == BB && BB == CC && CC == 128){ IN_W = 64;   IN_H = 64; IN_C = 4; F_W = S_W =16  ; F_H = S_H = 2; OUT_C = 128; }
  else if(AA == BB && BB == CC && CC == 256){ IN_W = 8;   IN_H = 128; IN_C = 64; F_W = S_W =2  ; F_H = S_H = 2; OUT_C = 256; }
  else if(AA == BB && BB == CC && CC == 512){ IN_W = 512; IN_H = 128; IN_C = 4;  F_W = S_W =64 ; F_H = S_H = 2; OUT_C = 512; }
  else if(AA == BB && BB == CC && CC == 1024){ IN_W = 512; IN_H = 512; IN_C = 4;  F_W = S_W =128; F_H = S_H = 2; OUT_C = 1024;}
  else if(AA == BB && BB == CC && CC == 2048){ IN_W = 2048; IN_H = 64; IN_C = 32; F_W = 4; F_H = 16; S_W = 4; S_H = 16; OUT_C = 2048;}
  else if(AA == 1 && BB == CC && CC == 1024){ IN_W = 1; IN_H = 1; IN_C = 1024; F_W = S_W = 1; F_H = S_H = 1; OUT_C = 1024; }
  else if(AA == 1 && BB == CC && CC == 2048){ IN_W = 1; IN_H = 1; IN_C = 2048; F_W = S_W = 1; F_H = S_H = 1; OUT_C = 2048; }
  else if(AA == 1 && BB == CC && CC == 4096){ IN_W = 1; IN_H = 1; IN_C = 4096; F_W = S_W = 1; F_H = S_H = 1; OUT_C = 4096; }
  else if(AA == 1 && BB == CC && CC == 8192){ IN_W = 1; IN_H = 1; IN_C = 8192; F_W = S_W = 1; F_H = S_H = 1; OUT_C = 8192; }
  else if(AA == 1024 && AA == BB && CC == 1){ IN_W = 256; IN_H = 256; IN_C = 16; }
  else{
    std::cout << __func__ << ": block size: " << AA << "x" << BB << "x" << CC << " is not supported for input size " << A << "x" << B << "x" << C << std::endl;
    exit(0);
  }
}

float get_auto_scale_factor_mm(int*a, int*b, int A, int B, int C){
  int b_max = 0, a_max = 0, b_min = INT_MAX, a_min = INT_MAX;
  for(int i  = 0 ; i< B*C ; i++){
    if(b[i] > b_max){ b_max = b[i]; }
    if(b[i] < b_min){ b_min = b[i]; }
  }
  for(int i  = 0 ; i< A*B ; i++){
    if(a[i] > a_max){ a_max = a[i]; }
    if(a[i] < a_min){ a_min = a[i]; }
  }
// the assumption is that data is normally distributed within this range
  float a_avg =(float)(a_max + a_min)/2;
  float b_avg =(float)(b_max + b_min)/2;
  float IN_SCALE = float(UCHAR_MAX)/float(B * b_avg * a_avg);// (float)((float(255)/float(B))/float(max_value));
  IN_SCALE *= 200.0/255.0;
// statistically if the average result is smaller than 255, no need to scale for exact result
//TODO: maybe consider the feature of normal distribution
  if((B * a_avg * b_avg ) < UCHAR_MAX ){IN_SCALE = 1;}
  std::cout << "in gptpu, the IN_SCALE is decided as: " << IN_SCALE << ", b_min: " << b_min << ", a_min: " << a_min << ", b_max: " << b_max << ", a_max: " << a_max << ", a_avg: " << a_avg << ", b_avg: " <<b_avg << std::endl;
  std::cout << "b_max * IN_SCALE: " << b_max * IN_SCALE << std::endl;
  return IN_SCALE;
}

void search_256mmblk_optimal(std::string& in_dir, int iter){
  set_dev_cnt(1);
  open_devices(0, 1);
  int IN_W, IN_H, IN_C, OUT_C, F_W, F_H, S_W, S_H;
  int MAX_IN_W, MAX_IN_H, MAX_IN_C, MAX_OUT_C, MAX_F_W, MAX_F_H;
  int SEC_IN_W, SEC_IN_H, SEC_IN_C, SEC_OUT_C, SEC_F_W, SEC_F_H;
  int THR_IN_W, THR_IN_H, THR_IN_C, THR_OUT_C, THR_F_W, THR_F_H;
  std::string model_path, sub_model_path, tmp;
  char* token;
  int* in_a;
  double GOPS, MAX_GOPS = 0, SEC_GOPS = 0, THR_GOPS = 0, us, MAX_us = DBL_MAX, SEC_us = DBL_MAX, THR_us = DBL_MAX;
  long long int term1=0, term2=0;
  int A, B, C;
  long long int op_cnt=0;
  glob_t glob_result;
  std::string postfix = "/*_edgetpu.tflite";
  glob((in_dir+postfix).c_str(), GLOB_TILDE, NULL, &glob_result);
//  interpreter_initialization(glob_result.gl_pathc);
  timing s, e;
  for(int i = 0 ; i < glob_result.gl_pathc; i++){
    model_path = glob_result.gl_pathv[i];
    std::stringstream ss(model_path);
    while(std::getline(ss, sub_model_path, '/')){}
    if(strlen(sub_model_path.c_str()) != 70){continue;}
    IN_W  = atoi(sub_model_path.substr(11, 4).c_str());
    IN_H  = atoi(sub_model_path.substr(16, 4).c_str());
    IN_C  = atoi(sub_model_path.substr(21, 4).c_str());
// for OUT_C <= 99999
//    OUT_C = atoi(sub_model_path.substr(26, 5).c_str());
//    F_W   = atoi(sub_model_path.substr(32, 4).c_str());
//    F_H   = atoi(sub_model_path.substr(37, 4).c_str());
// for OUT_C <= 9999
    OUT_C = atoi(sub_model_path.substr(26, 4).c_str());
    if(OUT_C < 8192){ continue; }
    F_W   = atoi(sub_model_path.substr(31, 4).c_str());
    F_H   = atoi(sub_model_path.substr(36, 4).c_str());
   std::cout << "sub_model_path: " << sub_model_path << ", IN_W: " << IN_W << ", IN_H: " << IN_H << ", IN_C: " << IN_C << ", OUT_C: " << OUT_C << ", F_W: " << F_W << ", F_H: " << F_H << std::endl;
    A = (IN_W/F_W)*(IN_H/F_H);
    B = F_W*F_H*IN_C;
    C = OUT_C;
    term1 = 2*B-1;
    term2 = A*C;
    op_cnt = term1*term2;
    build_model(model_path, i);
    build_interpreter(0, i);
    in_a = (int*)malloc(IN_W*IN_H*IN_C*sizeof(int));
    populate_input(in_a, IN_W*IN_H*IN_C, i);
    s = clk::now();
    invoke_model(i, ITER);
    e = clk::now();
    us = (std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count()/1000.0);
    for(int j = 0 ; j < iter-1 ; j++){
      invoke_model(i, ITER);
    }
    e = clk::now();
    us = (std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count()/1000.0)/iter;
    GOPS = (float)op_cnt/(us*1000);
    if(us < MAX_us){
      MAX_IN_W = IN_W;
      MAX_IN_H = IN_H;
      MAX_IN_C = IN_C;
      MAX_OUT_C = OUT_C;
      MAX_F_W = F_W;
      MAX_F_H = F_H;
      MAX_GOPS = GOPS;
      MAX_us = us;
    }else if(us < SEC_us){
      SEC_IN_W = IN_W;
      SEC_IN_H = IN_H;
      SEC_IN_C = IN_C;
      SEC_OUT_C = OUT_C;
      SEC_F_W = F_W;
      SEC_F_H = F_H;
      SEC_GOPS = GOPS;
      SEC_us = us;
    }else if(us < THR_us){
      THR_IN_W = IN_W;
      THR_IN_H = IN_H;
      THR_IN_C = IN_C;
      THR_OUT_C = OUT_C;
      THR_F_W = F_W;
      THR_F_H = F_H;
      THR_GOPS = GOPS;
      THR_us = us;
    }
    if(true){
      printf("GOPS: %10.4f, time: %12.3f\t", GOPS, us);
       std::cout << "shape: " << IN_W << "x" << IN_H << "x" << IN_C << "x" << OUT_C << "x" << F_W << "x" << F_H << ", time: " << us << "(us)" << std::endl;
    }
    free(in_a);
  }
  printf("max us time: %12.3f\t", MAX_us);
  std::cout << "the shape: " << MAX_IN_W << "x" << MAX_IN_H << "x" << MAX_IN_C << "x" << MAX_OUT_C << "x" << MAX_F_W << "x" << MAX_F_H << ", time: " << MAX_us << "(us)" << std::endl;
  printf("sec us time: %12.3f\t", SEC_us);
  std::cout << "the shape: " << SEC_IN_W << "x" << SEC_IN_H << "x" << SEC_IN_C << "x" << SEC_OUT_C << "x" << SEC_F_W << "x" << SEC_F_H << ", time: " << SEC_us << "(us)" << std::endl;
  printf("3rd us time: %12.3f\t", THR_us);
  std::cout << "the shape: " << THR_IN_W << "x" << THR_IN_H << "x" << THR_IN_C << "x" << THR_OUT_C << "x" << THR_F_W << "x" << THR_F_H << ", time: " << THR_us << "(us)" << std::endl;
  printf("op_cnt: %lld\n", op_cnt);
}

struct search_args{
   int tid;
   int size; // section size
   int iter;
   glob_t glob_ptr;
};

int MAX_IN_W, MAX_IN_H, MAX_IN_C, MAX_OUT_C, MAX_F_W, MAX_F_H;
int SEC_IN_W, SEC_IN_H, SEC_IN_C, SEC_OUT_C, SEC_F_W, SEC_F_H;
int THR_IN_W, THR_IN_H, THR_IN_C, THR_OUT_C, THR_F_W, THR_F_H;
double MAX_GOPS = 0, SEC_GOPS = 0, THR_GOPS = 0, us, MAX_us, SEC_us, THR_us;
long long int MAX_op = 0, SEC_op = 0, THR_op = 0;

pthread_mutex_t search_mtx;

void *search_func(void *a){
  struct search_args * args = (struct search_args*)a;
  int tid = args->tid;
  int size = args->size;
  int iter = args->iter;
  glob_t glob_result = (args->glob_ptr);
  timing s, e;
  int IN_W, IN_H, IN_C, OUT_C, F_W, F_H, S_W, S_H;
  std::string model_path, sub_model_path, tmp;
  char* token;
  int* in_a;
  double GOPS;
  long long int term1=0, term2=0, term3=0, term4=0, term5=0;
  int A, B, C;
  long long int op_cnt=0;
//  set_dev_cnt(1);
//  open_devices(0, 1);
  for(int i = tid*size ; i < std::min((tid+1)*size, (int)glob_result.gl_pathc) ; i++){
    std::cout << __func__ << ": i: " << i << ", glob_result.gl_pathc: " << glob_result.gl_pathc << std::endl;
    model_path = glob_result.gl_pathv[i];
    std::stringstream ss(model_path);
    while(std::getline(ss, sub_model_path, '/')){}
    if(strlen(sub_model_path.c_str()) != 70){continue;} // rule out invalid naming
    IN_W  = atoi(sub_model_path.substr(11, 4).c_str());
    IN_H  = atoi(sub_model_path.substr(16, 4).c_str());
    IN_C  = atoi(sub_model_path.substr(21, 4).c_str());
// for OUT_C <= 99999
//    OUT_C = atoi(sub_model_path.substr(26, 5).c_str());
//    F_W   = atoi(sub_model_path.substr(32, 4).c_str());
//    F_H   = atoi(sub_model_path.substr(37, 4).c_str());
// for OUT_C <= 9999
    OUT_C = atoi(sub_model_path.substr(26, 4).c_str());
    F_W   = atoi(sub_model_path.substr(31, 4).c_str());
    F_H   = atoi(sub_model_path.substr(36, 4).c_str());
   std::cout << "sub_model_path: " << sub_model_path << ", IN_W: " << IN_W << ", IN_H: " << IN_H << ", IN_C: " << IN_C << ", OUT_C: " << OUT_C << ", F_W: " << F_W << ", F_H: " << F_H << std::endl;
    A = (IN_W/F_W)*(IN_H/F_H);
    B = F_W*F_H*IN_C;
    C = OUT_C;
    //op_cnt = 2*((F_W*F_H*IN_C)-1)*(IN_W/F_W)*(IN_H/F_H)*OUT_C
    term1 = F_W*F_H;
    term2 = term1*IN_C;
    term3 = IN_W/F_W;
    term4 = IN_H/F_H;
    term5 = 2*(term2-1);
    term5 = term5*term3;  
    term5 = term5*term4;  
    op_cnt = term5*OUT_C;  

    build_model(model_path, i);
    build_interpreter(0, i);
    in_a = (int*)malloc(IN_W*IN_H*IN_C*sizeof(int));
    populate_input(in_a, IN_W*IN_H*IN_C, i);
    s = clk::now();
    invoke_model(i, ITER);
    e = clk::now();
    us = (std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count()/1000.0);
    for(int j = 0 ; j < iter-1 ; j++){
      invoke_model(i, ITER);
    }
    e = clk::now();
    us = (std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count()/1000.0)/iter;
    GOPS = (float)op_cnt/(us*1000);
// naive way to rule out unreasonable results
    if(GOPS > 4000.0 || us  < 400.0){
      std::cout << "GOPS is too high: " << GOPS << " (or time is too short: " << us << ") and error was happened. continue (" << i << "/" << glob_result.gl_pathc << ")" << std::endl;
      free(in_a);
      continue;
    }
    pthread_mutex_lock(&search_mtx);
    if(GOPS > MAX_GOPS){
      MAX_IN_W = IN_W;
      MAX_IN_H = IN_H;
      MAX_IN_C = IN_C;
      MAX_OUT_C = OUT_C;
      MAX_F_W = F_W;
      MAX_F_H = F_H;
      MAX_GOPS = GOPS;
      MAX_us = us;
      MAX_op = op_cnt;
    }else if(GOPS > SEC_GOPS){
      SEC_IN_W = IN_W;
      SEC_IN_H = IN_H;
      SEC_IN_C = IN_C;
      SEC_OUT_C = OUT_C;
      SEC_F_W = F_W;
      SEC_F_H = F_H;
      SEC_GOPS = GOPS;
      SEC_us = us;
      SEC_op = op_cnt;
    }else if(GOPS > THR_GOPS){
      THR_IN_W = IN_W;
      THR_IN_H = IN_H;
      THR_IN_C = IN_C;
      THR_OUT_C = OUT_C;
      THR_F_W = F_W;
      THR_F_H = F_H;
      THR_GOPS = GOPS;
      THR_us = us;
      THR_op = op_cnt;
    }
    if(true){
      printf("GOPS: %10.4f, time: %12.3f\t", GOPS, us);
       std::cout << "shape: " << IN_W << "x" << IN_H << "x" << IN_C << "x" << OUT_C << "x" << F_W << "x" << F_H << ", time: " << us << "(us) | op_cnt: " << op_cnt << ", (" << i << "/" << glob_result.gl_pathc << ")" << std::endl;
    }
    pthread_mutex_unlock(&search_mtx);
  //for(int i = 0 ; i < glob_result.gl_pathc; i++){
    free(in_a);
  }
//  close_devices();
//  printf("op_cnt: %lld\n", op_cnt);
}

void search_random_conv_optimal(std::string& in_dir, int iter, int sec/*the 0-index section*/){
  glob_t glob_result;
  std::string postfix = "/*_edgetpu.tflite";
  glob((in_dir+postfix).c_str(), GLOB_TILDE, NULL, &glob_result);
  std::cout << __func__ << ": # of found files: " << glob_result.gl_pathc << std::endl;
  if(glob_result.gl_pathc <= 0){
    printf("no file found\n");
    exit(0);
  }
  int size = 2500;
  int num = (glob_result.gl_pathc/size)+((glob_result.gl_pathc%size!=0)?1:0); // avoid accumulated mem occupation
  printf("glob_result.gl_pathc/size: %d\n", glob_result.gl_pathc/size);
  printf("glob_result.gl_pathc: %d, size: %d, num: %d\n", glob_result.gl_pathc, size, num);
  pthread_t tid[num];
  struct search_args args[num];
  for(int i = sec ; i < sec+1/*num*/ ; i++){
    args[i].tid  = i;
    args[i].size = size;
    args[i].iter = iter;
    args[i].glob_ptr = glob_result;
    open_devices(0, 1);
    pthread_create(&tid[i], NULL, search_func, (void *)&args[i]);
    pthread_join(tid[i], NULL);
  //  close_devices();
  }
  printf("max GOPS: %10.4f, time: %12.3f, op_cnt: %lld\t", MAX_GOPS, MAX_us, MAX_op);
  std::cout << "the shape: " << MAX_IN_W << "x" << MAX_IN_H << "x" << MAX_IN_C << "x" << MAX_OUT_C << "x" << MAX_F_W << "x" << MAX_F_H << ", time: " << MAX_us << "(us)" << std::endl;
  printf("sec GOPS: %10.4f, time: %12.3f, op_cnt: %lld\t", SEC_GOPS, SEC_us, SEC_op);
  std::cout << "the shape: " << SEC_IN_W << "x" << SEC_IN_H << "x" << SEC_IN_C << "x" << SEC_OUT_C << "x" << SEC_F_W << "x" << SEC_F_H << ", time: " << SEC_us << "(us)" << std::endl;
  printf("3rd GOPS: %10.4f, time: %12.3f, op_cnt: %lld\t", THR_GOPS, THR_us, THR_op);
  std::cout << "the shape: " << THR_IN_W << "x" << THR_IN_H << "x" << THR_IN_C << "x" << THR_OUT_C << "x" << THR_F_W << "x" << THR_F_H << ", time: " << THR_us << "(us)" << std::endl;
}

void search_conv_optimal(std::string& in_dir, int iter){
  set_dev_cnt(1);
  open_devices(0, 1);
  int IN_W, IN_H, IN_C, OUT_C, F_W, F_H, S_W, S_H;
  int MAX_IN_W, MAX_IN_H, MAX_IN_C, MAX_OUT_C, MAX_F_W, MAX_F_H;
  int SEC_IN_W, SEC_IN_H, SEC_IN_C, SEC_OUT_C, SEC_F_W, SEC_F_H;
  int THR_IN_W, THR_IN_H, THR_IN_C, THR_OUT_C, THR_F_W, THR_F_H;
  std::string model_path, sub_model_path, tmp;
  char* token;
  int* in_a;
  double GOPS, MAX_GOPS = 0, SEC_GOPS = 0, THR_GOPS = 0, us, MAX_us, SEC_us, THR_us;
  long long int term1=0, term2=0;
  int A, B, C;
  long long int op_cnt=0;
  glob_t glob_result;
  std::string postfix = "/*_edgetpu.tflite";
  glob((in_dir+postfix).c_str(), GLOB_TILDE, NULL, &glob_result);
//  interpreter_initialization(glob_result.gl_pathc);
  timing s, e;
  for(int i = 0 ; i < glob_result.gl_pathc; i++){
    model_path = glob_result.gl_pathv[i];
    std::stringstream ss(model_path);
    while(std::getline(ss, sub_model_path, '/')){}
    if(strlen(sub_model_path.c_str()) != 70){continue;}
    IN_W  = atoi(sub_model_path.substr(11, 4).c_str());
    IN_H  = atoi(sub_model_path.substr(16, 4).c_str());
    IN_C  = atoi(sub_model_path.substr(21, 4).c_str());
// for OUT_C <= 99999
//    OUT_C = atoi(sub_model_path.substr(26, 5).c_str());
//    F_W   = atoi(sub_model_path.substr(32, 4).c_str());
//    F_H   = atoi(sub_model_path.substr(37, 4).c_str());
// for OUT_C <= 9999
    OUT_C = atoi(sub_model_path.substr(26, 4).c_str());
    F_W   = atoi(sub_model_path.substr(31, 4).c_str());
    F_H   = atoi(sub_model_path.substr(36, 4).c_str());
   std::cout << "sub_model_path: " << sub_model_path << ", IN_W: " << IN_W << ", IN_H: " << IN_H << ", IN_C: " << IN_C << ", OUT_C: " << OUT_C << ", F_W: " << F_W << ", F_H: " << F_H << std::endl;
    A = (IN_W/F_W)*(IN_H/F_H);
    B = F_W*F_H*IN_C;
    C = OUT_C;
    term1 = 2*B-1;
    term2 = A*C;
    op_cnt = term1*term2;
    build_model(model_path, i);
    build_interpreter(0, i);
    in_a = (int*)malloc(IN_W*IN_H*IN_C*sizeof(int));
    populate_input(in_a, IN_W*IN_H*IN_C, i);
    s = clk::now();
    invoke_model(i, ITER);
    e = clk::now();
    us = (std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count()/1000.0);
    for(int j = 0 ; j < iter-1 ; j++){
      invoke_model(i, ITER);
    }
    e = clk::now();
    us = (std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count()/1000.0)/iter;
    GOPS = (float)op_cnt/(us*1000);
    if(GOPS > MAX_GOPS){
      MAX_IN_W = IN_W;
      MAX_IN_H = IN_H;
      MAX_IN_C = IN_C;
      MAX_OUT_C = OUT_C;
      MAX_F_W = F_W;
      MAX_F_H = F_H;
      MAX_GOPS = GOPS;
      MAX_us = us;
    }else if(GOPS > SEC_GOPS){
      SEC_IN_W = IN_W;
      SEC_IN_H = IN_H;
      SEC_IN_C = IN_C;
      SEC_OUT_C = OUT_C;
      SEC_F_W = F_W;
      SEC_F_H = F_H;
      SEC_GOPS = GOPS;
      SEC_us = us;
    }else if(GOPS > THR_GOPS){
      THR_IN_W = IN_W;
      THR_IN_H = IN_H;
      THR_IN_C = IN_C;
      THR_OUT_C = OUT_C;
      THR_F_W = F_W;
      THR_F_H = F_H;
      THR_GOPS = GOPS;
      THR_us = us;
    }
    if(true){
      printf("GOPS: %10.4f, time: %12.3f\t", GOPS, us);
       std::cout << "shape: " << IN_W << "x" << IN_H << "x" << IN_C << "x" << OUT_C << "x" << F_W << "x" << F_H << ", time: " << us << "(us)" << std::endl;
    }
    free(in_a);
  }
  printf("max GOPS: %10.4f, time: %12.3f\t", MAX_GOPS, MAX_us);
  std::cout << "the shape: " << MAX_IN_W << "x" << MAX_IN_H << "x" << MAX_IN_C << "x" << MAX_OUT_C << "x" << MAX_F_W << "x" << MAX_F_H << ", time: " << MAX_us << "(us)" << std::endl;
  printf("sec GOPS: %10.4f, time: %12.3f\t", SEC_GOPS, SEC_us);
  std::cout << "the shape: " << SEC_IN_W << "x" << SEC_IN_H << "x" << SEC_IN_C << "x" << SEC_OUT_C << "x" << SEC_F_W << "x" << SEC_F_H << ", time: " << SEC_us << "(us)" << std::endl;
  printf("3rd GOPS: %10.4f, time: %12.3f\t", THR_GOPS, THR_us);
  std::cout << "the shape: " << THR_IN_W << "x" << THR_IN_H << "x" << THR_IN_C << "x" << THR_OUT_C << "x" << THR_F_W << "x" << THR_F_H << ", time: " << THR_us << "(us)" << std::endl;
  printf("op_cnt: %lld\n", op_cnt);
}

void run_a_model_16x8(std::string& model_path, int iter){
  // Assign # of edgTPUs you want to use
  set_dev_cnt(1);

  // Device initialization
  open_devices(0, 1);

  int* in_a;
  int* out_c;
  timing s, e;
  double us;

  // set_scale
  set_scale(255);
  
  // Call libedgepu runtime library to build model
  build_model(model_path, 0/*model_id*/);

  // Call libedgetpu runtime library to build interpreter
  // At this point, model is pre-decided which edgeTPU it needs to go to.
  build_interpreter(0/*tpu_id*/, 0/*model_id*/);

  // Allocate required size of input array
  // You need to know how large the input array is required, otherwise may segfault
  unsigned long long int size = 1024;//16384*16384;
  in_a  = (int*)malloc(size*sizeof(int));
  out_c = (int*) calloc(size, sizeof(int));
  // A dummy call to pretend filling data to the input array
  srand(9487);
  for(int i = 0 ; i < size ; i++){
    in_a[i] = rand()%256;
  }

  populate_input_16x8(in_a, size/*a pre-determinded size*/, 0);
  std::cout << "start invoking..." << std::endl;
  s = clk::now();

  // The actual invoke function
  // Here, the runtie library needs to copy data from host mem to device mem for the first time. All the following invokation for the same model can use cache to avoid data movement. 
  invoke_model(0, iter/*# of invoke*/);
  e = clk::now();

  // Note: output reading phase is ignored in this simple example.
  simple_populate_output(out_c, 0, 0);

  us = (std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count()/1000.0);
  printf("total invoke time: %12.3f (us)\n", us);
  us = (std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count()/1000.0)/iter;
  printf("avg. invoke time: %12.3f (us), iter = %d.\n", us, iter);
  free(in_a);
  free(out_c);
}
// A general way to run a .tflite model on edgetpu
void run_a_model(std::string& model_path, int iter, int input_size){
  // Assign # of edgTPUs ypu want to use
  set_dev_cnt(1);

  // Device initialization
  open_devices(0, 1);

  int* in_a;
  int* out_c;
  timing s, e;
  double us;

  // set_scale
  set_scale(255);
  
  // Call libedgepu runtime library to build model
  build_model(model_path, 0/*model_id*/);

  // Call libedgetpu runtime library to build interpreter
  // At this point, model is pre-decided which edgeTPU it needs to go to.
  build_interpreter(0/*tpu_id*/, 0/*model_id*/);

  // Allocate required size of input array
  // You need to know how large the input array is required, otherwise may segfault
  unsigned long long int size = input_size;//16384*16384;
  in_a  = (int*)malloc(size*sizeof(int));
  out_c = (int*) calloc(size, sizeof(int));
  // A dummy call to pretend filling data to the input array
  srand(9487);
  for(int i = 0 ; i < size ; i++){
    in_a[i] = rand()%256;
  }
//  int fd = open("./input_vector.txt", O_RDWR | O_CREAT | O_TRUNC, 0664);
//  char *map;
//  map = static_cast<char*>(mmap(NULL, size*sizeof(int32_t), PROT_WRITE, MAP_SHARED, fd, 0));
//  assert(map != MAP_FAILED);
//  if(ftruncate(fd, size*sizeof(int32_t)) != 0){
//    std::cout << __func__ << ": ftruncate fail" << std::endl;
//    exit(0);
//  }
//  memcpy(map, reinterpret_cast<char*>(in_a), size*sizeof(int32_t));
//  munmap(map, size*sizeof(int32_t));
//  close(fd);

  std::cout << "populating input..." << std::endl;
  populate_input(in_a, size/*a pre-determinded size*/, 0);
  std::cout << "start invoking..." << std::endl;
  s = clk::now();

  // The actual invoke function
  // Here, the runtie library needs to copy data from host mem to device mem for the first time. All the following invokation for the same model can use cache to avoid data movement. 
  invoke_model(0, iter/*# of invoke*/);
  e = clk::now();

  // Note: output reading phase is ignored in this simple example.
  std::cout << "populating output ..." << std::endl;
  simple_populate_output(out_c, 0, 0);

  for(int i = 0 ; i < 10 ; i++){
    std::cout << "in[" << i << "]: " << in_a[i] << ", out[" << i << "]: " << out_c[i] << std::endl;
  }

//  int cnt = 0;
//  int i = 0;
//  while(cnt < 10){
//    if(out_c[i] != 0){
//      std::cout << "in[" << i << "]: " << in_a[i] << ", out[" << i << "]: " << out_c[i] << std::endl;
//      cnt++;
//    }
//    i++;
//  }
// 
//  fd = open("./output_vector.txt", O_RDWR | O_CREAT | O_TRUNC, 0664);
//  map = static_cast<char*>(mmap(NULL, size*sizeof(int32_t), PROT_WRITE, MAP_SHARED, fd, 0));
//  assert(map != MAP_FAILED);
//  if(ftruncate(fd, size*sizeof(int32_t)) != 0){
//    std::cout << __func__ << ": ftruncate fail" << std::endl;
//    exit(0);
//  }
//  memcpy(map, reinterpret_cast<char*>(out_c), size*sizeof(int32_t));
//  munmap(map, size*sizeof(int32_t));
//  close(fd);
//

//  FILE *fp;
//  fp = fopen("./output_vector_float.out", "rb");
//  float* ans = (float*) malloc(size*sizeof(float));
//  if(fread(ans, sizeof(float), size, fp) != size){
//    fclose(fp);
//  }
//  fclose(fp);

//  float* myans = (float*) malloc(size*sizeof(float));
//  for(int i = 0 ; i < size ; i++){
//    myans[i] = (float)out_c[i] / 0.0008;
////    std::cout << "my: " << myans[i] << ", ans: " << ans[i] << std::endl;
//  }

//  double MSE = 0;
//  double mean = 0;
//  for(int i = 0 ; i < size; i++){
//    MSE = (MSE * i + pow(myans[i] - ans[i], 2)) / (i+1);
//    mean = (mean * i + ans[i]) / (i+1);
//  }
//  printf("RMSE: %f, mean: %f, RMSE%%: %f %%\n", sqrt(MSE), mean, (sqrt(MSE)/mean)*100);

  us = (std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count()/1000.0);
  printf("total invoke time: %12.3f (us)\n", us);
  us = (std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count()/1000.0)/iter;
  printf("avg. invoke time: %12.3f (us), iter = %d.\n", us, iter);
  //free(in_a);
  //free(out_c);
}
void run_a_pagerank(std::string& model_path, int iter, int input_size){
  // Assign # of edgTPUs ypu want to use
  set_dev_cnt(1);

  // Device initialization
  open_devices(0, 1);

  int* in_a;
  int* out_c;
  timing s, e;
  double us;

  // set_scale
  set_scale(255);
  
  // Call libedgepu runtime library to build model
  build_model(model_path, 0/*model_id*/);

  // Call libedgetpu runtime library to build interpreter
  // At this point, model is pre-decided which edgeTPU it needs to go to.
  build_interpreter(0/*tpu_id*/, 0/*model_id*/);

  // Allocate required size of input array
  // You need to know how large the input array is required, otherwise may segfault
  unsigned long long int size = input_size;//16384*16384;
  in_a  = (int*)malloc(size*sizeof(int));
  out_c = (int*) calloc(size, sizeof(int));
  // A dummy call to pretend filling data to the input array
  srand(9487);
  for(int i = 0 ; i < size ; i++){
    in_a[i] = rand()%2;
  }

  //std::cout << "populating input..." << std::endl;
  populate_input(in_a, size/*a pre-determinded size*/, 0);
  //std::cout << "start invoking..." << std::endl;
  s = clk::now();

  // The actual invoke function
  // Here, the runtie library needs to copy data from host mem to device mem for the first time. All the following invokation for the same model can use cache to avoid data movement. 
  invoke_model(0, iter/*# of invoke*/);
  e = clk::now();

  // Note: output reading phase is ignored in this simple example.
  //std::cout << "populating output ..." << std::endl;
  simple_populate_output(out_c, 0, 0);

  int MAX = 0;
  int MIN = 255;
  long long int sum = 0;  

  for(int i = 0 ; i < size ; i++){
    sum += out_c[i];
    if(out_c[i] < MIN){ MIN = out_c[i]; }
    if(out_c[i] > MAX){ MAX = out_c[i]; }
  }
//  std::cout << "max: " << MAX << ", avg: " << sum/size << ", min: " << MIN << std::endl;
//  for(int i = 0 ; i < 10 ; i++){
//    std::cout << "in[" << i << "]: " << in_a[i] << ", out[" << i << "]: " << out_c[i] << std::endl;
//  }
// ===== start to re-scale back to 0~1 distribution =====
  double tsum = 0;
  for(int i = 0 ; i < size ; i++){
    tsum += out_c[i];
  }
  double* out_float = (double*) malloc(size * sizeof(double));
  for(int i = 0 ; i < size ; i++){
    out_float[i] = out_c[i] / tsum;
  } // tpu out double is ready
// ===== run cblas C++ as ref here =====
  double* in_rank1  = (double*) malloc(size * sizeof(double));
  double* out_rank1 = (double*) malloc(size * sizeof(double));
  double* w1        = (double*) malloc(size*size * sizeof(double));
  double* tmp;
  // init
  for(int i = 0 ; i < size ; i++){
    in_rank1[i] =  1.0 / size;
  }
  std::string name = "~/GPTPU/src/pagerank_1K_iter1_weight.txt";
  std::ifstream f (name/*"~/GPTPU/data/pagerank_1K_iter1_weight.txt"*/, std::ios::in | std::ios::binary);

  if(!f.is_open()){
//    std::cout << "error code: " << strerror(errno) << ", name: " << name << std::endl;
//    exit(0);
  }
  int idx = 0 ;
  union weight_byte{
    float f;
    char c[sizeof(float)];
  };
  union weight_byte the_w_tmp;
  while(f.read(reinterpret_cast<char*>(&the_w_tmp.c[0]), sizeof(float))){
    w1[idx] = the_w_tmp.f;
    idx++;
  }
//  for(int i = 0 ; i < 10 ; i++){
//    for(int j = 0 ; j < 10; j++){
//      std::cout << w1[i*size+j] << " ";
//    }
//    std::cout << std::endl;
//  }
  int page_iter = 1;//= (size == 1024)?5:(size == 2048)?4:(size == 512)?6:(size == 256)?6:(size == 128)?7:1;
  //std::cout << __func__ << ", iter=" << page_iter << std::endl;
  for(int i = 0 ; i < page_iter ; i++){
    cblas_dgemv(CblasRowMajor, CblasNoTrans, size, size, 1, w1, size, in_rank1, 1, 0, out_rank1, 1);
    tmp = in_rank1; in_rank1 = out_rank1; out_rank1 = tmp;
  }
  double MSE = 0;
  double rate = 0;
  double rank1_mean = 0;
  for(int i = 0 ; i < size; i++){
    MSE = (MSE * i + pow(out_rank1[i] - out_float[i], 2)) / (i+1);
    rank1_mean = (rank1_mean * i + out_rank1[i]) / (i+1);
//    if(i < 10){
//      std::cout << "out_rank1: "<<out_rank1[i] << ", out_float: " << out_float[i] << std::endl;
//    }
    rate = (rate * (double)i + fabs(out_rank1[i] - out_float[i])) / (i+1);
  }
  printf("RMSE: %f, out_rank1 avg: %f, RMSE%%: %f %%, errorr rate: %f %% (rate: %f)\n", sqrt(MSE), rank1_mean, (sqrt(MSE)/rank1_mean)*100, (rate/rank1_mean)*100, rate);

  us = (std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count()/1000.0);
  //printf("total invoke time: %12.3f (us)\n", us);
  us = (std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count()/1000.0)/iter;
  //printf("avg. invoke time: %12.3f (us), iter = %d.\n", us, iter);
  //free(in_a);
  //free(out_c);
  GPTPU_cleanup();
}

#define STR_SIZE 256
void fatal(const char *s){
  fprintf(stderr, "Error: %s\n", s);
}
void readinput(float *vect, int grid_rows, int grid_cols, int layers, char *file){
  int i,j,k;
  FILE *fp;
  char str[STR_SIZE];
  float val;

//  if( (fp  = fopen(file, "r" )) ==0 ){
//       std::cout << strerror(errno) << ": " << file << std::endl;    
//       fatal( "The file was not opened" );
//  }
  fp = fopen(file, "r");
  if(fp == NULL){
    std::cout << "file " << file << " is not opened." << std::endl;
    exit(0);
  }

  for (i=0; i <= grid_rows-1; i++)
    for (j=0; j <= grid_cols-1; j++)
      for (k=0; k <= layers-1; k++)
        {
          if (fgets(str, STR_SIZE, fp) == NULL) fatal("Error reading file\n");
          if (feof(fp))
            fatal("not enough lines in file");
          if ((sscanf(str, "%f", &val) != 1))
            fatal("invalid file format");
          vect[i*grid_cols+j+k*grid_rows*grid_cols] = val;
        }
  fclose(fp); 
}

float amb_temp = 80.0;
void computeTempCPU(float *pIn/*powerIn*/, float* tIn/*tempCpoy*/, float *tOut/*answer*/, int nx, int ny, int nz, int numiter){   
  float ce, cw, cn, cs, ct, cb, cc;
  float stepDivCap;

  cc = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);
  ce = cw = cn = cs = 0.136533;
  cc = 0.453667;
  cb = ct = 0.000067;
  stepDivCap = 1.365333;
  int c,w,e,n,s,b,t;
  int x,y,z;
  int i = 0;
  do{
    for(z = 0 ; z < nz; z++){
      for(y = 0 ; y < ny ; y++){
        for(x = 0 ;x < nx ; x++){
          c = x + y * nx + z * nx * ny;
          w = (x == 0)?      c : c-1;
          e = (x == nx - 1)? c : c+1;
          n = (y == 0)?      c : c - nx;
          s = (y == ny - 1)? c : c + nx;
          b = (z == 0)?      c : c - nx * ny; // for nz == 1, b == c
          t = (z == nz - 1)? c : c + nx * ny; // for nz == 1, t == c
          tOut[c] = tIn[c]*cc + tIn[n]*cn + tIn[s]*cs + tIn[e]*ce + tIn[w]*cw + tIn[t/*c*/]*ct + tIn[b/*c*/]*cb + /*(dt/Cap)*/ stepDivCap * pIn[c] + ct*amb_temp;
        }
      }
    }
    float *temp = tIn;
    tIn = tOut;
    tOut = temp;
  }while(i < numiter);
}


void run_a_hotspot(const char* c_model_path, int iter, int input_size, float* pIn, float* tIn, int *tOut){
  std::string model_path(c_model_path);
  double hot_time_sum = 0;
  int hot_cnt = 0;
// ===== run default CPU version ( same as app ) ========
//  char *pfile, *tfile, *ofile;
//  pfile = (char*) malloc(15*sizeof(char));
//  tfile = (char*) malloc(14*sizeof(char));
//  ofile = (char*) malloc(100*sizeof(char));
//  int iterations = iter;
//  int numCols = input_size;
//  int numRows = input_size;
//  int layers = 1;
//
//  float *powerIn, *tempOut, *tempIn, *tempCopy, *TPUCopy, *multiCopy;
//  int total_size = numCols * numRows * layers;
//
//  powerIn = (float*)calloc(total_size, sizeof(float));
//  tempCopy = (float*)malloc(total_size * sizeof(float));
//  TPUCopy = (float*)malloc(total_size * sizeof(float));
//  multiCopy = (float*)malloc(total_size * sizeof(float));
//  tempIn = (float*)calloc(total_size,sizeof(float));
//  tempOut = (float*)calloc(total_size, sizeof(float));
//  float* answer = (float*)calloc(total_size, sizeof(float));
//  float* TPUanswer = (float*)calloc(total_size, sizeof(float));
//  float* multiTPUanswer = (float*)calloc(total_size, sizeof(float));
//
//  strcpy(pfile, "./power_1024x1");
//  strcpy(tfile, "./temp_1024x1");
//
////  readinput(powerIn,numRows, numCols, layers,pfile);
////  readinput(tempIn, numRows, numCols, layers, tfile);
//
//  memcpy(tempCopy,tempIn, total_size * sizeof(float));
//  memcpy(TPUCopy,tempIn, total_size * sizeof(float));
//  memcpy(multiCopy,tempIn, total_size * sizeof(float));
//
//  computeTempCPU(powerIn, tempCopy, answer, numCols, numRows, layers, iterations);


// ===== TPU part =====================================
  // Assign # of edgTPUs ypu want to use
  set_dev_cnt(1);

  // Device initialization
  open_devices(0, 1);

  int* in_a;
  int* out_c;
  timing s, e;
  double us;

  // set_scale
  set_scale(255);
  
  // Call libedgepu runtime library to build model
  build_model(model_path, 0/*model_id*/);

  // Call libedgetpu runtime library to build interpreter
  // At this point, model is pre-decided which edgeTPU it needs to go to.
  build_interpreter(0/*tpu_id*/, 0/*model_id*/);

  // Allocate required size of input array
  // You need to know how large the input array is required, otherwise may segfault
  unsigned long long int size = input_size;//input_size;//16384*16384;
//  float* pIn_TPU_float = (float*) malloc(size*size*sizeof(float));
//  float* tIn_TPU_float = (float*) malloc(size*size*sizeof(float));

//  readinput(pIn_TPU_float, numRows, numCols, layers, pfile);
//  readinput(tIn_TPU_float, numRows, numCols, layers, tfile);

  in_a  = (int*)malloc(2*size*size*sizeof(int));
  out_c = (int*) calloc(size*size, sizeof(int));
//  std::cout << __func__ << ", single input_size: " << input_size << "in_a size: " << 2*size*size << ", out_c size: " << size*size << std::endl;
  int total_size = input_size * input_size;
// qunatize float real data to uint8 array for edgeTPU
  float p_max = 0;
  float t_max = 0;
  float p_scale = 0;
  float t_scale = 0;
  int iMAX = 0;
  int iMIN = 255;
  unsigned long long int isum = 0;  
  int MAX = 0;
  int MIN = 255;
  unsigned long long int sum = 0;  

  for(int idx = 0 ; idx < ((input_size / 256) * (input_size / 256)) ; idx++){
//    std::cout << "idx: " << idx << std::endl;
    p_max = 0;
    t_max = 0;
    for(int i = 0 ; i < total_size ; i++){
      if(pIn[idx*(256*256)+i] > p_max){ p_max = pIn[idx*(256*256)+i]; }
      if(tIn[idx*(256*256)+i] > t_max){ t_max = tIn[idx*(256*256)+i]; }
    }  
    p_scale = p_max / 255.0;
    t_scale = t_max / 255.0;
    for(int i = 0 ; i < size*size ; i++){
      in_a[2*i]   = (int)(tIn[idx*(256*256)+i] / t_scale);
      in_a[2*i+1] = (int)(tIn[idx*(256*256)+i] / t_scale);
    }  
    iMAX = 0;
    iMIN = 255;
    isum = 0;
    for(int i = 0 ; i < 2*size*size ; i++){
      isum += in_a[i];
      if(in_a[i] < iMIN){ iMIN = in_a[i]; }
      if(in_a[i] > iMAX){ iMAX = in_a[i]; }
    }
//    std::cout << "uint8 input max: " << iMAX << ", avg: " << isum/(size*size) << ", min: " << iMIN << ", isum: " << isum << std::endl;
//    std::cout << "populating input..." << std::endl;
    populate_input(in_a, 2*size*size/*a pre-determinded size*/, 0);
//    for(int i = 0 ; i < 10 ; i++){
//      for(int j = 0 ; j < 10 ; j++){
//        std::cout << in_a[i*size+j] << " ";
//      }
//      std::cout << std::endl;
//    }
//    std::cout << "start invoking..." << std::endl;
    s = clk::now();

    // The actual invoke function
    // Here, the runtie library needs to copy data from host mem to device mem for the first time. All the following invokation for the same model can use cache to avoid data movement. 
    invoke_model(0, iter/*# of invoke*/);
    e = clk::now();
//    us = (std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count()/1000.0);
//    hot_time_sum += us;
//    hot_cnt += 1;
//    std::cout << "this us: " << us << ", hot_time_sum: " << hot_time_sum << ", cnt: " << hot_cnt << std::endl;
    // Note: output reading phase is ignored in this simple example.
//    std::cout << "populating output ..." << std::endl;
    simple_populate_output(out_c, 0, 0);
//    std::cout << "done read output" << std::endl;
    MAX = 0;
    MIN = 255;
    sum = 0;  
    for(int i = 0 ; i < size*size ; i++){
      sum += out_c[i];
      if(out_c[i] < MIN){ MIN = out_c[i]; }
      if(out_c[i] > MAX){ MAX = out_c[i]; }
      tOut[idx*(256*256)+i] = out_c[i];
    }
//    std::cout << "idx: " << idx << ", uint8 output max: " << MAX << ", avg: " << sum/(size*size) << ", min: " << MIN << std::endl;
//    for(int i = 0 ; i < 10 ; i++){
//      for(int j = 0 ; j < 10 ; j++){  
//        std::cout << out_c[i*size+j] << " ";
//      }
//      std::cout << std::endl;
//    }
  }



// ===== measurement ========================
//  float *out_float;
//  double MSE = 0;
//  double rate = 0;
//  double rank1_mean = 0;
//  for(int i = 0 ; i < size; i++){
//    MSE = (MSE * i + pow(answer[i] - out_float[i], 2)) / (i+1);
//    rank1_mean = (rank1_mean * i + answer[i]) / (i+1);
//    if(i < 10){
//      std::cout << "answer: "<< answer[i] << ", out_float: " << out_float[i] << std::endl;
//    }
//    rate = (rate * (double)i + fabs(answer[i] - out_float[i])) / (i+1);
//  }
//  printf("RMSE: %f, answer avg: %f, RMSE%%: %f %%, errorr rate: %f %% (rate: %f)\n", sqrt(MSE), rank1_mean, (sqrt(MSE)/rank1_mean)*100, (rate/rank1_mean)*100, rate);
//
//  printf("total invoke time: %12.3f (us)\n", us);
//  us = (std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count()/1000.0)/iter;
//  printf("avg. invoke time: %12.3f (us), iter = %d.\n", us, iter);
  //free(in_a);
  //free(out_c);
  GPTPU_cleanup();
}

struct args_struct{
  int tid;
  int A;
};

void *func(void *arguments){
  struct args_struct *args = (struct args_struct*)arguments;
  int tid = args->tid;
  int A   = args->A;
  long long int run_ns = 0;
  printf("tid: %d, A: %d, dev_cnt: %d\n", tid, A, dev_cnt);
  for(int i = 0 ; i < A/dev_cnt ; i++){
    run_ns += invoke_model(tid, ITER);
  }
  printf("mpmm: dev %d, invoke time: %f (us)\n", tid, run_ns/1000.0);
}

void run_a_model_parallel(std::string& model_path, int iter, int dev_cnt){
  // Assign # of edgTPUs ypu want to use
  set_dev_cnt(dev_cnt);

  // Device initialization
  open_devices(0, 1);

  int* in_a;
  int* out_c;
  timing s, e;
  double us;

  // set_scale
  set_scale(255);
  
  // Call libedgepu runtime library to build model
  for(int i = 0 ; i < dev_cnt ; i++)
    build_model(model_path, i/*model_id*/);

  // Call libedgetpu runtime library to build interpreter
  // At this point, model is pre-decided which edgeTPU it needs to go to.
  for(int i = 0 ; i < dev_cnt ; i++)
    build_interpreter(i/*tpu_id*/, i/*model_id*/);

  // Allocate required size of input array
  // You need to know how large the input array is required, otherwise may segfault
  unsigned long long int size = 4096*16;//16384*16384;
  in_a  = (int*)malloc(size*sizeof(int));
  out_c = (int*) calloc(size, sizeof(int));
  // A dummy call to pretend filling data to the input array
  srand(9487);
  for(int i = 0 ; i < size ; i++){
    in_a[i] = rand()%256;
  }
//  int fd = open("./input_vector.txt", O_RDWR | O_CREAT | O_TRUNC, 0664);
//  char *map;
//  map = static_cast<char*>(mmap(NULL, size*sizeof(int32_t), PROT_WRITE, MAP_SHARED, fd, 0));
//  assert(map != MAP_FAILED);
//  if(ftruncate(fd, size*sizeof(int32_t)) != 0){
//    std::cout << __func__ << ": ftruncate fail" << std::endl;
//    exit(0);
//  }
//  memcpy(map, reinterpret_cast<char*>(in_a), size*sizeof(int32_t));
//  munmap(map, size*sizeof(int32_t));
//  close(fd);

  populate_input(in_a, size/*a pre-determinded size*/, 0);
  std::cout << "start invoking..." << std::endl;
  pthread_t tid[dev_cnt];
  struct args_struct args[dev_cnt];
  for(int i = 0 ; i < dev_cnt ; i++){
    args[i].tid  = i;
    args[i].A    = 4096;
    pthread_create(&tid[i], NULL, func, (void*)&args[i]);
  }
  for(int i = 0 ; i < dev_cnt ; i++){
    pthread_join(tid[i], NULL);
  }
  free(in_a);
}

void quantize(float* data, int* data_int, float& scale, int& mean){

}
#endif
