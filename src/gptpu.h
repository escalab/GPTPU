#ifndef GPTPU_H
#define GPTPU_H
#include <iostream>
#include <string.h>
#include <stdarg.h>
#include <vector>
#define mode 0777   // file I/O permission bits
#define MAX(a, b) ((a) > (b)? (a): (b))
#define MIN(a, b) ((a) < (b)? (a): (b))

#define queue_size (128)

// 7 bits for 0~127 s.t. 127+127=254 is still within 8 bits
// 4 bits for 0~15 s.t. 15*15=225 is still within 8 bits
// 8 bits for 0~255 s.t. output is still within 8 bits
#define add_chunk_size  7
#define mul_chunk_size  4
#define mean_chunk_size 8
// MM_impl is currently disabled by default setting:
// MM uses "MM_impl" == 1
// MV uses "MM_impl" == 0
// for performance consideration
#define MM_impl 0// either 1:mm2conv or 0:mv_stacking

struct Flatbufs{
  char* buf;
  size_t size;
};

struct OP_node{
  std::string op;// = "None"; // model_name
  int model_id;// = 0;
  bool b_major;// = false;
  bool output_no_chunk;
  bool mm256; // flag for using mm 256, exact mode design or not
  int A;// = 1024;
  int B;// = 1024;
  int C;// = 1024;
  int in;// = 1024;
  int blk_A;// = 1024;
  int blk_B;// = 1024;
  int blk_C;// = 1024;
  int ROW_BLK_CNT;// = 1;
  int INNER_BLK_CNT;// = 1;
  int COL_BLK_CNT;// = 1;
  int* a;// = nullptr;
  int** a_feed;// = nullptr;
  int* b;// = nullptr;
  int* c;// = nullptr;
  int** partial_c;// = nullptr;
  int xi;// = 0;
  int yi;// = 0;
  int i;// = 0;
  int j;// = 0;
  int k;// = 0;
  int idx; // for mm2mul
  int w_chunk_idx;// = 0;
  int start_chunk;
  int chunk_num;
  unsigned long long int offset;// = 0;
  float SCALE;// = 1.0;
};

// ===== openctpu APIs start =====

class openctpu_dimension{
public:
  openctpu_dimension();
  ~openctpu_dimension();
  void set_n_of_dims(int n);
  void set_dims(int x, int y, int z);
  int  get_n_of_dims(void);
  void get_dims(int& x, int& y, int& z);
private:
  int n; // number of dim
  int dims[3]; //
};

struct TILE_INFO{
// tile algorithm may apply
  int A, B, C, blk_A, blk_B, blk_C, ROW_BLK_CNT, INN_BLK_CNT, COL_BLK_CNT, ROW_BLK_REM, INN_BLK_REM, COL_BLK_REM;
};

// kernel-wise configuration arugments, shared by all tensors in at least one operation upto a kernel 
class openctpu_config{
public:
  openctpu_config();
  ~openctpu_config();
  void set_data_type(int);
  void set_chunk_num(int);
  void set_exact_mode(bool);
  void set_mm256_mode(bool);
  void set_blks(int blk_A, int blk_B, int blk_C);
  int get_data_type(void);
  int  get_chunk_num(void);
  bool get_exact_mode(void);
  bool get_mm256_mode(void);
  void get_blks(int& blk_A, int& blk_B, int& blk_C);
private:
  int data_type; // 0: int, 1: float
  int chunk_num;
  bool exact_mode;
  bool mm256_mode;
  int blk_A, blk_B, blk_C; // for tile
};

class openctpu_buffer: public openctpu_dimension{
public:
  openctpu_buffer();
  ~openctpu_buffer();
// set 
  void set_config(openctpu_config *config);
  void set_flags(bool b_major, int tensor_type);
  void set_blk_sizes(int blk_A, int blk_B, int blk_C);
  void set_BLK_CNTs(int A, int B, int C);
  void set_conv_shape(int IN_W, int IN_H, int IN_C, int OUT_C, int F_W, int F_H, int S_W, int S_H);
  void set_mm256_conv_shape(int chunk_num);
  void set_tile_info(int A, int B, int C, int blk_A ,int blk_B, int blk_C);
  void set_params(double scale, int mean);
  void set_maxmin(float max, float min);
  void set_int_or_float(bool type);
// get flags
  openctpu_config* get_config(void);
  bool get_data_type(void);
  bool get_b_major(void);
  bool get_is_out(void);
  bool get_exact_mode(void);
  bool get_mm256_mode(void);
  int  get_chunk_num(void);
  void get_sizes(int& A, int& B, int& C);
  void get_blk_sizes(int& blk_A, int& blk_B, int& blk_C);
  void get_BLK_CNTs(int& ROW_BLK_CNT, int& INN_BLK_CNT, int& COL_BLK_CNT, int& ROW_BLK_REM, int& INN_BLK_REM, int& COL_BLK_REM);
  struct TILE_INFO get_tile_info(void);
  void get_params(double& scale, int& mean);
  void get_maxmin(float& max, float& min);
  void get_int_or_float(bool& type);
//quantizations
  void quantize(float* in, int* out, int length);
  void dequantize(int* in, float* out, int depth, int length);
//conv
  void get_conv_shape(int& IN_W, int& IN_H, int& IN_C, int& OUT_C, int& F_W, int& F_H, int& S_W, int& S_H);
// allocate
  void allocate_a(bool mm256_mode);
  void fill_a(int* data, bool mm256_mode);
  void allocate_c(void* data, int type, bool mm256_mode); // record the pointer from caller

// in public for runtime use
  int** a_blk; // data array
  int** a_feed;
  int*** blk_exact_c; // for mm256_mode use
  int** partial_c;    // for mm2conv use
  int* c;         // for int type use. (with experimental exact mode)
  float* float_c; // for float type use
  openctpu_config *config;
private:
  int chunk_num; // either 16 bits or 8bits
  int tensor_type; // 0: model wieght, 1: input data,  2: output tensor
  bool data_type;
  bool int_or_float;
  bool b_major;
  bool is_out; 
  bool exact_mode;
  bool mm256_mode; // implies mm2conv otherwise
  float scale;
  int   mean;
  float data_max;
  float data_min;
  TILE_INFO tile_info;
// conv
  int IN_W, IN_H, IN_C, OUT_C, F_W, F_H, S_W, S_H;
};

void openctpu_init(int opening_order, int dev_cnt); 
openctpu_config*    openctpu_setConfig(int data_type/*0: int, 1: float*/, bool excat_mode, bool mm256_mode, int chunk_num);
openctpu_dimension* openctpu_alloc_dimension(int dim/*up to 3*/, ...);
openctpu_buffer*    openctpu_create_buffer(openctpu_dimension* dim, void* data, openctpu_config* config, bool b_major, int tensor_type);
void openctpu_enqueue(void(*func)(struct openctpu_buffer* a, struct openctpu_buffer* b, struct openctpu_buffer* c), struct openctpu_buffer* a, struct openctpu_buffer* b, struct openctpu_buffer*c);
void openctpu_invoke_operator(const std::string op, openctpu_buffer* tensor_a, openctpu_buffer* tensor_b, openctpu_buffer* tensor_c);
void openctpu_sync(openctpu_buffer* tensor_c);
void openctpu_verify(openctpu_buffer* a, int* ref, int dim, ...);
void openctpu_verify(openctpu_buffer* a, float* ref, int dim, ...);
// ===== openctpu APIs end =====
void set_exact_mode(int exact_mode);
void set_chunk_num(int x/*input bit width*/);
void set_start_chunk(int start_chunk);
void set_chunk_size(int size);
void set_data_type(const std::string&);
void set_scale(float scale);
void set_zp(int zero_point);
void set_tpu_id(int tpu_id); // can be removed later
void set_iteration(int iter);
void set_verbose(int verbose);
void set_breakdown(int breakdown);
void set_dev_cnt(int dev_cnt);
void set_block(int A, int B);
void set_ramdisk(int ramdisk);
void open_devices(int opening_order);

// tools API
void search_conv_optimal(std::string& in_dir, int iter); 
void search_random_conv_optimal(std::string& in_dir, int iter, int sec); 
void search_256mmblk_optimal(std::string& in_dir, int iter); 
void run_a_model(std::string& model_path, int iter, int input_size); 
void run_a_pagerank(std::string& model_path, int iter, int input_size); 
void run_a_hotspot(const char* model_path, int iter, int input_size, float *pIn, float *tIn, int *tOut); 
void run_a_model_16x8(std::string& model_path, int iter); //experimental
void run_a_model_parallel(std::string& model_path, int iter, int dev_cnt); 

int gptpu_mpmm(const std::string& model_path, int* a, int* b, int* c, int A, int B, bool b_major); // min-plus matrix multiplication => Cij = min(Aik, Bkj)
int gptpu_mm(int* a, int* b, int* c, int A, int B, int C, bool b_major); // 0 as row-major, 1 as col-major
int gptpu_mv(int* a, int* b, int* c, int A, int B);
int gptpu_imv(int* a, int* b, int* c, int A, int B);
int gptpu_smv(int* a, int* b, int* c, int A, int B, float scale); // scaling mv with one more paramete on wieghts
const std::string none = std::string();
float gptpu_conv2D(int* a, int* f/*filter*/, int* c, int A, int B, int M/*filter width*/, int N/*fitler height*/, const std::string& padding = none/*same or replication*/);
void gptpu_pagerank(double* w, double* in_rank, double* out_rank, int size, int iter);
// same padding pad with zeros, replication pad with border values
int gptpu_mac(int*a, int* b, int&c , int A);
int gptpu_vs(int* a, int* c, int A, float scale);
int gptpu_tanh(int* a, int* c, int A, int B); // tanh with size A*B
int gptpu_relu(int* a, int* c, int A, int B); // relu with size A*B
int gptpu_add(int* a, int* b, int* c, int A, int B); //element-wise matrix addition
int gptpu_sub(int* a, int* b, int* c, int A, int B); //element-wise matrix substraction
int gptpu_mul(int* a, int* b, int* c, int A, int B); //element-wise matrix multiply
int gptpu_mean(int* a, int* c, int A, int B);        //reduction operation: mean over a vector
int gptpu_sum(int* a, int* c, int A, int B);         //reduction operation: sum  over a vector
int gptpu_max(int* a, int* c, int A, int B);         //reduction operation: max  over a vector
int gptpu_min(int* a, int* c, int A, int B);         //reduction operation: min  over a vector
int gptpu_crop(int*a, int*c, int A, int B, int blk_row, int blk_col, int start_i, int start_j, bool transpose);
int gptpu_ext( int*a, int*c, int blk_row, int blk_col, int A, int B, int start_i, int start_j, bool transpose); // reverse op of crop

int gptpu_tf_mm(int* a, int* b, int* c, int A, int B, int C, bool b_major);
int gptpu_tf_mv(int* a, int* b, int* c, int A, int B); //matrix-vector multiplication
int gptpu_tf_bmv(int* a, int* b, int* c, int A, int B); //matrix-vector multiplication
int gptpu_tf_log(int* a, int* c, int A, int B);
int gptpu_tf_vs(int* a, int* c, int A, float scale); //vector scaling
int gptpu_tf_add(int* a, int* b, int* c, int A, int B); //element-wise matrix addition
int gptpu_tf_sub(int* a, int* b, int* c, int A, int B);  //element-wise matrix substraction
int gptpu_tf_mul(int* a, int* b, int* c, int A, int B); // element-wise matrix multiply
int gptpu_tf_maxpool(int* a, int* c, int A, int B); // element-wise MAX() function on matrix
int gptpu_tf_mean(int* a, int* c, int A, int B); // element-wise MEAN() function on matrix
int gptpu_tf_crop(int*a, int*c, int A, int B, int blk_row, int blk_col, int start_i, int start_j, bool transpose);

int gptpu_black(int* a, int* b, int*c, int A, int B);

#endif
