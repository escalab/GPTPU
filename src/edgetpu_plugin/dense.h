void          edgetpu_cleanup(void);
void          dense_set_chunk_size(int chunk_size);
int           dense_get_chunk_size();
void          set_verbose(int verbose);
long long int ListTheDevices(int verbose, int& count);
long long int open_device(int tpuid, int verbose);
void          close_device(int tpuid, int verbose);
//int           interpreter_initialization(int model_cnt);
int           read_input(const std::string& input_path, int input_mode, int model_id);
long long int build_model(const std::string& model_path, int model_id);
long long int build_model_from_buffer(char* buf, size_t size, int model_id);
long long int build_interpreter(int tpuid, int model_id);
int           run_model(int iter, int* output_size, int model_id, int verbose);
int           run_modelV2(int* b, int size, int iter, int& output_size, int* pre_result, 
                          int tpu_id, int model_id, const std::string& data_type, 
                          int w_chunk_idx, int in_chunk_idx, int verbose, 
                          long long int& mem_ns, long long int& run_ns, long long int& pop_ns);
int           run_modelV3(int* b, int size, int iter, int& output_size, int* partial_result, 
                          int A ,int C, int blk_A, int blk_C, int i, int k, int tpu_id, int model_id, 
                          int verbose, long long int& mem_ns, long long int& run_ns, long long int& pop_ns);
int           run_element_wise_modelV2(int* a, int* b, int size, int iter, int& output_size, 
                                       int* pre_result, int tpu_id, int model_id, int verbose, 
                                       long long int& mem_ns, long long int& run_ns, long long int& pop_ns);
int           run_element_wise_modelV3(int* a, int* b, int size, int iter, int& output_size, 
                                       int* pre_result, int tpu_id, int model_id, 
                                       int w_chunk_idx, int in_chunk_idx, int verbose, 
                                       long long int& mem_ns, long long int& run_ns, long long int& pop_ns);
int           run_element_wise_modelV4(int* a, int* b, int size, int iter, int& output_size, 
                                       int* pre_result, int tpu_id, int model_id, 
                                       int w_chunk_idx, int in_chunk_idx, int verbose, 
                                       long long int& mem_ns, long long int& run_ns, long long int& pop_ns);
int           populate_input(int* b, int size, int model_id);
int           populate_input_16x8(int* b, int size, int model_id);
int           populate_input_uint8(uint8_t* b, int size, int model_id);
int           populate_input_chunking(int* b, int size, int model_id, int chunk_idx, const std::string& data_type);
int           populate_input_exact(int* b, int size_A, int size_C, int chunk_num, int model_id, const std::string& data_type);
long long int invoke_model(int model_id, int ITER);
int           populate_output(int* partial_result, int A , int C, int blk_A, int blk_C, int i , int k, int model_id);
int       simple_populate_output(int* result, int model_id, int verbose);
int           populate_output_exact(int* partial_result, int A , int C, int blk_A, int blk_C, 
                                    int i , int k, int model_id, int chunk_num, float SCALE);
int           populate_output_chunking(int* partial_result, int A , int C, int blk_A, int blk_C, 
                                       int i , int k, int model_id, int in_chunk_idx, int w_chunk_idx, float SCALE);
int           populate_element_wise_input_chunking(int* a, int* b, int size, int xi, int yi, int chunk_size, int model_id, std::string& model_name, long long int& mem_ns);
int           populate_mm2mul_input_chunking(int* a, int blk_A, int blk_B, unsigned long long int offset, int* b, int idx, int i, int j, int ROW_BLK_CNT, int COL_BLK_CNT, int size, int xi, int yi, int chunk_size, int model_id, std::string& model_name, long long int& mem_ns);
long long int populate_element_wise_output(int* partial_result, int size, int model_id, 
                                           float SCALE);
long long int populate_element_wise_output_chunking(int* partial_result, int size, int model_id, 
                                                    int in_chunk_idx, int w_chunk_idx, float SCALE);

