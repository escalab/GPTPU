#include <fcntl.h>
#include <fstream>
#include <assert.h>
#include <unistd.h>
#include <filesystem>
#include <sys/mman.h>
#include <sys/stat.h>
#include <bits/stdc++.h>
#include "gptpu_utils.h"

namespace gptpu_utils{

#define MAX(a, b) ((a) > (b) ? (a) : (b))

unsigned long long int get_file_size(const std::string& file_path){
    std::filesystem::path p = file_path;
    unsigned long long int ret = std::filesystem::file_size(p);
    assert(ret > 0);
    return ret;
}

/* Check if the file exists or not. */
bool file_exist(std::string& file_path){
    std::ifstream file(file_path);
    return (!file)? false : true;
}

/* Create parent dir of a file if not exist */
void create_file_parent_path(std::string& file_path){
    std::string parent_dir = std::filesystem::path(file_path).parent_path().u8string();
    if(std::filesystem::is_directory(parent_dir)){
        if(! std::filesystem::exists(parent_dir)){
	    std::string cmd = "mkdir -p ";
            system((cmd+parent_dir).c_str());
	    }// else: do nothing as it exists already
    }else{
        std::cout << __func__ << ": parent_dir: " << parent_dir
                  << " isn't a valid dir." << std::endl;
        exit(0);
    }
}

void open_file_with_check(std::fstream& fd, std::string& file_path, std::ios_base::openmode flag){
    fd.open(file_path, flag);
    if(!fd.is_open()){
        std::cout << __func__ << ": file: " << file_path << " open fail." << std::endl;
        exit(0);
    }
}

void open_file_with_check(int& fd, std::string& file_path, int flag){
    fd = open(file_path.c_str(), flag);
    if(fd < 0){ 
        std::cout << __func__ << ": file " << file_path 
		  << " opening fails." << std::endl; 
	exit(0); 
    }
}

void open_file_with_check(int& fd, std::string& file_path, int flag, int mode){
    fd = open(file_path.c_str(), flag, mode);
    if(fd < 0){ 
        std::cout << __func__ << ": file " << file_path 
		  << " opening fails." << std::endl; 
	exit(0); 
    }
}
void mmap_with_check(uint8_t* ptr, int size, int mode, int fd){
    ptr = static_cast<uint8_t*>(mmap(NULL, size, mode, MAP_SHARED, fd, 0));
    assert(ptr != MAP_FAILED);
}
void set_file_size_with_check(int& fd, int size){
    if(ftruncate(fd, size) != 0){
        std::cout << __func__ << ": ftruncate on fd :" << fd << " with size: " 
		  << size << " fails." << std::endl;
    }
}

std::string select_example_model_path(std::string op, int M, int N, int K){
    std::string ret;
    if(op == "conv2d"){ // A conv-based GEMM (mm2conv operation)
        if(M == 1024 && N == 1024 && K == 1024){
            ret = "/home/kernels/templates/mm2conv/1024x1024x1024/conv2d_512-512-4-1024-128-2-128-2-same/conv2d_512-512-4-1024-128-2-128-2-same_edgetpu.tflite";
            return ret;
        }else if(M == 2048 && N == 2048 && K == 2048){
            ret = "/home/kernels/templates/conv2d_2048-64-32-2048-32-2-32-2-same/conv2d_2048-64-32-2048-32-2-32-2-same_edgetpu.tflite";
	    return ret;
	}
    }
// exception
    std::cout << __func__ << ": op " << op
              << " with M: " << M << ", N: " << N << ",K: " << K
              << " is not supported yet" << std::endl;
    exit(0);
}

/*
 Get concatenated params string of given operator.
 This is sepcificly for mm2conv operator that actual
 conv-based params are implicitly choosen.
 */
std::string get_params_string(std::string op, int M, int N, int K){
    std::string ret, token;
    std::string path = select_example_model_path(op, M, N, K);
    std::vector<std::string> tokens;
    std::stringstream full_check(path);
    // get file name with path excluded
    while(getline(full_check, token, '/')){
        tokens.push_back(token);
    }
    assert(tokens.size() >= 1); // at least the actual file name exists
    std::string file_name = tokens[tokens.size()-1];
    std::stringstream file_check(file_name);
    tokens.clear();
    while(getline(file_check, token, '_')){
        tokens.push_back(token);
    }
    //basic: [prefix]_[params]_[postfix]
    assert(tokens.size()>=3); 
    // get actual params section exclude prefix/postfix
    std::string params_section = tokens[tokens.size()-2] ;
    std::stringstream params_check(params_section);
    tokens.clear();
    while(getline(params_check, token, '-')){
        tokens.push_back(token);
    }
    // concate tokens
    ret = "";
    // at least one parameter except for op name and postfix
    assert(tokens.size() >= 3); 
    for(long unsigned int i = 0 ; i < (tokens.size()-1) ; i++){
        ret += tokens[i];    
        ret += " ";
    }
    ret += tokens[tokens.size()-1];
    return ret;
}

std::string select_template_path(std::string op, int M, int N, int K){
    std::string ret;
    if(op == "conv2d"){ // A conv-based GEMM (mm2conv operation)
        if(M == 1024 && N == 1024 && K == 1024){
            //ret = "/home/kernels/templates/mm2conv/1024x1024x1024/conv2d_256-1024-4-1024-64-4-64-4-same_edgetpu.tflite";
            ret = "/home/kernels/templates/mm2conv/1024x1024x1024/conv2d_512-512-4-1024-128-2-128-2-same_edgetpu.tflite";
            return ret;
        }else if(M == 2048 && N == 2048 && K == 2048){
            ret = "/home/kernels/templates/mm2conv/2048x2048x2048/conv2d_2048-64-32-2048-32-2-32-2-same_edgetpu.tflite";
	    return ret;
	}
    }
// exception
    std::cout << __func__ << ": op " << op
              << " with M: " << M << ", N: " << N << ",K: " << K
              << " is not supported yet" << std::endl;
    exit(0);
}

std::string define_kernel_path(std::string op, int M, int N, int K){
    std::string ret;
    if(op == "conv2d"){ // A conv-based GEMM (mm2conv operation)
        if(M == 1024 && N == 1024 && K == 1024){
            ret = "/home/kernels/mm2conv/1024x1024x1024_edgetpu.tflite";
            return ret;
        }else if(M == 2048 && N == 2048 && K == 2048){
            ret = "/home/kernels/mm2conv/2048x2048x2048_edgetpu.tflite";
	    return ret;
	}
    }
// exception
    std::cout << __func__ << ": op " << op
              << " with M: " << M << ", N: " << N << ",K: " << K
              << " is not supported yet" << std::endl;
    exit(0);

}

void create_template_from_example_model(openctpu_buffer* openctpu_buffer,
					std::string op, 
                                        std::string in_path, 
                                        std::string out_path, 
                                        int blk_A, int blk_B, int blk_C){
    /*
        Now only 'mm2conv' with certain shape is supported 
     due to the fact that manual offset searching and mapping is exhaustic.
     
    TODO: separated sub-functions should be introduced if there are mutiple
        type of operators.
     */
    unsigned long long int before_data, after_data;
    if(blk_A == blk_B && blk_B == blk_C && blk_C == 1024){
        before_data = 0x3128; 
        after_data  = 0x103128;
    }else{
        std::cout << __func__ << ": the shape: " 
                  << blk_A << "x" << blk_B << "x" << blk_C 
                  << " is not supported yet." << std::endl;
        exit(0);
    }
    openctpu_buffer->set_template_offsets(before_data, after_data);

    std::fstream in, out;
    unsigned long long int file_size = get_file_size(in_path);
    unsigned long long int max_len = MAX(before_data, file_size - after_data);
    char *buf = new char[max_len];
   
    open_file_with_check(in, 
                         in_path, 
                         std::ios::binary|std::ios::in|
                         std::ios::out|std::ios::ate);
    //create template model's parent dir if not exist
    create_file_parent_path(out_path);
    open_file_with_check(out, 
                         out_path, 
                         std::ios::binary|std::ios::in|
                         std::ios::out|std::ios::trunc);
    
    // copy first section before data section
    in.seekg(0);
    out.seekp(0);
    in.read(&buf[0], before_data);
    out.write(&buf[0], before_data);
    
    // copy section after data section
    in.seekg(after_data);
    out.seekp(before_data);
    in.read(&buf[0], file_size - after_data);
    out.write(&buf[0], file_size - after_data);
  
    delete [] buf;
    in.close();
    out.close();
}

void get_array_minmax(float* data, 
		      float& max, 
		      float& min, 
		      int m, 
		      int n, 
		      int ldm){
    max = FLT_MIN;
    min = FLT_MAX;
    for(int i = 0 ; i < m ; i++){
        for(int j = 0 ; j < n ; j++){
            if(data[i*(ldm)+j] > max){ max = data[i*(ldm)+j]; }
            if(data[i*(ldm)+j] < min){ min = data[i*(ldm)+j]; }
        }
    }
}

void ChooseQuantizationParams(float max, float min, float& scale, uint8_t& mean/*nudged_zero_point*/){
    const float qmin = 0;
    const float qmax = 255;
    scale = (max - min)/(qmax - qmin);
    const float initial_zero_point = qmin - min / scale;
    std::uint8_t nudged_zero_point = 0;
    if(initial_zero_point < qmin){
        nudged_zero_point = qmin;
    }else if(initial_zero_point > qmax){
        nudged_zero_point = qmax;
    }else{
        nudged_zero_point = 
  	    static_cast<std::uint8_t>(std::round(initial_zero_point));
    }
    mean = (uint8_t)nudged_zero_point;
}

void dequantization(uint8_t* in, 
                    float* out, 
                    int depth, 
                    int m, 
                    int n,  
                    int ldn, 
                    float scale,
                    uint8_t mean){
    for(int i = 0 ; i < m ; i++){
        for(int j = 0 ; j < n ; j++){
            out[i*ldn+j] = (in[i*n+j] - depth * mean) * scale;
        }
    }

}

void array_casting(float* in, uint8_t* out, float scale, uint8_t mean, int m, int n, int ldm, bool transpose){
    float transformed_val;
    int offset;
    for(int i = 0 ; i < m ; i++){
        for(int j = 0 ; j < n ; j++){
            transformed_val = mean + in[i*(ldm)+j] / scale;
            offset = (transpose == true)?(j*m+i):(i*n+j); // transpose or not
            out[offset] = (uint8_t)lrint(transformed_val); 
        }
    }
}

/*Copy data section following tflite's pattern (not serial copy) */
void copy_tflite_data_section(uint8_t* out, uint8_t* in, int before_data, int m, int n){
    // TODO: generalization, but now make 1Kx1K as special case, which is a confirmed layout
    if(m == 1024 && n == 1024){ 
        int in_offset = 0;
        int out_offset = 0;
        for(int sec_idx = 0 ; sec_idx < 256/*=m/4*/ ; sec_idx++){ // in GEMM row-major layout: 4 consecutive rows as one s     ection
            for(int in_sec_row_idx = 0; in_sec_row_idx < 4/*# of rows in a section*/ ; in_sec_row_idx++){ // the row idx wit     hin one section (local index)
                for(int in_row_blk_idx = 0 ; in_row_blk_idx < 16/*# of blks in a row*/ ; in_row_blk_idx++){ // the block idx w     ithin one row (local index)
                    for(int in_blk_idx = 0; in_blk_idx < 64/*# of elements in a blk*/ ; in_blk_idx++){ // the data element idx w     ithin one block (local index)
                        in_offset  = (sec_idx * 4 + in_sec_row_idx ) * n/*row size*/ + (in_row_blk_idx * 64 + in_blk_idx);
                        out_offset = in_row_blk_idx * 0x10000 + sec_idx * 0x100 + in_blk_idx * 0x4 + in_sec_row_idx;
                        out[before_data+out_offset] = in[in_offset];
                   }
               }
           }
       }
    }
}

void set_scale_in_tflite_model(uint8_t* dst, 
		               float scale, 
			       int scale_loc, 
			       int scale_num, 
			       int scale_stride_len){
    union scaling the_scale;
    the_scale.f = scale;
    for(int i = 0 ; i < scale_num ; i++){
        memcpy(dst+scale_loc+(i*scale_stride_len),   
	       reinterpret_cast<uint8_t*>(&the_scale.c[0]), 
	       sizeof(float));
    }
}

void reorder_mm2conv_array(uint8_t* in,
                           uint8_t* out,
                           int A,
                           int B,
                           int IN_W,
                           int IN_H,
                           int F_W,
                           int F_H,
                           int IN_C){
    if(!(A == ((IN_W/F_W)*(IN_H/F_H)) && 
         B == (F_W*F_H*IN_C) && ((IN_W % F_W) == 0) &&
         ((IN_H % F_H) == 0) )){
        std::cout << __func__ << ": sizes mismatch, A: " << A << ", B: " << B 
                  << ", IN_H     : " << IN_W << ", IN_H: " << IN_H 
                  << ", F_W: " << F_W << ", F_H: " << F_H 
                  << ", IN_C     : " << IN_C << std::endl;  
        std::cout << "constructed A: " << ((IN_W/F_W)*(IN_H/F_H)) 
                  << ", B: " << F_W*F_H*IN_C << std::endl; 
        exit(0); 
    } 
    int IN_W_cnt = IN_W / F_W ; // block count in row direction 
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
                memcpy(&out[section_offset + (i_len + jump) * block_size/*block offset*/], 
                       &in[section_offset + sid * block_size/*block offset*/], 
                       block_size*sizeof(uint8_t)); 
            } 
        }
    }  
}

std::vector<std::string> split_params_str(const std::string& params_str){
    std::vector<std::string> ret;
    std::stringstream ss (params_str);
    std::string item;
    while(getline(ss, item, ' ')){
        ret.push_back(item);
    }
    return ret;
}

void save_mm2conv_weight(float* data, 
                         const std::string out_file_path,
                         int B,
                         int C,
                         int blk_r,
                         int blk_c,
                         int i,
                         int j,
                         int row_blk_cnt,
                         int col_blk_cnt,
                         int inn_blk_rem,
                         int col_blk_rem){
    int fd = open(out_file_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0777);
    char* map;
    assert((map = static_cast<char*>(mmap(NULL,
                                          blk_r * blk_c * sizeof(float),
                                          PROT_WRITE,
                                          MAP_SHARED,
                                          fd,
                                          0))) != MAP_FAILED);
    assert(ftruncate(fd, blk_r * blk_c * sizeof(float)) == 0);

    //data mapping
    float* temp = (float*) malloc(blk_r * sizeof(float));
    for(int curr_j = 0 ; curr_j < blk_c ; curr_j++){
        for(int idx = 0 ; idx < blk_r ; idx++){
            temp[idx] = (((i == row_blk_cnt - 1) && 
                          (idx >= inn_blk_rem && inn_blk_rem > 0)) || 
                         ((j == col_blk_cnt - 1) && 
                         (curr_j >= col_blk_rem && col_blk_rem > 0)))
                        ?0
                        :data[B * (j * blk_c + curr_j) + i * blk_r + idx];
        }
        memcpy(map + (curr_j * blk_r) * sizeof(float),
               reinterpret_cast<char*>(temp),
               blk_r * sizeof(float));
    }

    //cleanup
    munmap(map, blk_r * blk_c * sizeof(float));
    close(fd);
    free(temp);
}

/* delimiters have to have length == 1 for this hepler function. */
std::string replace_delimiter(std::string in_string,
		       std::string old_delimiter,
		       std::string new_delimiter){
    size_t pos;
    std::string out_string = in_string;
    assert(old_delimiter.size() == 1 && new_delimiter.size() == 1);
    while((pos = out_string.find(old_delimiter)) != std::string::npos){
        out_string.replace(pos, 1, new_delimiter);
    }
    return out_string;
}

} // end namespace


