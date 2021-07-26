#define CROP_OFFSETS_CNT 8
#define SUB_OFFSETS_CNT  8 

namespace offset{
  namespace mm256conv{
    namespace b16{
      extern unsigned long long int before_data;
      extern unsigned long long int after_data;
      extern unsigned long long int ext_len;
      extern unsigned long long int scale_loc;
      extern unsigned long long int scale_stride_len;
      extern unsigned long long int scale_num;
      extern unsigned long long int f_loc;
      extern unsigned long long int f_locs[20];
    }
    namespace b8{
      extern unsigned long long int before_data;
      extern unsigned long long int after_data;
      extern unsigned long long int ext_len;
      extern unsigned long long int scale_loc;
      extern unsigned long long int scale_stride_len;
      extern unsigned long long int scale_num;
      extern unsigned long long int f_loc;
      extern unsigned long long int f_locs[2];
    }
  }
  namespace mm2conv{
    namespace oneKs{ //1K x 1K (mm)
      extern unsigned long long int before_data;
      extern unsigned long long int after_data;
      extern unsigned long long int ext_len;
      extern unsigned long long int scale_loc;
      extern unsigned long long int scale_stride_len;
      extern unsigned long long int scale_num;
      extern unsigned long long int f_loc;
    }
    namespace oneKmv{ //1K x 1K (mv)
      extern unsigned long long int before_data;
      extern unsigned long long int after_data;
      extern unsigned long long int ext_len;
      extern unsigned long long int scale_loc;
      extern unsigned long long int scale_stride_len;
      extern unsigned long long int scale_num;
    }
    namespace mm_256{ //256 x 256 (mm)
      extern unsigned long long int before_data;
      extern unsigned long long int after_data;
      extern unsigned long long int ext_len;
      extern unsigned long long int scale_loc;
      extern unsigned long long int scale_stride_len;
      extern unsigned long long int scale_num;
      extern unsigned long long int f_loc;
    }
    namespace mm_128{ //128 x 128 (mm)
      extern unsigned long long int before_data;
      extern unsigned long long int after_data;
      extern unsigned long long int ext_len;
      extern unsigned long long int scale_loc;
      extern unsigned long long int scale_stride_len;
      extern unsigned long long int scale_num;
    }
  }
// 
  extern unsigned long long int dense_size;  // size of data (in x out)
  extern unsigned long long int dense_data;
  // after data removed 
  extern unsigned long long int dense_4out;  // 4 x output size
  extern unsigned long long int dense_bias;  // actual bias data
  // count from EOF backward offsets
  extern unsigned long long int dense_matmul;  // size of output 
  extern unsigned long long int dense_flatten;  // size of input
  extern unsigned long long int dense_meta;  // count from EOF backward
  extern unsigned long long int dense_zerop;   // count from EOF backward 
  extern unsigned long long int dense_scale;   // count from EOF backward 
  extern unsigned long long int dense_bias_size;   // size of output
  extern unsigned long long int dense_bias_scale;   // count from EOF backward

// 
  extern unsigned long long int crop_offsets[CROP_OFFSETS_CNT];
//
  extern unsigned long long int sub_offsets[SUB_OFFSETS_CNT];
}
