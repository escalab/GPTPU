#include "gptpu.h"
#include "offset.h"
#ifndef OFFSET_H
#define OFFSET_H
namespace offset{
//mm2conv shape: IN_W, IN_H, IN_C, F_W, F_H, S_W, S_H
  namespace mm256conv{ // 8192x8x16x8x2x8x2 for blk_A:blk_B:blk_C => 4Kx256x4K
    namespace b16{
      unsigned long long int before_data      = 0x320c;
      unsigned long long int after_data       = 0x107bc0; 
      unsigned long long int ext_len          = 0xab4; // extend zeros (last zero gap total)
  //
      unsigned long long int scale_loc        = 0x10c918;
      unsigned long long int scale_stride_len = 0xa898; //0x1171b0 - 0x10c918
      unsigned long long int scale_num        = 2; 
      unsigned long long int f_loc            = 0x10dbfc; // the factor that is porpotinal to max * scale
      unsigned long long int f_locs[20]       = {0x10dbfc,
                                                 0x10dd0c,
                                                 0x10de1c,
                                                 0x10df2c,
                                                 0x10f40c,
                                                 0x10f51c,
                                                 0x1109fc,
                                                 0x110b0c,
                                                 0x1120ac,
                                                 0x1121bc,
                                                 0x1122cc,
                                                 0x1123dc,
                                                 0x11397c,
                                                 0x113a8c,
                                                 0x113b9c,
                                                 0x113cac,
                                                 0x11524c,
                                                 0x11535c,
                                                 0x11546c,          
                                                 0x11557c};
    //                        |  the diff to previous
    // 1st          = 0x10dbfc;
    // second f_loc = 0x10dd0c;  0x110
    // 3rd    f_loc = 0x10de1c;  0x110
    //              = 0x10df2c;  0x110
    //              = 0x10f40c;  0x14e0
    //              = 0x10f51c;  0x110
    //              = 0x1109fc;  0x14e0
    //              = 0x110b0c;  0x110
    //              = 0x1120ac;  0x15a0
    //              = 0x1121bc;  0x110
    //              = 0x1122cc;  0x110
    //              = 0x1123dc;  0x110
    //              = 0x11397c;  0x15a0
    //              = 0x113a8c;  0x110
    //              = 0x113b9c;  0x110
    //              = 0x113cac;  0x110
    //              = 0x11524c;  0x15a0
    //              = 0x11535c;  0x110
    //              = 0x11546c;  0x110
    // 20th         = 0x11557c;  0x110
    }
    namespace b8{
      unsigned long long int before_data      = 0x320c;
      unsigned long long int after_data       = 0x85c40; 
      unsigned long long int ext_len          = 0xb34; // extend zeros (last zero gap total)
  //
      unsigned long long int scale_loc        = 0x89408;
      unsigned long long int scale_stride_len = 0x3da4; //0x8d1ac - 0x89408
      unsigned long long int scale_num        = 2; 
      unsigned long long int f_loc            = 0x10dbfc; // the factor that is porpotinal to max * scale
      unsigned long long int f_locs[2]       = {0x8a2bc,
                                                0x8b62c};
    }
  }
  namespace mm2conv{
    namespace oneKs{ // 512x512x4x128x2x128x2
      unsigned long long int before_data      = 0x320c;
      unsigned long long int after_data       = 0x104bc0; // starting loc after ext_zero
      unsigned long long int ext_len          = 0x9b4; // extend zeros (last zero gap total)
      unsigned long long int scale_loc        = 0x1087e4;//old : 0x1095f0;
      unsigned long long int scale_stride_len = 0x29c8; //0x10b1ac - 0x1087e4
      unsigned long long int scale_num        = 2; 
      unsigned long long int f_loc            = 0x1095ec; // the factor that is porpotinal to max * scale
    }
    namespace oneKmv{ // 1x1x1024x1x1x1x1
      unsigned long long int before_data      = 0x320c;
      unsigned long long int after_data       = 0x104540;
      unsigned long long int ext_len          = 0x334; 
      // the same scale setting duplicated in 16 locations
      unsigned long long int scale_loc        = 0x109070;
      unsigned long long int scale_stride_len = 0x110;
      unsigned long long int scale_num        = 16;
    }
    namespace mm_256{ // 8x128x64x2x2x2x2
      unsigned long long int before_data      = 0x320c;
      unsigned long long int after_data       = 0x13cc0;
      unsigned long long int ext_len          = 0x6b4;
      unsigned long long int scale_loc        = 0x16874;
      unsigned long long int scale_stride_len = 0x2938; // 0x191ac - 0x16874
      unsigned long long int scale_num        = 2; 
      unsigned long long int f_loc            = 0x1766c;
    }
    namespace mm_128{ // 64x64x4x128x16x2
      unsigned long long int before_data      = 0x320a;
      unsigned long long int after_data       = 0x7cbe;
      unsigned long long int ext_len          = 0x8b4;
      unsigned long long int scale_loc        = 0xb66e;
      unsigned long long int scale_stride_len = 0x0; //dummy
      unsigned long long int scale_num        = 1; //dummy
    }
  }
// ===== offsets for dense model =====
  unsigned long long int dense_size = 0x74;  // size of data (in x out)
  unsigned long long int dense_data = 0x78;
  // after data removed 
  unsigned long long int dense_4out = 0x88;  // 4 x output size
  unsigned long long int dense_bias = 0x8c;  // actual bias data
  // count from EOF backward offsets
  unsigned long long int dense_matmul     = 0x230;  // size of output 
  unsigned long long int dense_flatten    = 0x1c0;  // size of input
  unsigned long long int dense_meta       = 0x150;  // count from EOF backward
  unsigned long long int dense_zerop      = 0x108;  // count from EOF backward
  unsigned long long int dense_scale      = 0xf8;   // count from EOF backward 
  unsigned long long int dense_bias_size  = 0xb8;   // size of output
  unsigned long long int dense_bias_scale = 0x74;   // count from EOF backward

// ===== offsets for crop model =====

  unsigned long long int crop_offsets[CROP_OFFSETS_CNT] = {0x80,   /* crop_i    , row offset in crop_A */
                                                           0x84,   /* crop_j    , col offset in crop_B */
                                                           0x98,   /* crop_end_i, down-left corner i   */
                                                           0x9c,   /* crop_end_j, down-left cornet j   */
                                                           0x124,  /* crop_A    , leading dim. of row  */
                                                           0x128,  /* crop_B    , leading dim. of col  */
                                                           0x26c,  /* crop_blk_A, crop row size        */
                                                           0x270}; /* crop_blk_B, crop col size        */

// ===== offsets for sub model =====
  unsigned long long int sub_offsets[SUB_OFFSETS_CNT] = {0xec,   /* sub_size_sum, A + B */
                                                         0xf0,   /* B1          , B     */
                                                         0x194,  /* A1          , A     */
                                                         0x198,  /* B2          , B     */
                                                         0x20c,  /* A2          , A     */
                                                         0x210,  /* B3          , B     */
                                                         0x290,  /* A3          , A     */
                                                         0x294}; /* B4          , B     */

}
#endif
