#pragma once
//#define PROFILING
#define IMPL_8
#ifdef PROFILING
#define PROFILING_TIME 5
#define N_FOR_VIS 500000
#define VISUALIZE 0
#define UNIFORM_GRID 1
#define COHERENT_GRID 1
#define blocksize 128
#define cell_width_mul 2.0f
#define SAVE_FILE_NAME "grid_impl__COHERENT_GRID__GRID_8.csv"
#endif