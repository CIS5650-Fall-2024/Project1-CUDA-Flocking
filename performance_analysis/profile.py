import subprocess
import os

'''
start -Wait -NoNewWindow 
-FilePath "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe"
-ArgumentList ".\cis565_boids.sln", "/property:Configuration=Release;OutDir=../build/"
'''
BUILD_CMD = [
	"powershell.exe",
	"start",
	"-Wait",
	"-NoNewWindow",
	"-FilePath",
	"\"C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\MSBuild\\Current\\Bin\\MSBuild.exe\"",
	"-ArgumentList",
	" \"..\\build\\cis565_boids.sln\", \"/property:Configuration=Release;OutDir=..\\build\\\""
]

EXE_CMD = [
	"powershell.exe",
	"start",
	"-Wait",
	"-NoNewWindow",
	"-WorkingDirectory",
	"\"..\\build\\\"",
	"-FilePath",
	"\"..\\build\\cis565_boids.exe\""
]

# kernel implementations
NAIVE = 1
UNIFORM_GRID = 2
COHERENT_GRID = 3

# grid search implementations
GRID_8 = 1
GRID_27 = 2

impl_id_2_name = {
	1 : "NAIVE",
	2 : "UNIFORM_GRID",
	3 : "COHERENT_GRID",
}
grid_impl_id_2_name = {
	1 : "GRID_8",
	2 : "GRID_27",
}

def gen_header(time, num_boids, test_type, block_size, impl, width_mul=2, grid_impl=GRID_8):
	with open('../src/profiling.h', 'w+') as f:
		f.write('#pragma once\n#define PROFILING\n')
		
		if(grid_impl == GRID_8):
			f.write("#define IMPL_8\n")
		
		f.write('#ifdef PROFILING\n')
		f.write(f'#define PROFILING_TIME {time}\n')
		f.write(f'#define N_FOR_VIS {num_boids}\n')
		f.write('#define VISUALIZE 0\n')

		if(impl == NAIVE):
			f.write("#define UNIFORM_GRID 0\n")
			f.write("#define COHERENT_GRID 0\n")
		elif(impl == UNIFORM_GRID):
			f.write("#define UNIFORM_GRID 1\n")
			f.write("#define COHERENT_GRID 0\n")
		else:
			f.write("#define UNIFORM_GRID 1\n")
			f.write("#define COHERENT_GRID 1\n")

		f.write(f'#define blocksize {block_size}\n')
		f.write(f"#define cell_width_mul {width_mul}.0f\n")
		f.write(f"#define SAVE_FILE_NAME \"{test_type}__{impl_id_2_name[impl]}__{grid_impl_id_2_name[grid_impl]}.csv\"\n")
		f.write('#endif')

PROFILE_TIME = 5 # in seconds
NUM_SAMPLES = 1

num_boids = [ 1000+i*2000 for i in range(NUM_SAMPLES) ]
block_sizes = [ 100+i*50 for i in range(NUM_SAMPLES) ]
cell_widths = [ 2+i for i in range(NUM_SAMPLES) ]

def boid_test():
	# number of boids analysis
	for x in num_boids:
		for impl_id in range(1,4):
			gen_header(time = PROFILE_TIME,
				num_boids = x,
				test_type = "num_boid_test",
				block_size = 128,
				impl = impl_id)
			subprocess.call(BUILD_CMD)
			subprocess.call(EXE_CMD)

def block_size_test():
	# block size analysis
	for x in block_sizes:
		for impl_id in range(1,4):
			gen_header(time = PROFILE_TIME,
				num_boids = 5000,
				test_type = "block_size_test",
				block_size = x,
				impl = impl_id)
			subprocess.call(BUILD_CMD)
			subprocess.call(EXE_CMD)

def two_factor_test():
	# 3D graph for framerate as function of (block size, number of boids)
	for x in block_sizes:
		for y in num_boids:
			for impl_id in range(1,4):
				gen_header(time = PROFILE_TIME,
					num_boids = y,
					test_type = "two_factor_test",
					block_size = x,
					impl = impl_id)
				subprocess.call(BUILD_CMD)
				subprocess.call(EXE_CMD)

def cell_width_test():
	# cell width analysis
	for w in cell_widths:
		for impl_id in range(1,4):
			gen_header(time = PROFILE_TIME,
				num_boids = 5000,
				test_type = "cell_width",
				block_size = 128,
				impl = impl_id,
				width_mul = w)
			subprocess.call(BUILD_CMD)
			subprocess.call(EXE_CMD)

def grid_impl_test():
	for (grid_impl_id, w) in [(GRID_27, 1), (GRID_27, 2), (GRID_8, 2)]:
		for impl_id in range(2,4):
			gen_header(time = PROFILE_TIME,
				num_boids = 500000,
				test_type = "grid_impl",
				block_size = 128,
				impl = impl_id,
				width_mul = w,
				grid_impl= grid_impl_id)
			subprocess.call(BUILD_CMD)
			subprocess.call(EXE_CMD)

# num boid, block size, two factors, grid width factor
TESTS = [ 
	# boid_test,
	# block_size_test,
	# two_factor_test,
	# cell_width_test,
	grid_impl_test,
]

if __name__ == '__main__':
	for test in TESTS:
		test()