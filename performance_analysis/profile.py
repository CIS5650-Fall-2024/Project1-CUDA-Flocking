import subprocess
build_cmd = [
	"powershell.exe",
	"start",
	"-FilePath",
	"\"C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\MSBuild\\Current\\Bin\\MSBuild.exe\"",
	"-ArgumentList",
	" \".\\cis565_boids.sln\", \"/property:Configuration=Release\""
]

NAIVE = 1
UNIFORM_GRID = 2
COHERENT_GRID = 3

impl_id_2_name = {
	1 : "NAIVE",
	2 : "UNIFORM_GRID",
	3 : "COHERENT_GRID",
}
def gen_header(time, num_boids, save_file_name, blocksize, impl):
	'''
	generates a header of the following format:

	#pragma once
	#define PROFILING 1
	#ifdef PROFILING
	#define PROFILING_TIME 10
	#define N_FOR_VIS 5000
	#define VISUALIZE 0
	#define blocksize 512
	#define SAVE_FILE_NAME "prof.csv"
	#endif
	'''
	with open('../src/profiling.h', 'w+') as f:
		f.write('#pragma once\n#define PROFILING 1\n')
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

		f.write(f'#define blocksize {blocksize}\n')
		f.write(f"#define SAVE_FILE_NAME \"{save_file_name}\"\n")
		f.write('#endif')


NUM_SAMPLES = 5
num_boids = [ 1000+i*2000 for i in range(NUM_SAMPLES) ]
block_sizes = [ 100+i*50 for in range(NUM_SAMPLES) ]

if __name__ == '__main__':
	# number of boids analysis
	for x in num_boids:
		for impl_id in range(3):
			gen_header(time = 60,
				num_boids = x,
				save_file_name = f"num_boid_test__{impl_id_2_name[impl_id]}.csv",
				block_sizes = 128,
				impl = impl_id)
			subprocess.run(build_cmd)

	# block size analysis
	for x in block_sizes:
		for impl_id in range(3):
			gen_header(time = 60,
				num_boids = 5000,
				save_file_name = f"block_size_test__{impl_id_2_name[impl_id]}.csv",
				block_sizes = x,
				impl = impl_id)
			subprocess.run(build_cmd)