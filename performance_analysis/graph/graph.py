import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
files = [
	# "num_boid_test__COHERENT_GRID.csv",
	# "num_boid_test__NAIVE.csv",
	# "num_boid_test__UNIFORM_GRID.csv",
	# "two_factor_test__COHERENT_GRID.csv",
	# "two_factor_test__NAIVE.csv",
	# "two_factor_test__UNIFORM_GRID.csv",
	# "block_size_test__COHERENT_GRID.csv",
	# "block_size_test__NAIVE.csv",
	# "block_size_test__UNIFORM_GRID.csv"
	"cell_width_test__COHERENT_GRID.csv",
	"cell_width_test__NAIVE.csv",
	"cell_width_test__UNIFORM_GRID.csv"
]

if __name__ == '__main__':
	for file in files:
		use_2d_plot = False
		if "num_boid" in file:
			x_name = "num_boids"
			use_2d_plot = True
		elif "block_size" in file:
			x_name = "blocksz"
			use_2d_plot = True
		elif "cell_width" in file:
			x_name = "cell_width"
			use_2d_plot = True
		
		df = pd.read_csv("../../build/" + file)

		pic = None
		if use_2d_plot:
			pic = sns.relplot(
					data=df,
					x=x_name,
					y="ave_fps",
					kind="line",
				)
			pic.set(title = f"fps vs {x_name}")
		else:
			fig = plt.figure()
			ax = fig.add_subplot(projection='3d')
			
			ax.plot_trisurf(df['num_boids'], df['blocksz'], df['ave_fps'],  linewidth=0.2, antialiased=True)
			ax.set_xlabel("num_boids")
			ax.set_ylabel("blocksz")
			ax.set_zlabel("ave_fps")
			ax.set_title(f"fps vs num of boids and block size ")

			pic = ax.figure
		pic.savefig(f"../../images/analysis/{file.split('.')[0]}.png")