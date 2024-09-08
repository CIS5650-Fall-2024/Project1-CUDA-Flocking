import matplotlib.pyplot as plt
import pandas as pd
import os

# Load the Excel file again to complete the code
file_path = '/mnt/data/Performance.xlsx'
xls = pd.ExcelFile("./performance.xlsx")

output_dir = './images'
os.makedirs(output_dir, exist_ok=True)

# Load data from both sheets
boids_df = pd.read_excel(xls, sheet_name='Boids')
blocksize_df = pd.read_excel(xls, sheet_name='Blocksize')
neighborsize_df = pd.read_excel(xls, sheet_name='Neighborsize')

# Clean up the Boids dataframe
boids_df_clean = boids_df.drop([0, 1]).reset_index(drop=True)
boids_df_clean.columns = ['Boids', 'Naïve', 'Scattered', 'Coherent']
boids_df_clean = boids_df_clean.dropna()

# Convert to numeric for plotting
boids_df_clean['Boids'] = pd.to_numeric(boids_df_clean['Boids'])
boids_df_clean['Naïve'] = pd.to_numeric(boids_df_clean['Naïve'])
boids_df_clean['Scattered'] = pd.to_numeric(boids_df_clean['Scattered'])
boids_df_clean['Coherent'] = pd.to_numeric(boids_df_clean['Coherent'])

# Clean up the Blocksize dataframe
blocksize_df_clean = blocksize_df.drop([0, 1]).reset_index(drop=True)
blocksize_df_clean.columns = ['Blocksize', 'Naïve', 'Scattered', 'Coherent']
blocksize_df_clean = blocksize_df_clean.dropna()

# Convert to numeric for plotting
blocksize_df_clean['Blocksize'] = pd.to_numeric(blocksize_df_clean['Blocksize'])
blocksize_df_clean['Naïve'] = pd.to_numeric(blocksize_df_clean['Naïve'])
blocksize_df_clean['Scattered'] = pd.to_numeric(blocksize_df_clean['Scattered'])
blocksize_df_clean['Coherent'] = pd.to_numeric(blocksize_df_clean['Coherent'])

# Clean up the Neighborsize dataframe
neighborsize_df_clean = neighborsize_df.drop([0, 1]).reset_index(drop=True)
neighborsize_df_clean.columns = ['Boids', '27 Neighbors', '8 Neighbors']
neighborsize_df_clean = neighborsize_df_clean.dropna()

# Convert to numeric for plotting
neighborsize_df_clean['Boids'] = pd.to_numeric(neighborsize_df_clean['Boids'])
neighborsize_df_clean['27 Neighbors'] = pd.to_numeric(neighborsize_df_clean['27 Neighbors'])
neighborsize_df_clean['8 Neighbors'] = pd.to_numeric(neighborsize_df_clean['8 Neighbors'])

# Plot the Boids sheet data with log scale for x-axis
plt.figure(figsize=(10, 6))
plt.plot(boids_df_clean['Boids'], boids_df_clean['Naïve'], label='Naïve')
plt.plot(boids_df_clean['Boids'], boids_df_clean['Scattered'], label='Scattered')
plt.plot(boids_df_clean['Boids'], boids_df_clean['Coherent'], label='Coherent')
plt.xscale('log')
plt.title('FPS vs Number of Boids')
plt.xlabel('Number of Boids (Log Scale)')
plt.ylabel('FPS')
plt.legend()
plt.grid(True)
boids_plot_path = os.path.join(output_dir, 'boids_plot.png')
plt.savefig(boids_plot_path)

# Plot the Blocksize sheet data with log scale for x-axis
plt.figure(figsize=(10, 6))
plt.plot(blocksize_df_clean['Blocksize'], blocksize_df_clean['Naïve'], label='Naïve')
plt.plot(blocksize_df_clean['Blocksize'], blocksize_df_clean['Scattered'], label='Scattered')
plt.plot(blocksize_df_clean['Blocksize'], blocksize_df_clean['Coherent'], label='Coherent')
plt.xscale('log')
plt.title('FPS vs Block Size')
plt.xlabel('Block Size (Log Scale)')
plt.ylabel('FPS')
plt.legend()
plt.grid(True)
blocksize_plot_path = os.path.join(output_dir, 'blocksize_plot.png')
plt.savefig(blocksize_plot_path)

# Plot the Neighborsize sheet data with log scale for x-axis
plt.figure(figsize=(10, 6))
plt.plot(neighborsize_df_clean['Boids'], neighborsize_df_clean['27 Neighbors'], label='27 Neighbors')
plt.plot(neighborsize_df_clean['Boids'], neighborsize_df_clean['8 Neighbors'], label='8 Neighbors')
plt.xscale('log')
plt.title('FPS vs Neighbor Size')
plt.xlabel('Number of Boids (Log Scale)')
plt.ylabel('FPS')
plt.legend()
plt.grid(True)
neighborsize_plot_path = os.path.join(output_dir, 'neighborsize_plot.png')
plt.savefig(neighborsize_plot_path)