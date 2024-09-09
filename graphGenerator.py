import matplotlib.pyplot as plt

# # Data
# x = [5000, 10000, 20000, 25000, 50000]
# naive = [394.49, 203.603, 68.1205, 48.8538, 15.6414]
# uniform = [905.843, 854.907, 875.946, 832.915, 760.093]
# coherent = [986.252, 971.805, 967.502, 962.861, 940.781]
#
# # Create plot
# plt.figure(figsize=(10, 6))
# plt.plot(x, naive, label='Naive', marker='o', linestyle='-', color='blue')
# plt.plot(x, uniform, label='Uniform', marker='s', linestyle='--', color='green')
# plt.plot(x, coherent, label='Coherent', marker='^', linestyle='-.', color='red')
#
# plt.xlabel('Number of Boids')
# plt.ylabel('Average FPS')
# plt.title('Average FPS vs Number of Boids in 60 seconds')
# plt.xticks(x)
# plt.legend()
# plt.grid(True)
# plt.show()

# Data
x = [32, 64, 128, 256, 512]
naive = [812.801, 834.839, 803.663, 804.612, 730.348]
uniform = [1186.55, 1193.94, 1177.4, 1191.46, 1187.47]
coherent = [1171.34, 1174.16, 1162.91, 1176.3, 1170.4]

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(x, naive, label='Naive', marker='o', linestyle='-', color='blue')
plt.plot(x, uniform, label='Uniform', marker='s', linestyle='--', color='green')
plt.plot(x, coherent, label='Coherent', marker='^', linestyle='-.', color='red')

plt.xlabel('Block Size')
plt.ylabel('Average FPS')
plt.title('Average FPS vs Block Size in 60 seconds')
plt.xticks(x)
plt.legend()
plt.grid(True)
plt.show()
