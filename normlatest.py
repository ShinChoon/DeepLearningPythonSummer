
# importing numpy as np
import numpy as np
 
# importing pyplot as plt
import matplotlib.pyplot as plt
 
# position
pos = 3
 
# scale
scale = 2.5
 
# size
size = 100
 
 
# random seed
# np.random.seed(10)
 
# creating a normal distribution data
values = np.random.normal(pos, scale, size)
 
print("Values:", values)

print("max: ", max(values))
print("min: ", min(values))

# plotting histograph
n, bins, patches = plt.hist(values, 100)

print("n: ", n)
print("bins: ", bins)
print("patches: ", patches)

# plotting mean line
plt.axvline(values.mean(), color='k', linestyle='dashed', linewidth=2)
 
# showing the plot
plt.show()