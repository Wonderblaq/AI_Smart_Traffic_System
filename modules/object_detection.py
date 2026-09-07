import numpy, torch, ultralytics
print(numpy.__version__)   # should be 1.26.4
print(torch.__version__)   # should be 2.2.0


import numpy as np
import torch
print('NumPy:', np.__version__)
print('Torch:', torch.__version__)
print(torch.from_numpy(np.array([1,2,3])))