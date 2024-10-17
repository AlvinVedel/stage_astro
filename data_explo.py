import numpy as np
import matplotlib.pyplot as plt

data = np.load("/lustre/fswork/projects/rech/kof/uve94ap/CUBES_HSC/PHOT/DEEP2/DEEP2_v11_ui_0130_phot.npz", allow_pickle=True)

print(data)

cube = data["cube"]
meta = data["info"]

print(cube.shape)
print(meta)
