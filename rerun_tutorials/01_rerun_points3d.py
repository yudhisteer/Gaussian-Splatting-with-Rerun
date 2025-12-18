import rerun as rr
import numpy as np

rr.init(application_id="rerun_example_my_data", spawn=True)

positions = np.zeros((10, 3))
print("Initial positions:", positions)
positions[:,0] = np.linspace(-10,10,10)
print("Updated positions:", positions)

colors = np.zeros((10,3), dtype=np.uint8)
print("Initial colors:", colors)
colors[:,0] = np.linspace(0,255,10)
print("Updated colors:", colors)


rr.log(
    entity_path="my_points",
    entity=rr.Points3D(positions=positions, colors=colors, radii=0.5)
)