import open3d as o3d
import time

times = []
for _ in range(100):
    start = time.time()
    tm = o3d.io.read_triangle_mesh("cow_mesh/cow.obj", False)
    end = time.time()
    times.append(end-start)

print(sum(times)/len(times))

# o3d.visualization.draw_geometries([tm])
