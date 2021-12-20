import time
import numpy as np
from configs.server_config import DataConfig
from datasets.ShapeNetPOVRemoval import BoxNet
from frompartialtopose import FromPartialToPose
from main import HyperNetwork
from utils.pose_visualizer import PoseVisualizer
import msvcrt
from configs.server_config import ModelConfig
import tqdm

try:
    from open3d.cuda.pybind.utility import Vector3dVector, Vector3iVector
    from open3d.cuda.pybind.visualization import draw_geometries
    from open3d.cuda.pybind.geometry import PointCloud
except ImportError:
    print("Open3d CUDA not found!")
    from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
    from open3d.cpu.pybind.visualization import draw_geometries
    from open3d.cpu.pybind.geometry import PointCloud

# o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(0))
device = "cuda"
print_time = False

if __name__ == "__main__":

    # Visualizer
    test = PoseVisualizer()

    # Dataset
    valid_set = BoxNet(DataConfig, 10000)

    # Grid resolution
    res = 0.01

    # Pose generator
    model = HyperNetwork.load_from_checkpoint('./checkpoint/best', config=ModelConfig)
    model = model.to(device)
    model.eval()
    generator = FromPartialToPose(model, res, device)

    for s in valid_set:

        # test.reset_coords()

        # for _ in tqdm.tqdm(range(100)):
        while True:
            start1 = time.time()
            partial_points = s[1]  # Take partial point cloud only
            partial_points = np.array(partial_points)  # From list to array

            # Reconstruct partial point cloud
            start = time.time()
            complete_pc_aux, fast_weights = generator.reconstruct_point_cloud(partial_points)
            if print_time:
                print("Reconstruct: {}".format(time.time() - start))

            # Refine point cloud
            start = time.time()
            complete_pc_aux = generator.refine_point_cloud(complete_pc_aux, fast_weights, n=5, show_loss=False)
            if print_time:
                print("Refine: {}".format(time.time() - start))

            # start = time.time()  # TODO REMOVE BOTTLENECK
            pc = PointCloud()
            pc.points = Vector3dVector(complete_pc_aux.squeeze(0).detach().cpu())
            complete_pc_aux = pc
            # print("TIME: {}".format(time.time() - start))

            # Find poses
            start = time.time()  # 1.5 100 1000
            poses = generator.find_poses(complete_pc_aux, mult_res=1.5, n_points=1000, iterations=1000, debug=False)
            if print_time:
                print("Find poses: {}".format(time.time() - start))

            # Visualize results
            partial_pc_aux = PointCloud()
            partial_pc_aux.points = Vector3dVector(partial_points)  # NOTE: not a bottleneck because only 2048 points

            start = time.time()
            test.run(partial_pc_aux, complete_pc_aux, poses)  # just pass partial points
            if print_time:
                print("Render results: {}".format(time.time() - start))

            # If a key is pressed, break
            if msvcrt.kbhit():
                print(msvcrt.getch())
                break

            if print_time:
                print("LOOP TIME: {}".format(time.time() - start1))