from threading import Thread

import open3d.cpu.pybind.geometry

try:
    from open3d.cuda.pybind.utility import Vector3dVector, Vector3iVector
    from open3d.cuda.pybind.visualization import draw_geometries, Visualizer
    from open3d.cuda.pybind.geometry import PointCloud, TriangleMesh
except ImportError:
    print("Open3d CUDA not found!")
    from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
    from open3d.cpu.pybind.visualization import draw_geometries, Visualizer
    from open3d.cpu.pybind.geometry import PointCloud, TriangleMesh


def threaded_function():
    try:
        from open3d.cuda.pybind.utility import Vector3dVector, Vector3iVector
        from open3d.cuda.pybind.visualization import draw_geometries, Visualizer
        from open3d.cuda.pybind.geometry import PointCloud
    except ImportError:
        print("Open3d CUDA not found!")
        from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
        from open3d.cpu.pybind.visualization import draw_geometries, Visualizer
        from open3d.cpu.pybind.geometry import PointCloud

    while True:
        vis = Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(TriangleMesh.create_sphere())
        pass


if __name__ == "__main__":
    thread1 = Thread(target=threaded_function)
    thread2 = Thread(target=threaded_function)
    thread3 = Thread(target=threaded_function)

    thread1.start()
    thread2.start()
    thread3.start()

    print("thread finished...exiting")
