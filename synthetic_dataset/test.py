import open3d as o3d
import cv2

# img = cv2.imread("dataset/" + str(which) + "_rgb.png")
# cv2.imshow("RGB", img)
#
# img = cv2.imread("dataset/" + str(which) + "_depth.png")
# cv2.imshow("RGB", img)

which = 0
while True:
    pcd = o3d.io.read_point_cloud("dataset/" + str(which) + "_partial-pc.pcd", format='pcd')
    o3d.visualization.draw_geometries([pcd])

    pcd = o3d.io.read_point_cloud("dataset/" + str(which) + "_complete-pc.pcd", format='pcd')
    o3d.visualization.draw_geometries([pcd])

    which += 1
