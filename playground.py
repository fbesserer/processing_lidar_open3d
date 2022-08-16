import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.geometry import TriangleMesh
from open3d.cpu.pybind.visualization import SelectionPolygonVolume

""" Cropping """
# demo_crop_data = o3d.data.DemoCropPointCloud()
# pcd: PointCloud = o3d.io.read_point_cloud(demo_crop_data.point_cloud_path)
# vol: SelectionPolygonVolume = o3d.visualization.read_selection_polygon_volume(demo_crop_data.cropped_json_path)
# chair = vol.crop_point_cloud(pcd)
# pcd = pcd.uniform_down_sample(every_k_points=5)
# o3d.visualization.draw_geometries([pcd],
#                                   zoom=0.7,
#                                   front=[0.5439, -0.2333, -0.8060],
#                                   lookat=[2.4615, 2.1331, 1.338],
#                                   up=[-0.1781, -0.9708, 0.1608])

# demo_crop_data = o3d.data.DemoCropPointCloud()
# pcd = o3d.io.read_point_cloud(demo_crop_data.point_cloud_path)
# vol = o3d.visualization.read_selection_polygon_volume(demo_crop_data.cropped_json_path)
# chair = vol.crop_point_cloud(pcd)
#
# dists = pcd.compute_point_cloud_distance(chair)
# dists = np.asarray(dists)
# ind = np.where(dists > 0.01)#[0]
# pcd_without_chair = pcd.select_by_index(ind)
# o3d.visualization.draw_geometries([pcd_without_chair],
#                                   zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024])
#
# print()

""" Hidden Point Removal """
# print("Convert mesh to a point cloud and estimate dimensions")
# armadillo = o3d.data.ArmadilloMesh()
# mesh: TriangleMesh = o3d.io.read_triangle_mesh(armadillo.path)
# mesh.compute_vertex_normals()
# pcd = mesh.sample_points_poisson_disk(5000)
# diameter = np.linalg.norm(
#     np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
# # o3d.visualization.draw_geometries([pcd])
#
# pcd: PointCloud = pcd.voxel_down_sample(0.5)
#
# print("Define parameters used for hidden_point_removal")
# camera = [0, 0, diameter]
# radius = diameter * 100
#
# print("Get all points that are visible from given view point")
# _:TriangleMesh
# _, pt_map = pcd.hidden_point_removal(camera, radius)
# _.compute_vertex_normals()
# # _ enthält die TraingleMeshes, entwpricht also einer Hülle
# # _ = _.sample_points_uniformly(10000)
# _ = _.sample_points_poisson_disk(10000)
#
#
# print("Visualize result")
# pcd = pcd.select_by_index(pt_map)
#
#
# o3d.visualization.draw_geometries([_])
# # o3d.visualization.draw_geometries([pcd])

""" Subdividing meshes into 4 new triangles """
# mesh = o3d.geometry.TriangleMesh.create_box()
# mesh: TriangleMesh = o3d.geometry.TriangleMesh.create_box()
#
# mesh.compute_vertex_normals()
# print(
#     f'The mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles'
# )
# o3d.visualization.draw_geometries([mesh],  mesh_show_wireframe=True)
# mesh = mesh.subdivide_midpoint(number_of_iterations=1)
# print(
#     f'After subdivision it has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles'
# )
# o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)

""" Outlier Removal """
print("Load a ply point cloud, print it, and render it")
sample_pcd_data = o3d.data.PCDPointCloud()
pcd: PointCloud = o3d.io.read_point_cloud(sample_pcd_data.path)

print("Downsample the point cloud with a voxel of 0.02")
voxel_down_pcd: PointCloud = pcd.voxel_down_sample(voxel_size=0.02)
o3d.visualization.draw_geometries([voxel_down_pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

print("Statistical oulier removal")
cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=1.0)
display_inlier_outlier(voxel_down_pcd, ind)
print()