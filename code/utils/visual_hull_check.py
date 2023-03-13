import mcubes
import numpy as np
import open3d as o3d
from tqdm import tqdm


def visual_hull_creation(mask_list, P_list, num_grid=256):
    print("Creating the visual hull...")
    # intialize a numpy 3d array
    # bounded in [-1, 1]^3, with num_grid dots along each dimension
    # numpy conversion
    axis0, axis1, axis2 = np.meshgrid(range(num_grid), range(num_grid), range(num_grid), indexing='ij')

    # right-hand coordinates
    def normalize(axis_3d_array):
        # to [-1, 1]
        return 2 * axis_3d_array / num_grid - 1

    xx = normalize(axis1)  # (n, n, n)
    yy = normalize(axis0)
    zz = normalize(axis2)

    # concatenate 3D arrays to a 4D array along the last dimension
    voxel_world_coordinates = np.stack((xx, yy, zz), axis=-1)
    voxel_world_coordinates_flatten = voxel_world_coordinates.reshape((-1, 3))  # (num_voxel, 3)
    voxel_world_coordinates_homo = np.concatenate((voxel_world_coordinates_flatten,
                                                   np.ones((num_grid ** 3, 1))), axis=-1)  # (num_voxel, 4)

    assert len(mask_list) == len(P_list)
    num_view = len(mask_list)
    view_fill = np.zeros(num_grid ** 3)
    for i in tqdm(range(num_view)):
        mask = mask_list[i]
        img_height, img_width = mask.shape[:2]
        P = P_list[i]  # (3, 4) or (4, 4) projection matrix, projecting a point in the world space to the screen space
        pixel_coord_homo = (P @ voxel_world_coordinates_homo.T).T
        pixel_xx = pixel_coord_homo[:, 0] / pixel_coord_homo[:, -1]
        pixel_yy = pixel_coord_homo[:, 1] / pixel_coord_homo[:, -1]

        # numpy conversion
        pixel_axis0 = np.floor(np.clip(pixel_yy, 0, img_height - 1)).astype(int)
        pixel_axis1 = np.floor(np.clip(pixel_xx, 0, img_width - 1)).astype(int)

        view_fill += mask[pixel_axis0, pixel_axis1].astype(int)
    view_fill = view_fill.reshape((num_grid, num_grid, num_grid))
    visual_hull = view_fill > (0.9 * num_view)

    vertices, triangles = mcubes.marching_cubes(visual_hull, 0.5)
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(triangles)
    mesh_o3d.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_o3d])


if __name__ == "__main__":
    visual_hull_creation([], [])
