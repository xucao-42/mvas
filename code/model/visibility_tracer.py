import sys

sys.path.append(".")
sys.path.append("../model")
import torch.nn as nn
from rend_util import *
from general_utils import device
import open3d as o3d
from open3d.visualization import *


def visualize_visibility_intesection(camera_centers, point, dist_from_point, unit_ray_direction):
    c = camera_centers.detach().cpu().numpy()
    num_cams, _ = c.shape
    point = point.detach().cpu().numpy()
    dist_from_point = dist_from_point.detach().cpu().numpy()
    unit_ray_direction = unit_ray_direction.detach().cpu().numpy()

    sphere = o3d.geometry.TriangleMesh.create_sphere(1, 200)
    sphere.compute_vertex_normals()

    point_end = point[None, ...] + unit_ray_direction * dist_from_point[..., None]
    points_end = np.concatenate([point_end, c])
    lines_end = [[i, i + num_cams] for i in range(num_cams)]
    colors_end = [[0, 1, 0] for i in range(len(lines_end))]

    points = np.concatenate([point[None, ...], point_end])
    lines = [[0, i + 1] for i in range(num_cams)]
    colors = [[1, 0, 0] for i in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    line_set_end = o3d.geometry.LineSet()
    line_set_end.points = o3d.utility.Vector3dVector(points_end)
    line_set_end.lines = o3d.utility.Vector2iVector(lines_end)
    line_set_end.colors = o3d.utility.Vector3dVector(colors_end)

    draw_geometries([line_set, line_set_end])
    pass


class VisibilityTracing(nn.Module):
    def __init__(
            self,
            object_bounding_sphere=1.0,
            sphere_tracing_iters=30,
            initial_epsilon=1e-3
    ):
        super().__init__()

        self.object_bounding_sphere = object_bounding_sphere
        self.sphere_tracing_iters = sphere_tracing_iters
        self.start_epsilon = initial_epsilon

    def forward(self,
                sdf,
                unique_camera_centers,
                points,  # (num_points, 3)
                ):
        assert sdf is not None, 'sdf cannot be None'
        assert unique_camera_centers is not None, 'camera_centers cannot be None'
        assert points is not None, 'points cannot be None'

        num_points, _ = points.shape
        ray_directions = unique_camera_centers.to(device).unsqueeze(0) - points.unsqueeze(
            1)  # (num_points, num_cams, 3)
        point_to_camera_distance = ray_directions.norm(dim=-1)  # (num_points, num_cams)
        unit_ray_directions = ray_directions / point_to_camera_distance.unsqueeze(-1)

        visibility_mask = self.sphere_tracing_for_visibility(sdf, points, unit_ray_directions,
                                                             unique_camera_centers)  # (num_points, num_cams)
        return visibility_mask

    def sphere_tracing_for_visibility(self, sdf, points, unit_ray_directions, unique_camera_centers):
        num_points, _ = points.shape
        num_cams, _ = unique_camera_centers.shape
        in_progress_mask = torch.ones(num_points, num_cams).to(device).bool()

        dist_from_sphere_intersection_to_points, mask_intersection = get_sphere_intersection_for_visibility(points,
                                                                                                            unit_ray_directions,
                                                                                                            r=self.object_bounding_sphere)

        current_points = points.unsqueeze(1) + self.start_epsilon * unit_ray_directions  # (num_points, num_cams, 3)
        current_sdf = sdf(current_points)  # (num_points, num_cams)

        in_progress_mask[
            current_sdf < 0] = 0  # if sdf<0, visibility ray tracing finished (=0)  # (num_points, num_cams)
        current_distance_to_start_points = self.start_epsilon + current_sdf
        in_progress_mask_old = in_progress_mask.detach().clone()
        iters = 0
        while True:
            current_points = points.unsqueeze(1) + current_distance_to_start_points.unsqueeze(
                -1) * unit_ray_directions  # (num_points, num_cams, 3)
            current_sdf[in_progress_mask] = sdf(current_points[in_progress_mask])  # (num_points, num_cams)

            # update visibility mask
            in_progress_mask[current_sdf < 5e-7] = 0
            current_distance_to_start_points[in_progress_mask] = current_distance_to_start_points[in_progress_mask] + \
                                                                 current_sdf[in_progress_mask]
            in_progress_mask[current_distance_to_start_points > dist_from_sphere_intersection_to_points] = 0

            if iters == self.sphere_tracing_iters or in_progress_mask_old.sum() == 0:
                break

            iters += 1
        visibility_mask = current_distance_to_start_points > dist_from_sphere_intersection_to_points
        visibility_mask = visibility_mask & mask_intersection
        return visibility_mask
