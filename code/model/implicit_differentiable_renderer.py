import numpy as np
import torch
import torch.nn as nn

from general_utils import device
from model.ray_tracing import RayTracing
from model.sample_network import SampleNetwork
from model.visibility_tracer import VisibilityTracing


class PositionalEncoding(object):
    def __init__(self, L=10):
        self.L = L

    def __call__(self, p):
        pi = 1.0
        p_transformed = torch.cat([torch.cat(
            [torch.sin((2 ** i) * pi * p),
             torch.cos((2 ** i) * pi * p)],
            dim=-1) for i in range(self.L)], dim=-1)
        return torch.cat([p, p_transformed], dim=-1)


class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,  # the radius of the initialized sphere
            skip_in=(),
            weight_norm=True,
            multires=0
    ):
        super().__init__()

        dims = [d_in] + dims + [d_out]

        # self.embed_fn = None
        if multires > 0:
            self.pe = PositionalEncoding(L=multires)
            input_dim = d_in * multires * 2 + d_in
            dims[0] = input_dim

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        self.softplus = nn.Softplus(beta=100)

    def forward(self, input):
        if self.pe is not None:
            input = self.pe(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)
        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients


class MVASNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.implicit_network = ImplicitNetwork(**conf.get_config('implicit_network'))
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')
        self.visible_ray_tracer = VisibilityTracing(**conf.get_config('visibility_ray_tracer'))

    def forward(self, model_input, dataset):
        view_directions, camera_centers, object_mask = \
            model_input["view_direction"], model_input["camera_center"], model_input["object_mask"]

        num_pixels, _ = view_directions.shape

        self.implicit_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[..., 0],
                                                                 cam_loc=camera_centers,
                                                                 object_mask=object_mask,
                                                                 ray_directions=view_directions)
        self.implicit_network.train()

        points = camera_centers + dists.reshape(num_pixels,
                                                1) * view_directions  # non differentiable w.r.t. network parameters
        sdf_output = self.implicit_network(points)[..., 0:1]

        if self.training:
            surface_mask = network_object_mask & object_mask  # has intersection and within mask
            surface_points = points[surface_mask]
            surface_normals = self.implicit_network.gradient(surface_points)

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).to(device)
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0)

            grad_theta = self.implicit_network.gradient(eikonal_points)
            with torch.no_grad():
                visibility_mask = self.visible_ray_tracer(sdf=lambda x: self.implicit_network(x)[..., 0],
                                                          unique_camera_centers=dataset.unique_camera_centers.to(
                                                              device),
                                                          points=points[surface_mask])  # (num_points, num_cams)

                num_vis_points = visibility_mask.shape[0]
                visibility_mask[torch.arange(num_vis_points), model_input["view_idx"][surface_mask].long()] = 1
                assert torch.all(visibility_mask.sum(-1) > 0)
                points_homo = torch.cat(
                    (points[surface_mask], torch.ones((surface_mask.sum(), 1), dtype=float, device=device)), -1).float()
                # project points onto all image planes
                # (num_cams, 3, 4) x (4, num_points)->  (num_cams, 3, num_points)
                pixel_coordinates_homo = torch.einsum("ijk, kp->ijp", dataset.projection_matrices,
                                                      points_homo.T).cpu().detach().numpy()
                pixel_coordinates_xx = (pixel_coordinates_homo[:, 0, :] / (
                            pixel_coordinates_homo[:, -1, :] + 1e-9)).T  # (num_points, num_cams)
                pixel_coordinates_yy = (pixel_coordinates_homo[:, 1, :] / (
                            pixel_coordinates_homo[:, -1, :] + 1e-9)).T  # (num_points, num_cams)

                # opencv convention to numpy axis convention
                #  (top left) ----> x    =>  (top left) ---> axis 1
                #    |                           |
                #    |                           |
                #    |                           |
                #    y                         axis 0
                index_axis0 = np.round(pixel_coordinates_yy)  # (num_points, num_cams)
                index_axis1 = np.round(pixel_coordinates_xx)  # (num_points, num_cams)
                # index_axis0 = np.round(dataset.img_height-pixel_coordinates_yy)  # (num_points, num_cams)
                # index_axis1 = np.round(dataset.img_width-pixel_coordinates_xx)  # (num_points, num_cams)
                index_axis0 = np.clip(index_axis0, int(0), int(dataset.img_height - 1)).astype(
                    np.uint)  # (num_points, num_cams)
                index_axis1 = np.clip(index_axis1, int(0), int(dataset.img_width - 1)).astype(np.uint)

                # (num_cams, img_height, img_width, 3) -> (num_points, num_cams, 3)
                num_cams = index_axis0.shape[1]
                tangent_vectors_all_view_list = []
                tangent_vectors_half_pi_all_view_list = []
                for cam_idx in range(num_cams):
                    azimuth_angles = dataset.azimuth_map_all_view[cam_idx,
                                                                  index_axis0[:, cam_idx],
                                                                  index_axis1[:, cam_idx]]  # (num_surface_points)
                    R_list = dataset.W2C_list[cam_idx]
                    r1 = R_list[0, :3]
                    r2 = R_list[1, :3]
                    tangent_vectors_all_view_list.append(
                        r1 * np.sin(azimuth_angles[:, None]) - r2 * np.cos(azimuth_angles[:, None]))
                    tangent_vectors_half_pi_all_view_list.append(r1 * np.sin(azimuth_angles[:, None] + np.pi / 2) -
                                                                 r2 * np.cos(azimuth_angles[:, None] + np.pi / 2))

                tangent_vectors_all_view = torch.stack(tangent_vectors_all_view_list, dim=1).to(
                    device)  # (num_points, num_cams, 3)
                tangent_vectors_half_pi_all_view = torch.stack(tangent_vectors_half_pi_all_view_list, dim=1).to(
                    device)  # (num_points, num_cams, 3)

            output = {
                'points': surface_points,
                "grad_normals": surface_normals,
                'sdf_output': sdf_output,  # signed distances of all points from all pixels  (for mask loss)
                'network_object_mask': network_object_mask,
                # the mask indicating which pixels hit the zero level set (for mask loss)
                'grad_theta': grad_theta,  # for eikonal loss
                "surface_mask": surface_mask,
                "tangent_vectors_all_view": tangent_vectors_all_view,
                "tangent_vectors_all_view_half_pi": tangent_vectors_half_pi_all_view,
                "visibility_mask": visibility_mask,
            }

        else:
            surface_mask = network_object_mask
            differentiable_surface_points = points[surface_mask]
            surface_normals = self.implicit_network.gradient(differentiable_surface_points)
            grad_theta = None

            output = {
                'points': differentiable_surface_points,
                "grad_normals": surface_normals,
                'sdf_output': sdf_output,  # signed distances of all points from all pixels  (for mask loss)
                'network_object_mask': network_object_mask,
                # the mask indicating which pixels hit the zero level set (for mask loss)
                'grad_theta': grad_theta,  # for eikonal loss
                "surface_mask": surface_mask
            }

        return output


if __name__ == "__main__":
    from pyhocon import ConfigFactory

    conf = ConfigFactory.parse_file("dtu_fixed_cameras.conf")
    neural_sdf = ImplicitNetwork(**conf.get_config('model').get_config('implicit_network'))
    print(neural_sdf)
    print(neural_sdf(torch.tensor([[0, 0, 0],
                                   [1, 1, 1]])))
