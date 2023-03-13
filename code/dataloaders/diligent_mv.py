import cv2
import json
import os

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from camera_normalization import normalize_camera
from general_utils import device, boundary_expansion_mask


class DiligentMVDataloader(Dataset):
    def __init__(self,
                 data_dir,
                 obj_name,
                 input_normal_method="SDPS",
                 downscale=None,  # not used for diligent dataset
                 exclude_views=[],
                 debug_mode=False
                 ):
        self.data_dir = os.path.join(data_dir, input_normal_method, obj_name)
        calib_info = loadmat(os.path.join(self.data_dir, "Calib_Results.mat"))
        json_open = open(os.path.join(self.data_dir, "params.json"), "r")
        self.K = np.array(json.load(json_open)["K"])

        self.num_views = 20 - len(exclude_views)
        self.W2C_list = []
        self.C2W_list = []
        self.P_list = []
        tangent_list = []
        self.camera_center_all_view = []

        azimuth_list = []
        mask_vectorized_list = []
        view_direction_list = []  # in world coordinates
        camera_center_repeated_list = []  # in world coordinates
        mask_list = []
        # for single view evaluation
        num_pixel_in_each_view = []
        azimuth_map_all_view = []
        unique_camera_center_list = []
        # =======================================normalize camera center=================================================
        R_list = []
        t_list = []
        view_idx_list = []
        for view_idx in range(1, 21):
            if view_idx in exclude_views:
                continue
            R = calib_info[f"Rc_{view_idx}"]
            t = calib_info[f"Tc_{view_idx}"]
            R_list.append(R)
            t_list.append(t)
        self.normalized_coordinate_center, self.normalized_coordinate_scale = normalize_camera(R_list, t_list)
        print(f"camera centers are shifted by {self.normalized_coordinate_center} "
              f"and scaled by {self.normalized_coordinate_scale}!")

        # ==============================================================================================================
        view_count = -1
        for view_idx in range(1, 21):
            view_count += 1
            if view_idx in exclude_views:
                view_count -= 1
                print(f"View {view_idx} is excluded!")
                continue
            else:
                print(f"Processing view {view_idx}...")
            # ==========================================load masks======================================================
            view_mask = cv2.imread(os.path.join(self.data_dir, "input_azimuth_maps", f"view_{view_idx}.png"),
                                   -1)[..., -1].astype(bool)
            mask_list.append(view_mask)

            view_mask_padded = view_mask.copy()

            expanded_mask = boundary_expansion_mask(view_mask_padded)
            for _ in range(30):
                expanded_mask = boundary_expansion_mask(expanded_mask)

            mask_vectorized_list.append(view_mask_padded[expanded_mask])
            num_pixel_in_each_view.append(np.sum(expanded_mask))
            view_idx_list.append(np.full(shape=(np.sum(expanded_mask)), fill_value=view_count))
            # ==========================================load camera parameters==========================================
            R = calib_info[f"Rc_{view_idx}"]
            t = calib_info[f"Tc_{view_idx}"]

            W2C = np.zeros((4, 4), float)
            W2C[:3, :3] = R
            W2C[:3, 3] = np.squeeze(t)
            W2C[-1, -1] = 1
            self.W2C_list.append(W2C)

            C2W = np.linalg.inv(W2C)
            self.C2W_list.append(C2W)

            W2C_scaled = np.zeros((4, 4), float)
            W2C_scaled[:3, :3] = self.normalized_coordinate_scale * R
            W2C_scaled[:3, 3] = np.squeeze(t) + R @ self.normalized_coordinate_center
            W2C_scaled[-1, -1] = 1
            P = self.K[:3, :3] @ W2C_scaled[
                                 :3]  # shape: (3, 4), project a 3D point in the world coordinate onto the pixel coordinate
            self.P_list.append(P)

            camera_center = - R.T @ t  # in world coordinate
            # shift then scale
            camera_center_scaled = (camera_center - self.normalized_coordinate_center[:,
                                                    None]) / self.normalized_coordinate_scale
            unique_camera_center_list.append(camera_center_scaled)
            camera_center_repeated_list.append(np.tile(camera_center_scaled.T, (np.sum(expanded_mask), 1)))
            self.camera_center_all_view.append(
                torch.from_numpy(np.tile(camera_center_scaled.T, (np.sum(expanded_mask), 1))).float())
            # ==========================================load azimuth====================================================
            azimuth_map = cv2.imread(os.path.join(self.data_dir, "input_azimuth_maps", f"view_{view_idx}.png"), -1)[..., 0]
            azimuth_map = np.pi * azimuth_map / 65535
            a = azimuth_map[expanded_mask]
            azimuth_list.append(a)
            azimuth_map_all_view.append(azimuth_map)

            r1 = R[0]
            r2 = R[1]
            tangents = r1[None, :] * np.sin(a)[:, None] - r2[None, :] * np.cos(a)[:, None]
            tangent_list.append(tangents)
            # ========================================compute ray direction=============================================
            H, W = view_mask.shape
            self.img_height = H
            self.img_width = W
            xx, yy = np.meshgrid(range(W), range(H))  # xx right yy bottom

            # view direction is invariant to camera center shift and scaling
            uv_homo = np.stack((xx[expanded_mask], yy[expanded_mask], np.ones_like(xx[expanded_mask])), axis=-1)
            view_direction = (C2W[:3, :3] @ np.linalg.inv(self.K[:3, :3]) @ uv_homo.T).T
            view_direction = view_direction / np.linalg.norm(view_direction, axis=-1, keepdims=True)
            view_direction_list.append(view_direction)

        if debug_mode:
            from visual_hull_check import visual_hull_creation
            visual_hull_creation(mask_list, self.P_list)

        # training data
        self.mask_vectorized = torch.from_numpy(np.concatenate(mask_vectorized_list, 0)).bool()
        self.view_direction = torch.from_numpy(np.concatenate(view_direction_list, 0)).float()
        self.camera_center = torch.from_numpy(np.concatenate(camera_center_repeated_list, 0)).float()
        self.tangents = torch.from_numpy(np.concatenate(tangent_list, 0)).float()
        self.view_idx = torch.from_numpy(np.concatenate(view_idx_list)).int()

        # self.azimuth = torch.from_numpy(np.concatenate(azimuth_list, 0)).float()
        # self.num_pixel_in_each_view_cumsum = np.cumsum(np.array(num_pixel_in_each_view))

        # for reprojection
        self.unique_camera_centers = torch.from_numpy(
            np.squeeze(np.array(unique_camera_center_list))).float()  # (num_cams, 3)
        self.projection_matrices = torch.from_numpy(np.array(self.P_list)).float().to(device)  # (num_cams, 3, 4)
        self.azimuth_map_all_view = np.array(azimuth_map_all_view)  # (num_cams, img_height, img_widht)
        self.W2C_list = torch.from_numpy(np.array(self.W2C_list)).float()

    def map_img_to_point_clouds(self, mask, K):
        #    z  (points from the camera to the scene)
        #   /
        #  /
        # o (top left corner) ---x
        # |
        # |
        # y

        H, W = mask.shape
        xx, yy = np.meshgrid(range(W), range(H))
        u = np.zeros((H, W, 3))
        u[..., 0] = xx
        u[..., 1] = yy
        u[..., 2] = 1
        u = u[mask].T  # 3 x m
        p_tilde = (np.linalg.inv(K) @ u).T  # m x 3
        return p_tilde

    def camera_to_object(self, n):
        no = n.copy()
        no[..., 2] = -no[..., 2]
        no[..., 1] = -no[..., 1]
        return no

    def __len__(self):
        return len(self.mask_vectorized)

    def __getitem__(self, idx):
        model_input = {
            "camera_center": self.camera_center[idx],
            "view_direction": self.view_direction[idx],
            "object_mask": self.mask_vectorized[idx],
            "tangents": self.tangents[idx],
            "view_idx": self.view_idx[idx],
        }
        return model_input

    # def __getitem__(self, idx):
    # azimuth = self.azimuth[idx]
    # view_idx = np.argmax(self.num_pixel_in_each_view_cumsum > idx)
    # W2C = self.W2C_list[view_idx]
    # R = W2C[:3, :3]
    # r1 = R[0]
    # r2 = R[1]
    # tangent_vec = r1 * np.sin(azimuth) - r2 * np.cos(azimuth)

    # model_input = {
    #     "camera_center": self.unique_camera_centers[view_idx],
    #     "view_direction": self.view_direction[idx],
    #     "object_mask": self.mask_vectorized[idx],
    #     "tangents": tangent_vec,
    #     "view_idx": torch.tensor(view_idx).int()
    # }
    # return model_input


# class DiLiGenT_MV_single_view_Loader(Dataset):
#     def __init__(self, obj_name, view_idx, normalized_center, normalized_scale):
#         diligent_dir = os.path.join("DiLiGenT-MV", "mvpmsData", f"{obj_name}PNG")
#
#         calib_info = loadmat(os.path.join(diligent_dir, "Calib_Results.mat"))
#         self.K = calib_info["KK"]
#         self.mesh_path = os.path.join(diligent_dir, "mesh_Gt.ply")
#         mesh = o3d.io.read_triangle_mesh(self.mesh_path)
#
#         # pv.PolyData(np.asarray(gt_points.points)).plot()
#         mesh_points = pv.read(self.mesh_path).points
#         # self.mesh_center = np.mean(mesh_points, 0, keepdims=True)
#         # self.scale = np.max(np.sqrt(np.sum((mesh_points-self.mesh_center) ** 2, -1)))
#         # normalize camera center
#         # A_camera_normalize = 0
#         # b_camera_normalize = 0
#         # camera_center_list = []
#         # for idx in range(1, 21):
#         #     R = calib_info[f"Rc_{idx}"]
#         #     t = calib_info[f"Tc_{idx}"]
#         #     camera_center = - R.T @ t  # in world coordinate
#         #     camera_center_list.append(camera_center)
#         #     vi = R[2][:, None]  # the camera's principal axis in the world coordinates
#         #     Vi = vi @ vi.T
#         #     A_camera_normalize += np.eye(3) - Vi
#         #     b_camera_normalize += camera_center.T @ (np.eye(3) - Vi)
#         # self.normalized_coordinate_center = np.linalg.lstsq(A_camera_normalize, np.squeeze(b_camera_normalize))[0]
#         # camera_center_dist_list = [np.sqrt(np.sum((np.squeeze(c)-self.normalized_coordinate_center)**2))
#         #                            for c in camera_center_list]
#
#         # self.normalized_coordinate_scale = np.max(camera_center_dist_list) / 10
#         self.normalized_coordinate_center = normalized_center
#         self.normalized_coordinate_scale = normalized_scale
#
#         self.mask_all_view = []
#         self.input_azimuth_angle_map_vis_all_view = []
#         self.tangents_map_list = []
#         self.tangents_map_list_pi2 = []
#         camera_center_repeated_list = []
#
#         print(f"Processing view {view_idx}...")
#
#         R = calib_info[f"Rc_{view_idx}"]
#         t = calib_info[f"Tc_{view_idx}"]
#
#         r1 = R[0]
#         r2 = R[1]
#         W2C = np.zeros((4, 4), float)
#         W2C[:3, :3] = R
#         W2C[:3, 3] = np.squeeze(t)
#         W2C[-1, -1] = 1
#
#         C2W = np.linalg.inv(W2C)
#         self.W2C = W2C
#
#         P = self.K @ W2C[:3]  # shape: (3, 4), project a 3D point in the world coordinate onto the pixel coordinate
#
#         view_mask = cv2.imread(os.path.join(diligent_dir, f"view_{view_idx:02}", "mask.png"),
#                                cv2.IMREAD_GRAYSCALE).astype(bool)
#         self.num_pixels = np.sum(view_mask)
#         self.view_mask = view_mask
#
#         camera_center = - R.T @ t  # in world coordinate
#         camera_center_scaled = (camera_center - self.normalized_coordinate_center[:,
#                                                 None]) / self.normalized_coordinate_scale
#         camera_center_repeated = np.tile(camera_center_scaled.T, (np.sum(view_mask), 1))
#
#         normal_map = loadmat(os.path.join(diligent_dir, f"view_{view_idx:02}", "Normal_gt.mat"))[
#             "Normal_gt"]  # in normal coordinates
#         # normal_map = \
#         # loadmat(os.path.join("DiLiGenT-MV", "estNormal", f"{obj_name}PNG_Normal_TIP19Li_view{view_idx}.mat"))[
#         #     "Normal_est"]  # in normal coordinates
#
#         normal_map_camera = self.camera_to_object(normal_map)
#         azimuth_map_in_camera, azimuth_jet, _ = normal_to_azimuth(normal_map_camera, view_mask,
#                                                                   random_pi_flip=False)
#         self.input_azimuth_angle_map_vis_all_view.append(azimuth_jet)
#         a = azimuth_map_in_camera[view_mask]
#         tangents = r1[None, :] * np.sin(a)[:, None] - r2[None, :] * np.cos(a)[:, None]
#         tangents_pi2 = r1[None, :] * np.sin(a + np.pi / 2)[:, None] - r2[None, :] * np.cos(a + np.pi / 2)[:, None]
#
#         normal_map_world = (C2W[:3, :3] @ normal_map_camera[view_mask].T).T
#         self.normal_map_camera = normal_map_camera
#         self.normal_map_vis = (255. * (self.camera_to_object(normal_map_camera) + 1) / 2).astype(np.uint8)
#         # plt.imshow((self.normal_map_camera+1)/2)
#         # plt.show()
#
#         H, W = view_mask.shape
#         xx, yy = np.meshgrid(range(W), range(H))
#
#         uv_homo = np.stack((xx[view_mask], yy[view_mask], np.ones_like(xx[view_mask])), axis=-1)
#         view_direction = (C2W[:3, :3] @ np.linalg.inv(self.K[:3, :3]) @ uv_homo.T).T
#         view_direction = view_direction / np.linalg.norm(view_direction, axis=-1, keepdims=True)
#
#         self.mask_vectorized = torch.ones(self.num_pixels, dtype=bool)  # .to(device)
#         self.view_direction = torch.from_numpy(view_direction).float()  # .to(device)
#         self.camera_center = torch.from_numpy(camera_center_repeated).float()  # .to(device)
#         self.tangents = torch.from_numpy(tangents).float()
#         self.tangents_pi2 = torch.from_numpy(tangents_pi2).float()
#         self.normals = torch.from_numpy(normal_map_world).float()
#
#     def map_img_to_point_clouds(self, mask, K):
#         #    z  (points from the camera to the scene)
#         #   /
#         #  /
#         # o (top left corner) ---x
#         # |
#         # |
#         # y
#
#         H, W = mask.shape
#         xx, yy = np.meshgrid(range(W), range(H))
#         u = np.zeros((H, W, 3))
#         u[..., 0] = xx
#         u[..., 1] = yy
#         u[..., 2] = 1
#         u = u[mask].T  # 3 x m
#         p_tilde = (np.linalg.inv(K) @ u).T  # m x 3
#         return p_tilde
#
#     def camera_to_object(self, n):
#         no = n.copy()
#         no[..., 2] = -no[..., 2]
#         no[..., 1] = -no[..., 1]
#         return no
#
#     def __len__(self):
#         return self.num_pixels
#
#     def __getitem__(self, idx):
#         model_input = {
#             "camera_center": self.camera_center[idx],
#             "view_direction": self.view_direction[idx],
#             "object_mask": self.mask_vectorized[idx],
#             "tangents": self.tangents[idx],
#             "tangents_pi2": self.tangents_pi2[idx],
#             "normals": self.normals[idx]
#         }
#         return model_input


if __name__ == '__main__':
    # sanity check
    DiligentMVDataloader("buddha", debug_mode=True)
