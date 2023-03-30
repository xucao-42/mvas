import os

import cv2
import numpy as np
import torch
from skimage.transform import rescale
from torch.utils.data import Dataset

import colmap_read_model as read_model
from camera_normalization import normalize_camera
from general_utils import device, boundary_expansion_mask


class SympsDataloader(Dataset):
    def __init__(self,
                 data_dir,
                 obj_name,
                 downscale=None,
                 camera_object_ratio=3,
                 exclude_views=[],
                 debug_mode=False):
        self.data_dir = os.path.join(data_dir, obj_name)
        self.downscale = downscale
        self.camera_object_ratio = camera_object_ratio

        mask_dir = os.path.join(self.data_dir, "mask")
        self.exclude_views = exclude_views
        # ==============================================load camera parameters =========================================
        self.poses_from_colmap()
        # ==============================================================================================================
        mask_vectorized_list = []
        azimuth_list = []
        view_direction_list = []  # in world coordinates
        num_pixel_in_each_view = []
        mask_list = []
        azimuth_map_all_view = []
        view_idx_list = []
        tangents_list = []
        camera_center_repeated_list = []
        self.camera_center_all_view = []

        view_count = -1
        for view_idx in self.image_id_list:
            view_count += 1
            print(f"Processing {view_idx}...")
            # ========================================load masks=========================================================
            view_mask = cv2.imread(os.path.join(mask_dir, f"{view_idx}.png"), -1).astype(bool)
            if downscale is not None:
                view_mask = rescale(view_mask, 1. / downscale, anti_aliasing=False, channel_axis=None)
            mask_list.append(view_mask)

            expanded_mask = boundary_expansion_mask(view_mask)
            for _ in range(100):
                expanded_mask = boundary_expansion_mask(expanded_mask)

            mask_vectorized_list.append(view_mask[expanded_mask])
            num_pixel_in_each_view.append(np.sum(expanded_mask))
            view_idx_list.append(np.full(shape=(np.sum(expanded_mask)), fill_value=view_count))

            # ====================================load azimuth angle maps================================================
            azimuth_map = cv2.imread(os.path.join(self.data_dir, "input_azimuth_maps", f"{view_idx}.png"), -1)
            azimuth_map = np.pi * azimuth_map / 65535
            if self.downscale is not None:
                azimuth_map = cv2.resize(azimuth_map,
                                         dsize=None, fx=1 / downscale, fy=1 / downscale,
                                         interpolation=cv2.INTER_NEAREST)

            a = azimuth_map[expanded_mask]

            azimuth_list.append(a)
            azimuth_map_all_view.append(azimuth_map)

            W2C = self.W2C_list[view_count]
            C2W = np.linalg.inv(W2C)
            R = W2C[:3, :3]
            t = W2C[:3, 3]

            r1 = R[0]
            r2 = R[1]
            tangents = r1[None, :] * np.sin(a)[:, None] - r2[None, :] * np.cos(a)[:, None]
            tangents_list.append(tangents)

            camera_center = - R.T @ t  # in world coordinate
            camera_center_scaled = (camera_center - self.normalized_coordinate_center) / self.normalized_coordinate_scale
            camera_center_repeated_list.append(np.tile(camera_center_scaled.T, (np.sum(expanded_mask), 1)))
            self.camera_center_all_view.append(
                torch.from_numpy(np.tile(camera_center_scaled.T, (np.sum(expanded_mask), 1))).float())
            # ======================================compute ray directions===============================================
            # xx -> right yy -> bottom
            xx, yy = np.meshgrid(range(self.img_width), range(self.img_height))

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
        self.tangents = torch.from_numpy(np.concatenate(tangents_list, 0)).float()
        self.view_idx = torch.from_numpy(np.concatenate(view_idx_list)).int()

        # self.azimuth = torch.from_numpy(np.concatenate(azimuth_list, 0)).float()
        # self.num_pixel_in_each_view_cumsum = np.cumsum(np.array(num_pixel_in_each_view))

        # for reprojection
        self.unique_camera_centers = torch.from_numpy(
            np.squeeze(np.array(self.unique_camera_center_list))).float()  # (num_cams, 3)
        self.projection_matrices = torch.from_numpy(np.array(self.P_list)).float().to(device)  # (num_cams, 3, 4)
        self.azimuth_map_all_view = np.array(azimuth_map_all_view)  # (num_cams, img_height, img_widht)
        self.W2C_list = torch.from_numpy(np.array(self.W2C_list)).float()

    def poses_from_colmap(self):
        # Adapted from llff pose_utils.py
        realdir = self.data_dir
        camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
        camdata = read_model.read_cameras_binary(camerasfile)

        list_of_keys = list(camdata.keys())
        cam = camdata[list_of_keys[0]]

        imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
        imdata = read_model.read_images_binary(imagesfile)
        imdata = sorted(imdata.values(), key=lambda x: int(x.name.split(".")[0][-4:]))
        self.image_id_list = [k.name.split(".")[0] for k in imdata]
        self.image_id_list = list((set(self.image_id_list) - set(self.exclude_views)))
        self.image_id_list.sort(key=lambda x: int(x.split("_")[-1]))

        self.num_views = len(imdata)
        self.img_height = int(cam.height)
        self.img_width = int(cam.width)
        f, x0, y0 = cam.params[0], cam.params[1], cam.params[2]
        if self.downscale is not None:
            x0 /= self.downscale
            y0 /= self.downscale
            f /= self.downscale

            self.img_height = int(np.round(cam.height / self.downscale))
            self.img_width = int(np.round(cam.width / self.downscale))

        self.K = np.array([[f, 0., x0],
                           [0., f, y0],
                           [0., 0., 1.]])

        self.W2C_list = []
        self.C2W_list = []
        self.W2C_scaled_list = []
        self.C2W_scaled_list = []
        self.P_list = []
        self.unique_camera_center_list = []

        R_list = []
        t_list = []
        for view_idx, im in enumerate(imdata):
            if im.name.split(".")[0] in self.exclude_views:
                continue
            R = im.qvec2rotmat()
            t = im.tvec.reshape([3, 1])
            R_list.append(R)
            t_list.append(t)
        self.normalized_coordinate_center, self.normalized_coordinate_scale = \
            normalize_camera(R_list, t_list, camera2object_ratio=self.camera_object_ratio)
        print(f"camera centers are shifted by {self.normalized_coordinate_center} "
              f"and scaled by {self.normalized_coordinate_scale}!")

        for view_idx, im in enumerate(imdata):
            if im.name.split(".")[0] in self.exclude_views:
                continue
            R = im.qvec2rotmat()
            t = im.tvec.reshape([3, 1])
            camera_center = - R.T @ t  # in world coordinate
            # shift then scale
            camera_center_scaled = (camera_center - self.normalized_coordinate_center[:,
                                                    None]) / self.normalized_coordinate_scale
            self.unique_camera_center_list.append(camera_center_scaled)

            W2C = np.zeros((4, 4))
            W2C[3, 3] = 1
            W2C[:3, :3] = R
            W2C[:3, [3]] = t
            self.W2C_list.append(W2C)
            C2W = np.linalg.inv(W2C)
            self.C2W_list.append(C2W)

            W2C_scaled = np.zeros((4, 4), float)
            W2C_scaled[:3, :3] = self.normalized_coordinate_scale * R
            W2C_scaled[:3, 3] = np.squeeze(t) + R @ self.normalized_coordinate_center
            W2C_scaled[-1, -1] = 1

            C2W_scaled = np.linalg.inv(W2C_scaled)
            self.W2C_scaled_list.append(W2C_scaled)
            self.C2W_scaled_list.append(C2W_scaled)
            P = self.K[:3, :3] @ W2C_scaled[
                                 :3]  # shape: (3, 4), project a 3D point in the world coordinate onto the pixel coordinate
            self.P_list.append(P)

    # def __getitem__(self, idx):
    #     azimuth = self.azimuth[idx]
    #     view_idx = np.argmax(self.num_pixel_in_each_view_cumsum>idx)
    #     W2C = self.W2C_list[view_idx]
    #     R = W2C[:3, :3]
    #     r1 = R[0]
    #     r2 = R[1]
    #     tangent_vec = r1 * np.sin(azimuth) - r2 * np.cos(azimuth)
    #     camera_center = self.unique_camera_centers[view_idx]
    #
    #     model_input = {
    #         "camera_center": camera_center,
    #         "view_direction": self.view_direction[idx],
    #         "object_mask": self.mask_vectorized[idx],
    #         "tangents": tangent_vec,
    #         "view_idx": torch.tensor(view_idx).int()
    #     }
    #     return model_input

    def __getitem__(self, idx):
        model_input = {
            "camera_center": self.camera_center[idx],
            "view_direction": self.view_direction[idx],
            "object_mask": self.mask_vectorized[idx],
            "tangents": self.tangents[idx],
            "view_idx": self.view_idx[idx],
        }
        return model_input

    def __len__(self):
        return len(self.mask_vectorized)


class symps_single_view_Loader(Dataset):
    def __init__(self, dataset, view_idx, eval_downscale=4):
        """
        If frustum scale is None, not preparing the pcds for multi-view visualization.

        """

        # self.mask_all_view = []
        # self.input_azimuth_angle_map_vis_all_view = []
        # self.tangents_map_list = []
        # self.tangents_map_list_pi2 = []
        # camera_center_repeated_list = []

        print(f"Processing view {view_idx}...")
        K_scale = dataset.downscale / eval_downscale
        C2W = dataset.C2W_list[view_idx][:3, :3]

        # view_mask = dataset.mask_all_view[view_idx]
        image_id = dataset.image_id_list[view_idx]
        # self.view_mask = rescale(view_mask, K_scale, anti_aliasing=False, multichannel=False)
        view_mask = cv2.imread(os.path.join(os.path.join(dataset.data_dir, "mask_cao"), f"{image_id}.png"), -1)[
                        ..., -1] > 240
        self.view_mask = rescale(view_mask, 1. / (eval_downscale / 2), anti_aliasing=False, multichannel=False)
        self.num_pixels = np.sum(self.view_mask)
        self.R_W2C = dataset.W2C_list[view_idx][:3, :3]

        camera_center = dataset.unique_camera_center_list[view_idx]  # in world coordinate
        # camera_center /= dataset.normalized_coordinate_scale
        # camera_center = (camera_center - dataset.normalized_coordinate_center[:, None]) / dataset.normalized_coordinate_scale
        camera_center_repeated = np.tile(camera_center.T, (np.sum(self.view_mask), 1))

        self.H, self.W = self.view_mask.shape
        xx, yy = np.meshgrid(range(self.W), range(self.H))

        K = dataset.K.copy()
        K[0, 0] *= K_scale
        K[1, 1] *= K_scale
        K[0, 2] *= K_scale
        K[1, 2] *= K_scale

        uv_homo = np.stack((xx[self.view_mask], yy[self.view_mask], np.ones_like(xx[self.view_mask])), axis=-1)
        view_direction = (C2W[:3, :3] @ np.linalg.inv(K) @ uv_homo.T).T
        view_direction = view_direction / np.linalg.norm(view_direction, axis=-1, keepdims=True)

        self.mask_vectorized = torch.ones(self.num_pixels, dtype=bool)  # .to(device)
        self.view_direction = torch.from_numpy(view_direction).float()  # .to(device)
        self.camera_center = torch.from_numpy(camera_center_repeated).float()  # .to(device)
        pass

    def __len__(self):
        return self.num_pixels

    def __getitem__(self, idx):
        model_input = {
            "camera_center": self.camera_center[idx],
            "view_direction": self.view_direction[idx],
            "object_mask": self.mask_vectorized[idx],
        }
        return model_input


if __name__ == '__main__':
    dataset = SympsDataloader(data_dir="../../data/SymPS",
                             obj_name="gargoyle",
                             downscale=8,
                             camera_object_ratio=3,
                             debug_mode=False,
                            exclude_views=["IMG_0466", "IMG_0476"])




