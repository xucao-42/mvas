import mcubes
import numpy as np
import torch
from tqdm.auto import tqdm

from general_utils import device


class QueryGrids:
    def __init__(self, grids_len=256, bbox_size=0.5) -> None:
        grids_dim, self.grid_size = np.linspace(-bbox_size, bbox_size, grids_len, retstep=True)
        query_points_xx, query_points_yy, query_points_zz = np.meshgrid(grids_dim, grids_dim, grids_dim, indexing="ij")
        query_points = np.stack((query_points_xx.flatten(), query_points_yy.flatten(), query_points_zz.flatten()), -1)
        self.points_torch = torch.from_numpy(query_points).float().to(device)
        self.num_grids = grids_len ** 3
        self.batch_size = 256 ** 2
        # self.batch_size = 4096
        self.batch_num = int(grids_len ** 3 / self.batch_size) + 1
        self.grids_len = grids_len

    def query_sdf(self, neural_sdf, fpath, scale=1, offset=0) -> None:
        neural_sdf.eval()
        grids_sdf_list = []
        pbar = tqdm(range(self.batch_num))
        for query_batch_idx in pbar:
            start_idx = query_batch_idx * self.batch_size
            end_idx = (query_batch_idx + 1) * self.batch_size
            batch_points = self.points_torch[start_idx:end_idx]
            grids_sdf_list.append(neural_sdf(batch_points).detach().cpu().numpy())
            pbar.set_description(f"Querying signed distance on {self.grids_len}^3 grids...")
        grids = np.concatenate(grids_sdf_list, 0).reshape(self.grids_len, self.grids_len, self.grids_len)
        vertices, triangles = mcubes.marching_cubes(grids, 0)
        triangles[:, 1], triangles[:, 2] = triangles[:, 2].copy(), triangles[:, 1].copy()
        vertices = ((vertices - (self.grids_len - 1) / 2) * self.grid_size) * scale + offset
        mcubes.export_obj(vertices, triangles, fpath)
        neural_sdf.train()
