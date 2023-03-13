import os.path
import sys

sys.path += [".", "..", "model", "utils"]

from model.implicit_differentiable_renderer import *
from torch.utils.data import DataLoader
from pyhocon import ConfigFactory
import shutil
from general_utils import device, get_class
from marching_cube_for_neural_sdf import QueryGrids
import time
from tqdm.auto import tqdm
from collections import OrderedDict

def main(conf_dir):
    exp_time = str(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())))

    conf = ConfigFactory.parse_file(conf_dir)
    torch.autograd.set_detect_anomaly(True)

    obj_list = conf.get_list("train.obj_list")
    print(obj_list)
    for obj_name in obj_list:
        batch_size = conf.get_int('train.batchsize')
        epoch_total = conf.get_int("train.epoch")

        downscale = conf.get_int("train.downscale")
        dataset = get_class(conf["train"]["dataset_class"])(obj_name=obj_name,
                                                            downscale=downscale,
                                                            **conf.get_config("dataset"))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        neural_sdf = get_class(conf["train"]["model_class"])(conf=conf.get_config('model'))
        neural_sdf.to(device)

        loss = get_class(conf["train"]["loss_class"])(**conf.get_config('loss'))

        exp_dir = os.path.join("../results", obj_name, f"exp_{exp_time}")
        os.makedirs(exp_dir, exist_ok=True)
        model_dir = os.path.join(exp_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        grid_res = conf.get_int("plot.grid_res")
        mesh_dir = os.path.join(exp_dir, f"surfaces_res_{grid_res}")
        os.makedirs(mesh_dir, exist_ok=True)
        loss_dir = os.path.join(exp_dir, "loss")
        os.makedirs(loss_dir, exist_ok=True)

        shutil.copyfile(conf_dir, os.path.join(exp_dir, "train.conf"))
        shutil.copytree("../code", os.path.join(exp_dir, "code"))

        np.save(os.path.join(exp_dir, "camera_normalization"),
                {"scale": dataset.normalized_coordinate_scale, "offset": dataset.normalized_coordinate_center})

        lr = conf.get_float('train.learning_rate')
        optimizer = torch.optim.Adam(neural_sdf.parameters(), lr=lr)
        total_batch = int(len(dataset) / batch_size) + 1

        query_grids = QueryGrids(grids_len=grid_res,
                                 bbox_size=conf.get_float("plot.bbox_size"))

        alpha_milestones = conf.get_list('train.alpha_milestones', default=[])
        alpha_factor = conf.get_float('train.alpha_factor', default=0.0)

        sched_milestones = conf.get_list('train.sched_milestones', default=[])
        sched_factor = conf.get_float('train.sched_factor', default=0.0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, sched_milestones, gamma=sched_factor)

        if loss.TSC_weight != 0:
            TSC_loss_list = []

        if loss.silhouette_weight != 0:
            silhouette_loss_list = []

        if loss.eikonal_weight != 0:
            eikonal_loss_list = []

        query_grids.query_sdf(neural_sdf.implicit_network,
                              os.path.join(mesh_dir, f"mesh_init.obj"),
                              dataset.normalized_coordinate_scale,
                              dataset.normalized_coordinate_center
                              )

        for epoch_idx in range(epoch_total):
            try:
                if epoch_idx+1 in conf["train"]["downscale_milestones"]:
                    del dataset, dataloader
                    # print(downscale)
                    downscale /= conf["train"]["downscale_factor"]
                    dataset = get_class(conf["train"]["dataset_class"])(obj_name=obj_name,
                                                                        downscale=downscale,
                                                                        **conf.get_config("dataset"))
                    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
            except:
                pass

            if epoch_idx == epoch_total - 1:
                grid_res_last = conf.get_int("plot.grid_res_last_epoch")
                mesh_dir = os.path.join(exp_dir, f"surfaces_res_{grid_res_last}")
                os.makedirs(mesh_dir, exist_ok=True)
                query_grids = QueryGrids(grids_len=grid_res_last,
                                         bbox_size=conf.get_float("plot.bbox_size"))
            torch.save(
                {"epoch": epoch_idx, "model_state_dict": neural_sdf.state_dict()},
                os.path.join(model_dir, str(epoch_idx) + ".pth"))
            torch.save(
                {"epoch": epoch_idx, "model_state_dict": neural_sdf.state_dict()},
                os.path.join(model_dir, "latest.pth"))

            if loss.TSC_weight != 0:
                np.save(os.path.join(loss_dir, "TSC_loss"), np.array(TSC_loss_list))
            if loss.silhouette_weight != 0:
                np.save(os.path.join(loss_dir, "silhouette_loss"), np.array(silhouette_loss_list))
            if loss.eikonal_weight != 0:
                np.save(os.path.join(loss_dir, "eikonal_loss"), np.array(eikonal_loss_list))

            if epoch_idx in alpha_milestones:
                loss.alpha = loss.alpha * alpha_factor

            pbar_batch = tqdm(enumerate(dataloader), total=len(dataloader))
            for batch_idx, model_input in pbar_batch:
                model_input["camera_center"] = model_input["camera_center"].to(device)
                model_input["view_direction"] = model_input["view_direction"].to(device)
                model_input["object_mask"] = model_input["object_mask"].to(device)
                model_input["tangents"] = model_input["tangents"].to(device)
                model_input["view_idx"] = model_input["view_idx"].to(device)
                if loss.use_half_pi_TSC_loss:
                    model_input["tangents_half_pi"] = model_input["tangents_half_pi"].to(device)

                model_output = neural_sdf(model_input, dataset)
                loss_output = loss(model_output, model_input)

                total_loss = loss_output["loss"]

                eikonal_loss = loss_output["eikonal_loss"]
                eikonal_loss_list.append(eikonal_loss.item())

                message_prefix = f"[Epoch: {epoch_idx + 1}/{epoch_total}]"
                message_loss = OrderedDict(total_loss=f"{total_loss:.3e}")
                if loss.TSC_weight != 0:
                    TSC_loss = loss_output["TSC_loss"]
                    TSC_loss_list.append(TSC_loss.item())
                    message_loss["TSC"] = f"{TSC_loss:.3e}"
                if loss.silhouette_weight != 0:
                    silhouette_loss = loss_output["silhouette_loss"]
                    silhouette_loss_list.append(silhouette_loss.item())
                    message_loss["silhouette"] = f"{silhouette_loss:.3e}"

                message_loss["eikonal"] = f"{eikonal_loss:.3e}"

                pbar_batch.set_description(message_prefix)
                pbar_batch.set_postfix(ordered_dict=message_loss)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            scheduler.step()
            query_grids.query_sdf(neural_sdf.implicit_network,
                                  os.path.join(mesh_dir, f"mesh_epoch_{epoch_idx + 1}.obj"),
                                  dataset.normalized_coordinate_scale,
                                  dataset.normalized_coordinate_center
                                  )


if __name__ == "__main__":
    import argparse
    import os


    def file_path(string):
        if os.path.isfile(string):
            return string
        else:
            raise FileNotFoundError(string)


    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=file_path)
    arg = parser.parse_args()

    main(arg.config)
