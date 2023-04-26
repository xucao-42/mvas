import torch
from torch import nn
from torch.nn import functional as F


class MVASLoss(nn.Module):
    def __init__(self,
                 TSC_weight,
                 eikonal_weight,
                 silhouette_weight,
                 alpha,
                 normalize_normal,
                 use_half_pi_TSC_loss=False):
        super().__init__()
        self.TSC_weight = TSC_weight
        self.eikonal_weight = eikonal_weight
        self.silhouette_weight = silhouette_weight
        self.alpha = alpha
        self.normalize_normal = normalize_normal
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.l2_loss = nn.MSELoss(reduction='sum')
        self.use_half_pi_TSC_loss = use_half_pi_TSC_loss

    def get_tangent_space_consistency_loss(self, normals, tangents_all_view, visibility_mask, object_mask):
        if self.normalize_normal:
            normal_norm = torch.norm(normals, dim=-1, keepdim=True)
            normals = normals / normal_norm

        not_nan_mask = ~torch.isnan(tangents_all_view.sum(-1))
        visibility_mask = visibility_mask & not_nan_mask

        tangents_all_view[torch.isnan(tangents_all_view)] = 1
        num_visible_views = visibility_mask.sum(-1)
        # compute the tangent space consistency loss for each surface point in each view
        loss_per_point_per_view = visibility_mask * ((normals.unsqueeze(1) * tangents_all_view).sum(-1)) ** 2
        # only consider surface points that are visible in at least one view
        visible_view_mask = num_visible_views > 0
        # sum over all views for each surface point
        loss_per_point = loss_per_point_per_view[visible_view_mask].sum(-1) / num_visible_views[visible_view_mask]
        # sum over all surface points
        return loss_per_point.sum() / float(object_mask.shape[0])


    def get_half_pi_tangent_space_consistency_loss(self, normals, tangents_all_view, tangents_all_view_pi2,
                                                   visibility_mask, object_mask):
        if self.normalize_normal:
            normal_norm = torch.norm(normals, dim=-1, keepdim=True)
            normals = normals / normal_norm
        not_nan_mask = ~torch.isnan(tangents_all_view.sum(-1))
        visibility_mask = visibility_mask & not_nan_mask
        tangents_all_view[torch.isnan(tangents_all_view)] = 1  # (num_surface_points, num_views, 3)
        tangents_all_view_pi2[torch.isnan(tangents_all_view_pi2)] = 1
        num_visible_views = visibility_mask.sum(-1)

        # compute the tangent space consistency loss for each surface point in each view, considering both possibilities
        loss_1 = visibility_mask * (((normals.unsqueeze(1) * tangents_all_view).sum(-1)) ** 2)
        loss_2 = visibility_mask * (((normals.unsqueeze(1) * tangents_all_view_pi2).sum(-1)) ** 2)

        visible_view_mask = num_visible_views > 0
        # sum over all views for each surface point
        loss = (loss_1 * loss_2)[visible_view_mask].sum(-1) / num_visible_views[visible_view_mask]
        # sum over all surface points
        return loss.sum() / float(object_mask.shape[0])

    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_silhouette_loss(self, sdf_output, network_object_mask, object_mask):
        mask = torch.logical_xor(network_object_mask,
                                 object_mask)  # object mask=0 or network object mask = 0 (no ray-surface intersection)
        if mask.sum() == 0:
            print("mask loss is 0!")
            return torch.tensor(0.0).cuda().float()
        sdf_pred = -self.alpha * sdf_output[mask]
        gt = object_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(), gt.squeeze(),
                                                                          reduction='sum') / float(object_mask.shape[0])
        return mask_loss

    def forward(self, model_outputs, model_inputs):
        network_object_mask = model_outputs['network_object_mask']  # ray-surface intersection
        object_mask = model_inputs['object_mask']  # input mask

        if self.TSC_weight != 0:
            if self.use_half_pi_TSC_loss:
                TSC_loss = self.get_half_pi_tangent_space_consistency_loss(model_outputs["grad_normals"],
                                                                           model_outputs["tangent_vectors_all_view"],
                                                                           model_outputs[
                                                                               "tangent_vectors_all_view_half_pi"],
                                                                           model_outputs["visibility_mask"],
                                                                           object_mask)
            else:
                TSC_loss = self.get_tangent_space_consistency_loss(model_outputs["grad_normals"],
                                                                   model_outputs["tangent_vectors_all_view"],
                                                                   model_outputs["visibility_mask"],
                                                                   object_mask)

        silhouette_loss = self.get_silhouette_loss(model_outputs['sdf_output'], network_object_mask, object_mask)
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])

        loss = self.eikonal_weight * eikonal_loss
        loss_dict = {'eikonal_loss': eikonal_loss}

        if self.TSC_weight != 0:
            loss += self.TSC_weight * TSC_loss
            loss_dict["TSC_loss"] = TSC_loss

        if self.silhouette_weight != 0:
            loss += self.silhouette_weight * silhouette_loss
            loss_dict["silhouette_loss"] = silhouette_loss

        loss_dict["loss"] = loss

        return loss_dict
