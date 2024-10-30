import torch
from .strategy import Strategy
from pcdet.models import load_data_to_gpu
from pcdet.datasets import build_active_dataloader
import torch.nn.functional as F
from torch.distributions import Categorical
import tqdm
import numpy as np
import wandb
import time
import scipy
from sklearn.cluster import kmeans_plusplus, KMeans, MeanShift, Birch
from sklearn.mixture import GaussianMixture
from scipy.stats import uniform
from sklearn.neighbors import KernelDensity
from scipy.cluster.vq import vq
from typing import Dict, List
# from apricot import MaxCoverageSelection
from apricot import CustomSelection, FeatureBasedSelection


class STONE_Sampling_active(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):

        super(STONE_Sampling_active, self).__init__(model, labelled_loader,
                                                                               unlabelled_loader, rank,
                                                                               active_label_dir, cfg)


    def enable_dropout(self, model):
        """ Function to enable the dropout layers during test-time """
        i = 0
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                i += 1
                m.train()
        print('**found and enabled {} Dropout layers for random sampling**'.format(i))

    def query(self, leave_pbar=True, cur_epoch=None):
        # counting label
        val_dataloader_iter_labelled = iter(self.labelled_loader)
        val_loader_labelled = self.labelled_loader
        total_it_each_epoch_labelled = len(self.labelled_loader)

        # feed forward the model
        if self.rank == 0:
            pbar = tqdm.tqdm(total=total_it_each_epoch_labelled, leave=leave_pbar,
                             desc='evaluating_unlabelled_set_epoch_%d' % cur_epoch, dynamic_ncols=True)
        self.model.eval()
        self.enable_dropout(self.model)
        num_class = len(self.labelled_loader.dataset.class_names)

        total_num_car = torch.zeros(1).to('cuda:0')
        total_num_cyclist = torch.zeros(1).to('cuda:0')
        total_num_pedestrain = torch.zeros(1).to('cuda:0')
        for cur_it in range(total_it_each_epoch_labelled):
            try:
                unlabelled_batch = next(val_dataloader_iter_labelled)
            except StopIteration:
                unlabelled_dataloader_iter = iter(val_loader_labelled)
                unlabelled_batch = next(unlabelled_dataloader_iter)
            with torch.no_grad():
                load_data_to_gpu(unlabelled_batch)
                pred_dicts, _ = self.model(unlabelled_batch)
                for batch_inx in range(len(pred_dicts)):
                    # save the meta information and project it to the wandb dashboard
                    value, counts = torch.unique(pred_dicts[batch_inx]['pred_labels'], return_counts=True)
                    if len(value) == 0:
                        entropy = 0
                    else:
                        # calculates the shannon entropy of the predicted labels of bounding boxes
                        unique_proportions = torch.ones(num_class).cuda()
                        unique_proportions[value - 1] = counts.float()
                        entropy = Categorical(probs=unique_proportions / sum(counts)).entropy()
                    total_num_car += pred_dicts[batch_inx]['num_bbox']['Car']
                    total_num_pedestrain += pred_dicts[batch_inx]['num_bbox']['Pedestrian']
                    total_num_cyclist += pred_dicts[batch_inx]['num_bbox']['Cyclist']


        select_dic = {}

        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
        val_dataloader_iter = iter(self.unlabelled_loader)
        val_loader = self.unlabelled_loader
        total_it_each_epoch = len(self.unlabelled_loader)

        # feed forward the model
        if self.rank == 0:
            pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                             desc='evaluating_unlabelled_set_epoch_%d' % cur_epoch, dynamic_ncols=True)
        self.model.eval()
        self.enable_dropout(self.model)
        num_class = len(self.labelled_loader.dataset.class_names)
        check_value = []
        cls_results = {}
        reg_results = {}
        density_list = {}
        label_list = {}
        predicted_label_dict = {}

        for cur_it in range(total_it_each_epoch):
            try:
                unlabelled_batch = next(val_dataloader_iter)
            except StopIteration:
                unlabelled_dataloader_iter = iter(val_loader)
                unlabelled_batch = next(unlabelled_dataloader_iter)
            with torch.no_grad():
                load_data_to_gpu(unlabelled_batch)
                pred_dicts, _ = self.model(unlabelled_batch)
                for batch_inx in range(len(pred_dicts)):
                    # save the meta information and project it to the wandb dashboard
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])

                    value, counts = torch.unique(pred_dicts[batch_inx]['pred_labels'], return_counts=True)
                    if len(value) == 0:
                        entropy = 0
                    else:
                        # calculates the shannon entropy of the predicted labels of bounding boxes
                        unique_proportions = torch.ones(num_class).cuda()
                        unique_proportions[value - 1] = counts.float()
                        entropy = Categorical(probs=unique_proportions / sum(counts)).entropy()
                        check_value.append(entropy)

                    # save the hypothetical labels for the regression heads at Stage 2
                    cls_results[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['batch_rcnn_cls']
                    reg_results[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['batch_rcnn_reg']
                    # used for sorting
                    select_dic[unlabelled_batch['frame_id'][batch_inx]] = entropy
                    # save the density records for the Stage 3
                    density_list[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx][
                        'pred_box_unique_density']
                    label_list[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['pred_labels']

                    total_num_car_pred = torch.zeros(1).to('cuda:0')
                    total_num_cyclist_pred = torch.zeros(1).to('cuda:0')
                    total_num_pedestrain_pred = torch.zeros(1).to('cuda:0')

                    for i_index in range(len(pred_dicts[batch_inx]['pred_labels'])):
                        if pred_dicts[batch_inx]['pred_labels'][i_index] == 1:
                            total_num_car_pred += 1
                        elif pred_dicts[batch_inx]['pred_labels'][i_index] == 2:
                            total_num_pedestrain_pred += 1
                        elif pred_dicts[batch_inx]['pred_labels'][i_index] == 3:
                            total_num_cyclist_pred += 1

                    predicted_label_dict[unlabelled_batch['frame_id'][batch_inx]] = {
                        "Car": total_num_car_pred,
                        "Cyclist": total_num_cyclist_pred,
                        "Pedestrian": total_num_pedestrain_pred  # Corrected spelling
                    }
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()



        selected_frames = list(label_list.keys())

        selected_id_list, selected_infos = [], []
        unselected_id_list, unselected_infos = [], []




        for i in range(len(self.pairs)):
            if self.pairs[i][0] in selected_frames:
                selected_id_list.append(self.pairs[i][0])
                selected_infos.append(self.pairs[i][1])
            else:
                # no need for unselected part
                if len(unselected_id_list) == 0:
                    unselected_id_list.append(self.pairs[i][0])
                    unselected_infos.append(self.pairs[i][1])

        selected_id_list, selected_infos, \
            unselected_id_list, unselected_infos = \
            tuple(selected_id_list), tuple(selected_infos), \
                tuple(unselected_id_list), tuple(unselected_infos)
        active_training = [selected_id_list, selected_infos, unselected_id_list, unselected_infos]

        labelled_set, _, \
            grad_loader, _, \
            _, _ = build_active_dataloader(
            self.cfg.DATA_CONFIG,
            self.cfg.CLASS_NAMES,
            1,
            False,
            workers=self.labelled_loader.num_workers,
            logger=None,
            training=True,
            active_training=active_training
        )
        grad_dataloader_iter = iter(grad_loader)
        total_it_each_epoch = len(grad_loader)

        self.model.train()
        fc_grad_embedding_list = []
        index_list = []


        if self.rank == 0:
            pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                             desc='inf_grads_unlabelled_set_epoch_%d' % cur_epoch, dynamic_ncols=True)
        for cur_it in range(total_it_each_epoch):
            try:
                unlabelled_batch = next(grad_dataloader_iter)

            except StopIteration:
                unlabelled_dataloader_iter = iter(grad_loader)
                unlabelled_batch = next(grad_dataloader_iter)

            load_data_to_gpu(unlabelled_batch)

            pred_dicts, _, _ = self.model(unlabelled_batch)

            roi_labels = pred_dicts['roi_labels']

            # Flatten the tensor
            class_labels = roi_labels.view(-1)

            # Calculate frequency of each class
            class_frequencies = torch.bincount(class_labels)

            # Since your classes start from 1 and do not include class 0
            # Ignore the zero-th index which counts instances of '0'
            class_frequencies = class_frequencies[1:]



            rcnn_cls_preds_per_sample = pred_dicts['rcnn_cls']  # 128， 1
            rcnn_cls_gt_per_sample = cls_results[unlabelled_batch['frame_id'][0]]

            rcnn_reg_preds_per_sample = pred_dicts['rcnn_reg']
            rcnn_reg_gt_per_sample = reg_results[unlabelled_batch['frame_id'][0]]

            cls_loss, _ = self.get_box_cls_layer_loss_LDAM(
                {'rcnn_cls': rcnn_cls_preds_per_sample, 'rcnn_cls_labels': rcnn_cls_gt_per_sample},class_labels=class_labels, class_frequencies=class_frequencies)

            reg_loss = self.model.roi_head.get_box_reg_layer_loss(
                {'rcnn_reg': rcnn_reg_preds_per_sample, 'reg_sample_targets': rcnn_reg_gt_per_sample})
            # summed lost
            summed_lost = torch.sum(reg_loss, dim=2)
            unique_labels = torch.unique(roi_labels)

            # Preallocate tensors for label sums and counts to maintain gradient tracking
            label_sums = torch.zeros_like(unique_labels, dtype=torch.float)
            label_counts = torch.zeros_like(unique_labels, dtype=torch.float)

            for i, label in enumerate(unique_labels):
                mask = (roi_labels == label)
                label_sums[i] = torch.sum(summed_lost[mask])
                label_counts[i] = torch.sum(mask)

            # Perform division using tensors directly to maintain gradient tracking
            result = label_sums / label_counts

            # Calculate weights as the inverse of the label counts
            weights = 1.0 / label_counts

            # Find the maximum weight and calculate scaling factor
            max_weight = weights.max()
            scaling_factor = 1.00 / max_weight
            adjusted_weights = weights * scaling_factor

            # Adjust the loss by multiplying with the normalized weights
            adjusted_loss = result * adjusted_weights
            result = torch.mean(adjusted_loss)

            # clean cache
            del rcnn_cls_preds_per_sample, rcnn_cls_gt_per_sample
            del rcnn_reg_preds_per_sample, rcnn_reg_gt_per_sample
            torch.cuda.empty_cache()

            loss = cls_loss + result
            self.model.zero_grad()
            loss.backward()

            fc_grads = self.model.roi_head.shared_fc_layer[4].weight.grad.clone().detach().cpu()
            fc_grad_embedding_list.append(fc_grads)
            index_list.append(unlabelled_batch['frame_id'][0])

            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()


        fc_grad_embeddings = torch.stack(fc_grad_embedding_list, 0)
        num_sample = fc_grad_embeddings.shape[0]
        fc_grad_embeddings = fc_grad_embeddings.view(num_sample, -1)
        start_time = time.time()
        fc_grad_embeddings = F.softmax(fc_grad_embeddings, dim=1)

        all_features = fc_grad_embeddings.numpy()

        selector = CustomSelection(n_samples=600, function=self.entropy_based_gradients, optimizer='stochastic',
                                   verbose=True)

        # Fit the selector to your data
        selector.fit(all_features)

        # Transform the data (optional here since you're more likely interested in the indices of the selected samples)
        selected_indices = selector.ranking
        selected_frame_ids = [index_list[i] for i in selected_indices]


        new_select_dic = {}
        for element in selected_frame_ids:
            new_select_dic[element] = select_dic[element]


        new_select_dic = dict(sorted(new_select_dic.items(), key=lambda item: item[1]))

        selected_frames = list(new_select_dic.keys())[::-1][:int(self.cfg.ACTIVE_TRAIN.SELECT_NUMS) * 4]

        self.model.eval()

        ################## greedy algo to select frame which increaes entropy#########
        selected_frames_final = []
        index_of_greedy = 0
        best_entropy = 0
        while len(selected_frames_final) < 100 and index_of_greedy <= 110:
            if_selected = False
            if index_of_greedy == 0:
                current_frame = selected_frames[0]
                total_num_car = total_num_car + predicted_label_dict[current_frame]['Car']
                total_num_pedestrain = total_num_pedestrain + predicted_label_dict[current_frame]['Pedestrian']
                total_num_cyclist = total_num_cyclist + predicted_label_dict[current_frame]['Cyclist']
                unique_proportions = torch.ones(num_class).cuda()
                value = torch.tensor([0, 1, 2], device='cuda')
                counts = torch.tensor([total_num_car, total_num_pedestrain, total_num_cyclist])
                unique_proportions[value] = counts.cuda()
                xx = Categorical(probs=unique_proportions / sum(counts)).probs
                best_entropy = -(xx * xx.log2()).sum()
                selected_frames_final.append(current_frame)
                del selected_frames[0]
            else:
                initial_entropy = best_entropy
                best_frame = None
                for e_index in selected_frames:
                    cur_car = total_num_car + predicted_label_dict[e_index]['Car']
                    cur_ped = total_num_pedestrain + predicted_label_dict[e_index]['Pedestrian']
                    cur_cyclist = total_num_cyclist + predicted_label_dict[e_index]['Cyclist']
                    unique_proportions = torch.ones(num_class).cuda()
                    value = torch.tensor([0, 1, 2], device='cuda')
                    counts = torch.tensor([cur_car, cur_ped, cur_cyclist])
                    unique_proportions[value] = counts.cuda()
                    xx = Categorical(probs=unique_proportions / sum(counts)).probs
                    current_entropy = -(xx * xx.log2()).sum()
                    if current_entropy > initial_entropy:
                        initial_entropy = current_entropy
                        best_frame = e_index
                        if_selected = True
                if best_frame:
                    selected_frames_final.append(best_frame)
                if not if_selected:
                    best_frame = selected_frames[0]
                    selected_frames_final.append(best_frame)
                total_num_car += predicted_label_dict[best_frame]['Car']
                total_num_pedestrain += predicted_label_dict[best_frame]['Pedestrian']
                total_num_cyclist += predicted_label_dict[best_frame]['Cyclist']
                unique_proportions = torch.ones(num_class).cuda()
                value = torch.tensor([0, 1, 2], device='cuda')
                counts = torch.tensor([total_num_car, total_num_pedestrain, total_num_cyclist])
                unique_proportions[value] = counts.cuda()
                xx = Categorical(probs=unique_proportions / sum(counts)).probs
                best_entropy = -(xx * xx.log2()).sum()
                selected_frames.remove(best_frame)
            index_of_greedy += 1
        # returned the index of acquired bounding boxes
        if len(selected_frames_final) < 100:
            # Calculate how many elements to extract from selected_frames
            elements_needed = 100 - len(selected_frames_final)

            # Extract the top elements_needed from selected_frames
            top_elements = selected_frames[:elements_needed]

            # Append extracted elements to selected_frames_final
            selected_frames_final.extend(top_elements)

        return selected_frames_final

    def adding_weight_to_unique_proportions(self, pred_dicts, batch_inx):
        pred_scores = pred_dicts[batch_inx]['pred_logits']
        pred_labels = pred_dicts[batch_inx]['pred_labels']
        device = pred_scores.device

        # Assuming labels are 1, 2, 3, adjust size if you have more labels
        scores_list = [torch.tensor([], device=device) for _ in range(3)]  # List to hold scores for each label

        # Group scores by label
        for label, score in zip(pred_labels, pred_scores):
            index = label.item() - 1  # Adjusting for 0-based indexing
            scores_list[index] = torch.cat((scores_list[index], score.unsqueeze(0)), dim=0)

        # Calculate entropy for each label
        entropy_list = torch.zeros(3, device=device)  # Initialize entropy list
        for i, scores in enumerate(scores_list):
            if scores.nelement() == 0:
                continue  # Skip if no scores for this label
            # Normalize scores to probabilities
            probs = F.softmax(scores, dim=1)
            # Calculate entropy: -sum(p * log(p))
            entropy = -(probs * torch.log(probs + 1e-9)).sum(-1)
            entropy_list[i] = entropy.mean()

        return entropy_list

    def adding_weight_regression(self, pred_dicts, batch_inx):
        pred_logits = pred_dicts[batch_inx]['pred_scores']
        device = pred_logits.device
        pred_labels = pred_dicts[batch_inx]['pred_labels']
        scores_sum = torch.zeros(3)  # For labels 1, 2, 3
        # Initialize a tensor to count the occurrences of each label
        label_counts = torch.zeros(3)

        # Iterate over each label and score
        for label, score in zip(pred_labels, pred_logits):
            # Subtract 1 from label to use it as an index (since labels are 1-based and indices are 0-based)
            index = label.item() - 1
            scores_sum[index] += score.item()
            label_counts[index] += 1

        # Compute the mean score for each label
        mean_confidence_box = scores_sum / label_counts
        mean_confidence_box = torch.nan_to_num(mean_confidence_box)
        return mean_confidence_box.to(device)

    def get_box_cls_layer_loss_LDAM(self, forward_ret_dict, class_labels, class_frequencies, reduce=True):
        loss_cfgs = self.model.roi_head.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        batch_size = forward_ret_dict['rcnn_cls_labels'].shape[0]
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        adjusted_indices = class_labels - 1
        # Compute the margins based on class frequencies
        margins = (1. / torch.sqrt(class_frequencies.float()))

        margins = margins[adjusted_indices]  # assign each label its corresponding margin

        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            # Apply margins to logits
            adjusted_logits = rcnn_cls_flat - margins
            # Compute BCE with logits
            batch_loss_cls = F.binary_cross_entropy_with_logits(adjusted_logits, rcnn_cls_labels.float(),
                                                                reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            if reduce:
                rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
            else:
                rcnn_loss_cls = (batch_loss_cls * cls_valid_mask) / torch.clamp(cls_valid_mask.sum(), min=1.0)
                rcnn_loss_cls = rcnn_loss_cls.view(batch_size, -1).sum(-1)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item() if reduce else rcnn_loss_cls[0].item()}
        return rcnn_loss_cls, tb_dict

    def get_box_reg_layer_loss_LDAM(self, reduce=True):
        box_preds = self.model.roi_head.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.model.roi_head.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.model.roi_head.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.model.roi_head.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.model.roi_head.anchors, list):
            if self.model.roi_head.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.model.roi_head.anchors], dim=0)
            else:
                anchors = torch.cat(self.model.roi_head.anchors, dim=-3)
        else:
            anchors = self.model.roi_head.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        if reduce:
            loc_loss = loc_loss_src.sum() / batch_size
        else:
            loc_loss = loc_loss_src.sum(-1).sum(-1)

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item() if reduce else loc_loss[0].item()
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            if reduce:
                dir_loss = dir_loss.sum() / batch_size
            else:
                dir_loss = dir_loss.sum(-1).sum(-1)
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item() if reduce else dir_loss.item()

        return box_loss, tb_dict

    # Define the entropy-based gradient selection function
    def entropy_based_gradients(self, X):
        # Normalize gradients to probability distributions along features
        if np.sum(X) == 0:
            return 0  # Avoid division by zero
        normalized_gradients = X / np.sum(X, axis=0, keepdims=True)
        entropy = -np.sum(normalized_gradients * np.log(normalized_gradients + 1e-9), axis=0)
        return np.sum(entropy)