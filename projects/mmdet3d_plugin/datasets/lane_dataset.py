"""
Dataset for RTMap that consumes pre‭detected structural lane lines instead of
raw sensor data.  This dataset wraps an existing HD”map and returns
artificial "observations" of lane lines by sampling the map and applying
small geometric perturbations.  These noisy observations emulate an upstream
lane”detector and enable training the RTMap localisation and change
detection components without requiring any camera or LiDAR inputs.

The dataset yields, for each sample:

* `external_lane_lines`: a list of lanes, where each lane is a fixed
  number of vertices specified by ``num_pts_per_lane``.  Coordinates are
  provided in the local vehicle coordinate frame.
* `rtmap_prior`: the corresponding prior map extracted around the vehicle
  pose.  This contains the ground truth lane lines from the map (without
  noise) to serve as the reference for matching and localisation.
* `gt_poses_list`: ground truth pose perturbations used to supervise the
  pose auxiliary loss during training.

During training we sample noisy lane observations by randomly jittering the
control points of each lane with small Gaussian noise in both lateral and
longitudinal directions.  In addition, with a configurable probability we
apply a larger perturbation that shifts an entire lane left or right to
mimic common lane”detector mistakes such as choosing the wrong lane.  These
perturbations ensure the network learns to be robust to typical detection
noise and misalignments.

Example usage in a config:

.. code:: python

    data = dict(
        train=dict(
            type='LaneInputDataset',
            map_info_file='data/tbv/tbv_map_infos_train.pkl',
            num_pts_per_lane=20,
            noise_std=0.1,
            lane_shift_std=1.0,
            lane_shift_prob=0.1,
            ...
        ),
        val=...,
        test=...,
    )

This dataset expects ``map_info_file`` to contain a pickled dictionary that
provides, for each sample, the vehicle pose and the list of HD map lane
elements within the perceptual range.  The file format follows the same
conventions as the original RTMap synthetic TbV annotations but without
the camera sensor fields.
"""

import copy
import math
import random
from typing import Dict, List, Tuple

import mmcv
import numpy as np
import torch
from mmdet.datasets import DATASETS
from mmcv.parallel import DataContainer as DC

@DATASETS.register_module()
class LaneInputDataset(object):
    """Dataset that yields noisy lane line observations for RTMap.

    Args:
        map_info_file (str): Path to a pickle file containing map
            annotations.  Each entry should include a vehicle pose and
            a list of lane polylines in local coordinates.
        num_pts_per_lane (int): Number of vertices sampled for each lane.
        noise_std (float): Standard deviation (in meters) for vertex
            jitter applied to every lane point.
        lane_shift_std (float): Standard deviation (in meters) for
            whole”lane shifts along the lateral axis.
        lane_shift_prob (float): Probability of applying a whole”lane
            shift to each lane.
    """

    CLASSES = ('divider', 'ped_crossing', 'boundary')

    def __init__(self,
                 map_info_file: str,
                 num_pts_per_lane: int = 20,
                 noise_std: float = 0.05,
                 lane_shift_std: float = 0.5,
                 lane_shift_prob: float = 0.1,
                 test_mode: bool = False):
        self.map_infos = mmcv.load(map_info_file)
        self.num_pts_per_lane = num_pts_per_lane
        self.noise_std = noise_std
        self.lane_shift_std = lane_shift_std
        self.lane_shift_prob = lane_shift_prob
        self.test_mode = test_mode

    def __len__(self) -> int:
        return len(self.map_infos)

    def _sample_lane(self, pts: np.ndarray) -> np.ndarray:
        """Uniformly resample a lane polyline to a fixed number of points."""
        # Compute cumulative distance along the polyline
        dists = np.cumsum(
            np.concatenate([[0], np.linalg.norm(pts[1:] - pts[:-1], axis=1)]))
        total_length = dists[-1]
        target_dists = np.linspace(0, total_length, self.num_pts_per_lane)
        sampled_pts = np.zeros((self.num_pts_per_lane, 2), dtype=np.float32)
        j = 0
        for i, td in enumerate(target_dists):
            # Advance along original polyline until distance exceeds target
            while j < len(dists) - 1 and dists[j + 1] < td:
                j += 1
            if j == len(dists) - 1:
                sampled_pts[i] = pts[-1]
            else:
                t = (td - dists[j]) / (dists[j + 1] - dists[j] + 1e‑8)
                sampled_pts[i] = (1 - t) * pts[j] + t * pts[j + 1]
        return sampled_pts

    def _perturb_lane(self, lane: np.ndarray) -> np.ndarray:
        """Apply per”vertex jitter and optional whole”lane shift."""
        noisy = lane + np.random.randn(*lane.shape) * self.noise_std
        if random.random() < self.lane_shift_prob:
            shift = np.random.randn() * self.lane_shift_std
            # shift along y axis (lateral) – swap x/y if coordinate system differs
            noisy[:, 1] += shift
        return noisy

    def __getitem__(self, idx: int) -> Dict:
        info = copy.deepcopy(self.map_infos[idx])
        # Extract ground truth lanes and vehicle pose
        gt_lanes = info['gt_lanes']  # list of polylines in local coords
        pose = info['vehicle_pose']   # 4×4 transformation matrix or SE3

        sampled_gt = []
        sampled_noisy = []
        for lane in gt_lanes:
            lane = np.asarray(lane, dtype=np.float32)
            pts = self._sample_lane(lane)
            sampled_gt.append(pts)
            if self.test_mode:
                sampled_noisy.append(pts)
            else:
                sampled_noisy.append(self._perturb_lane(pts))
        sampled_gt = np.stack(sampled_gt)
        sampled_noisy = np.stack(sampled_noisy)

        # Build return dict.  RTMap expects lane vertices as tensors
        external_lines = torch.from_numpy(sampled_noisy)  # [num_lanes, N, 2]
        rtmap_prior = torch.from_numpy(sampled_gt)
        # Build pose perturbations (zero in this dataset; can be randomised)
        gt_delta_pose = torch.eye(4)
        example = {
            'external_lane_lines': DC(external_lines, stack=True, cpu_only=False),
            'rtmap_prior': DC(rtmap_prior, stack=True, cpu_only=False),
            'gt_poses_list': DC(gt_delta_pose, cpu_only=True),
            'img_metas': DC({'can_bus': np.zeros(4)}, cpu_only=True),
        }
        return example
