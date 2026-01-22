import torch
import torch.nn as nn
from mmdet.models import DETECTORS
from projects.mmdet3d_plugin.rtmap.detectors.rtmap import RTMap


@DETECTORS.register_module()
class RTMapLane(RTMap):
    """RTMap variant that accepts pre-detected lane lines as input instead of raw sensor data.
    
    This detector bypasses camera and lidar feature extraction and uses a small
    MLP to embed sequences of lane vertices into a feature vector. The embedded
    lanes are passed to the existing RTMap head to perform localisation and
    change detection in BEV space.

    Note: Only inference is supported in this simplified example. Training
    will require overriding forward_train similar to forward_test to compute
    losses from lane-based observations.
    """

    def __init__(self, *args, pts_bbox_head=None, **kwargs):
        # Force modality to 'lane'
        kwargs = kwargs.copy()
        kwargs['modality'] = 'lane'
        super().__init__(*args, pts_bbox_head=pts_bbox_head, **kwargs)
        assert pts_bbox_head is not None, "RTMapLane requires pts_bbox_head with lane parameters"
        # Build a two-layer embedding network for lane vertices
        in_dim = pts_bbox_head.num_pts_per_gt_vec * 2
        embed_dim = pts_bbox_head.code_size * pts_bbox_head.num_classes
        self.lane_embed = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def extract_img_feat(self, img, img_metas, len_queue=None):
        # No image features used
        return None

    def extract_feat(self, img, img_metas=None, len_queue=None):
        # No image features used
        return None

    def _embed_lane_input(self, external_lane_lines):
        # external_lane_lines: (B, L, N, 2)
        B, L, N, _ = external_lane_lines.shape
        flat = external_lane_lines.view(B, L, N * 2)
        return self.lane_embed(flat)

    def forward(self, img_metas, img=None, points=None, prev_bev=None,
                external_lane_lines=None, rtmap_prior=None, **kwargs):
        """Forward function for lane input.

        Args:
            img_metas (list[dict]): Meta information of samples.
            external_lane_lines (Tensor): Noisy lane detections (B, L, N, 2).
            rtmap_prior (Tensor): Prior map elements.

        Returns:
            Tuple[Tensor, List[dict]]: BEV feature and dummy bbox results.
        """
        assert external_lane_lines is not None and rtmap_prior is not None, \
            "RTMapLane.forward requires external_lane_lines and rtmap_prior"
        lane_feats = self._embed_lane_input(external_lane_lines)
        outs = self.pts_bbox_head(
            lane_feats, None, img_metas, prev_bev, only_bev=False, rtmap_prior=rtmap_prior)
        outs.pop('depth', None)
        bbox_results = [dict(pts_bbox=None) for _ in range(len(img_metas))]
        return outs['bev_embed'], bbox_results
