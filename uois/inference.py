import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import uois.util.utilities as util_
import uois.evaluation as evaluation
import uois.segmentation as segmentation
import uois.data_augmentation as data_augmentation

DSN_CONFIG = {
    # Sizes
    "feature_dim": 64,  # 32 would be normal
    # Mean Shift parameters (for 3D voting)
    "max_GMS_iters": 10,
    "epsilon": 0.05,  # Connected Components parameter
    "sigma": 0.02,  # Gaussian bandwidth parameter
    "num_seeds": 200,  # Used for MeanShift, but not BlurringMeanShift
    "subsample_factor": 5,
    # Misc
    "min_pixels_thresh": 500,
    "tau": 15.0,
}

RRN_CONFIG = {
    # Sizes
    "feature_dim": 64,  # 32 would be normal
    "img_H": 224,
    "img_W": 224,
    # architecture parameters
    "use_coordconv": False,
}

UOIS3D_CONFIG = {
    # Padding for RGB Refinement Network
    "padding_percentage": 0.25,
    # Open/Close Morphology for IMP (Initial Mask Processing) module
    "use_open_close_morphology": True,
    "open_close_morphology_ksize": 9,
    # Largest Connected Component for IMP module
    "use_largest_connected_component": True,
}


class UOISInference:
    def __init__(self):
        checkpoint_dir = pathlib.Path(__file__).parents[1].resolve() / "checkpoints"
        dsn_filename = str(checkpoint_dir / "DepthSeedingNetwork_3D_TOD_checkpoint.pth")
        rrn_filename = str(checkpoint_dir / "RRN_OID_checkpoint.pth")
        UOIS3D_CONFIG["final_close_morphology"] = "TableTop_v5" in rrn_filename
        self.uois_net_3d = segmentation.UOISNet3D(
            UOIS3D_CONFIG, dsn_filename, DSN_CONFIG, rrn_filename, RRN_CONFIG
        )
        return

    def predict_from_depth(self, rgb, depth, depth_K):
        """
        Args:
            rgb: (H, W, 3) np.uint8 image
            depth: (H, W) np.float32 depth image (in meters)
            depth_K: (3, 3) np.float64 camera intrinsics
        Returns:
            seg_mask: (H, W) int64 segmentation mask
        """
        camera_params = {
            "fx": depth_K[0, 0],
            "fy": depth_K[1, 1],
            "x_offset": depth_K[0, 2],
            "y_offset": depth_K[1, 2],
        }
        xyz = util_.compute_xyz(depth, camera_params)
        return self.predict(rgb, xyz)

    def predict(self, rgb, xyz):
        """
        Args:
            rgb: (H, W, 3) np.uint8 image
            xyz: (H, W, 3) np.float32 xyz image (in meters)
        Returns:
            seg_mask: (H, W) np.int64 segmentation mask
        """
        if len(rgb.shape) == 3:
            rgb = np.expand_dims(rgb, 0)
            xyz = np.expand_dims(xyz, 0)
        rgb_augm = data_augmentation.standardize_image(rgb)
        batch = {
            "rgb": data_augmentation.array_to_tensor(rgb_augm),
            "xyz": data_augmentation.array_to_tensor(xyz),
        }

        _, _, _, seg_masks = self.uois_net_3d.run_on_batch(batch)
        seg_mask = seg_masks.cpu().numpy().squeeze(0)
        return seg_mask

    def visualize(self, rgb, xyz, seg_mask, seg_mask_gt=None):
        """
        Args:
            rgb: (H, W, 3) np.uint8 image
            xyz: (H, W, 3) np.float32 xyz image (in meters)
            seg_mask: (H, W) np.int64 segmentation mask
            seg_mask_gt: (H, W) np.int64 ground truth segmentation mask
        """
        num_objs = np.unique(seg_mask).max() + 1
        seg_mask_plot = util_.get_color_mask(seg_mask, nc=num_objs)
        depth = xyz[..., 2]
        images = [rgb, depth, seg_mask_plot]
        titles = [
            f"Image",
            "Depth",
            f"Refined Masks. #objects: {np.unique(seg_mask).shape[0]-1}",
        ]

        if seg_mask_gt is not None:
            num_objs = np.unique(seg_mask_gt).max() + 1
            seg_mask_gt_plot = util_.get_color_mask(seg_mask_gt, nc=num_objs)
            images.append(seg_mask_gt_plot)
            titles.append(f"Ground Truth. #objects: {np.unique(seg_mask_gt).shape[0]-1}")
            eval_metrics = evaluation.multilabel_metrics(seg_mask, seg_mask_gt)
            print(f"Image Metrics: ")
            print(json.dumps(eval_metrics, indent=4))

        util_.subplotter(images, titles)
        plt.show()
        print("Results visualized")
        return


if __name__ == "__main__":
    img_num = 0  # 0-3

    # Load Data
    example_images_dir = pathlib.Path(__file__).parents[1].resolve() / "example_images"
    images_path = sorted(example_images_dir.iterdir())
    img_file = images_path[img_num]
    data = np.load(img_file, allow_pickle=True, encoding="bytes").item()
    rgb = data["rgb"]
    xyz = data["xyz"]
    segm_gt = data["label"].astype(np.int64)

    # Inference
    uois_inference = UOISInference()
    segm = uois_inference.predict(rgb, xyz)
    uois_inference.visualize(rgb, xyz, segm, segm_gt)
