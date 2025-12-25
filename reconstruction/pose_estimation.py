# reconstruction/pose_estimation.py

import numpy as np
import yaml

class PoseEstimator:
    """
    Loads camera calibration and returns extrinsics/intrinsics.
    """

    def __init__(self, config_path="configs/cameras.yaml"):
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        self.cameras = {}
        for cam in data["cameras"]:
            cam_id = cam["camera_id"]
            self.cameras[cam_id] = {
                "intrinsics": np.array(cam["intrinsics"]),
                "distortion": np.array(cam["distortion"]),
                "resolution": cam["resolution"],
                # For now we assume fixed extrinsics (identity)
                "extrinsics": np.eye(4, dtype=np.float32)
            }

    def get_intrinsics(self, cam_id: int):
        return self.cameras[cam_id]["intrinsics"]

    def get_extrinsics(self, cam_id: int):
        return self.cameras[cam_id]["extrinsics"]

    def get_distortion(self, cam_id: int):
        return self.cameras[cam_id]["distortion"]
