# reconstruction/pose_estimation.py

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, List

CHECKERBOARD = (7, 7)   # inner corners
SQUARE_SIZE = 0.025     # meters (or anything, just be consistent)

class PoseEstimator:
    """
    Loads intrinsics from configs/cameras.yaml and estimates camera extrinsics
    using a checkerboard observed by all cameras.
    """

    def __init__(self, config_path: str = "configs/cameras.yaml"):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Camera config not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            self.data = yaml.safe_load(f)

        self.cameras = {c["camera_id"]: c for c in self.data["cameras"]}
        self._objp = self._generate_checkerboard_points()

    def _generate_checkerboard_points(self) -> np.ndarray:
        objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[:, :2] = np.indices(CHECKERBOARD).T.reshape(-1, 2)
        objp *= SQUARE_SIZE
        return objp

    def _estimate_extrinsics_single(
        self, cam_id: int, frame: np.ndarray
    ) -> Dict[str, Any]:
        cam = self.cameras[cam_id]
        K = np.array(cam["intrinsics"], dtype=np.float32)
        dist = np.array(cam["distortion"], dtype=np.float32)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not ret:
            raise RuntimeError(f"Checkerboard not detected for camera {cam_id}.")

        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=(
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            ),
        )

        ok, rvec, tvec = cv2.solvePnP(self._objp, corners_refined, K, dist)
        if not ok:
            raise RuntimeError(f"solvePnP failed for camera {cam_id}.")

        R, _ = cv2.Rodrigues(rvec)
        T = tvec.reshape(3, 1)

        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3:] = T

        return {
            "rotation_vector": rvec.tolist(),
            "translation_vector": tvec.tolist(),
            "extrinsics_4x4": extrinsic.tolist(),
        }

    def estimate_all_from_live_cameras(
        self, cam_ids: List[int], preview: bool = True
    ) -> None:
        caps = {cid: cv2.VideoCapture(cid, cv2.CAP_DSHOW) for cid in cam_ids}
        print("[PoseEstimator] Press SPACE when checkerboard is visible in all cameras.")
        print("Press Q to abort.")

        frames = {}
        while True:
            for cid, cap in caps.items():
                ret, frame = cap.read()
                if not ret:
                    continue
                frames[cid] = frame
                if preview:
                    cv2.imshow(f"PoseCam {cid}", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                print("[PoseEstimator] Capturing pose frames...")
                break
            if key == ord("q"):
                print("[PoseEstimator] Aborted by user.")
                self._release_caps(caps)
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()

        # Estimate extrinsics per camera
        for cid in cam_ids:
            if cid not in frames:
                print(f"[PoseEstimator] No frame for camera {cid}, skipping.")
                continue
            print(f"[PoseEstimator] Estimating pose for camera {cid}...")
            extrinsic_info = self._estimate_extrinsics_single(cid, frames[cid])
            self.cameras[cid]["extrinsics"] = extrinsic_info["extrinsics_4x4"]

        # Write back to YAML
        self.data["cameras"] = list(self.cameras.values())
        with open(self.config_path, "w") as f:
            yaml.safe_dump(self.data, f)
        print(f"[PoseEstimator] Updated extrinsics written to {self.config_path}")

        self._release_caps(caps)

    def _release_caps(self, caps: Dict[int, cv2.VideoCapture]):
        for cap in caps.values():
            cap.release()
