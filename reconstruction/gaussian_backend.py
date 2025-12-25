import numpy as np
import cv2


class GaussianReconstructionBackend:
    """
    Pure NumPy/OpenCV "Gaussian splat" style renderer.

    - Keeps a 3D point cloud in a small volume in front of the camera.
    - Projects points using pose + intrinsics.
    - Renders each point as a colored disk (splat).
    - Updates colors over time based on camera frames.

    This mimics a Gaussian Splatting look without depending on gsplat.
    """

    def __init__(self, num_points: int = 5000):
        self.num_points = num_points
        self.points_world = None   # (N,3)
        self.colors = None         # (N,3) in [0,1]
        self.initialized = False

    def initialize_model(self):
        print("[GS (custom)] Initializing synthetic Gaussian scene...")

        # Random 3D points in a sphere in front of the camera
        pts = np.random.uniform(-1.0, 1.0, (self.num_points, 3)).astype(np.float32)
        # Shift them slightly forward so z is mostly > 0
        pts[:, 2] += 2.0  # push points in front of camera

        self.points_world = pts
        self.colors = np.random.rand(self.num_points, 3).astype(np.float32)

        self.initialized = True

    def update_from_multicam(self, frames):
        """
        Update point colors based on incoming camera frames.

        `frames` is a list of BGR images.
        We softly blend point colors toward the mean image color.
        """
        if not self.initialized or len(frames) == 0:
            return

        # Compute average color across all frames (BGR → normalized)
        mean_bgr = np.mean([f.mean(axis=(0, 1)) for f in frames], axis=0)
        mean_rgb = mean_bgr[::-1] / 255.0  # BGR -> RGB in [0,1]

        # Blend toward this average color
        alpha = 0.05
        self.colors = (1.0 - alpha) * self.colors + alpha * mean_rgb

    def render_view(self, pose, intrinsics, resolution=(360, 640)):
        """
        Render the current point cloud given camera pose & intrinsics.

        pose: 4x4 numpy array, camera-to-world transform (c2w)
        intrinsics: 3x3 numpy array
        resolution: (H, W)

        Returns:
            BGR uint8 image of shape (H, W, 3)
        """
        if not self.initialized:
            raise RuntimeError("Gaussian model not initialized")

        H, W = resolution
        img = np.zeros((H, W, 3), dtype=np.uint8)

        # Invert pose to get world-to-camera
        world_to_cam = np.linalg.inv(pose).astype(np.float32)

        # Homogeneous world coords
        N = self.points_world.shape[0]
        pts_h = np.concatenate(
            [self.points_world, np.ones((N, 1), dtype=np.float32)],
            axis=1,
        )  # (N,4)

        # Transform to camera coordinates
        cam_pts = (world_to_cam @ pts_h.T).T  # (N,4)
        x_c = cam_pts[:, 0]
        y_c = cam_pts[:, 1]
        z_c = cam_pts[:, 2]

        # Keep only points in front of camera
        mask = z_c > 0.1
        if not np.any(mask):
            return img

        x_c = x_c[mask]
        y_c = y_c[mask]
        z_c = z_c[mask]
        colors = self.colors[mask]

        # Project using intrinsics
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        x_n = x_c / z_c
        y_n = y_c / z_c
        u = (fx * x_n + cx).astype(np.int32)
        v = (fy * y_n + cy).astype(np.int32)

        # Render each point as a small disk, size inversely to depth
        for ui, vi, c_rgb, z in zip(u, v, colors, z_c):
            if 0 <= ui < W and 0 <= vi < H:
                # closer points are larger
                base_radius = 2
                radius = int(base_radius + 8.0 / (z + 0.5))
                radius = max(1, min(radius, 12))

                color_bgr = (c_rgb[::-1] * 255.0).astype(np.uint8).tolist()
                cv2.circle(
                    img,
                    (ui, vi),
                    radius,
                    color_bgr,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

        return img
