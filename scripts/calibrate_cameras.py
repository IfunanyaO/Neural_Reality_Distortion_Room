import cv2
import numpy as np
import yaml
import os
from datetime import datetime

# You MUST use a checkerboard with 7x7 inner corners (8x8 squares!)
CHECKERBOARD = (7, 7)  
SQUARE_SIZE = 0.025  # size in meters, change if needed (not critical)

def generate_checkerboard_points():
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.indices(CHECKERBOARD).T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    return objp

def calibrate_camera(cam_id, num_frames=15):
    print(f"\n=== Starting calibration for camera {cam_id} ===")

    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise Exception(f"Camera {cam_id} cannot be opened.")

    objpoints = []
    imgpoints = []
    objp = generate_checkerboard_points()
    collected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No frame received. Check your camera connection.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Improve detection by enabling adaptive thresholding + normalization
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

        display = frame.copy()

        if found:
            # Increase corner refinement quality
            corners = cv2.cornerSubPix(
                gray,
                corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=(
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001
                )
            )
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners, found)

            collected += 1
            objpoints.append(objp)
            imgpoints.append(corners)

            print(f"[OK] Pattern detected ({collected}/{num_frames})")

            if collected >= num_frames:
                print("[INFO] Enough frames collected!")
                break

        else:
            print("[INFO] Checkerboard NOT detected. Move the board slowly…")

        cv2.imshow(f"Calibration Cam {cam_id}", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Abort requested by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) == 0:
        raise Exception("Calibration failed: No checkerboard detected at all.")

    # Finally calibrate
    print("[INFO] Running camera calibration…")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        None,
        None
    )

    print("[SUCCESS] Calibration complete!")

    return {
        "camera_id": cam_id,
        "intrinsics": mtx.tolist(),
        "distortion": dist.tolist(),
        "resolution": list(gray.shape[::-1]),
    }

def main():
    cam_ids = [0, 1]  # Change based on your system

    results = {"cameras": [], "timestamp": datetime.now().isoformat()}

    for cam in cam_ids:
        data = calibrate_camera(cam)
        results["cameras"].append(data)

    os.makedirs("configs", exist_ok=True)
    with open("configs/cameras.yaml", "w") as f:
        yaml.dump(results, f)

    print("\nSaved calibration as configs/cameras.yaml")


if __name__ == "__main__":
    main()
