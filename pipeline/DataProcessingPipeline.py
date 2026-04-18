import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from scipy.signal import find_peaks
import subprocess

# external .py imports
from pipeline.StrideRangeAnalysis import compute_stride_features

# -----------------------------------
# Constants
# -----------------------------------
FPS_DEFAULT = 30.0
MIN_VIS = 0.3
EPS = 1e-6

mp_pose = mp.solutions.pose
landmark_num = mp_pose.PoseLandmark

JOINTS = [
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]

JOINT_IDS = {name: landmark_num[name].value for name in JOINTS}


# Helpers
# ------------------------------------------------------------------------------

# convert video into browser format using FFmpeg
# recieves soruce of video file and destination converted video file 
def convert_to_browser_mp4(input_path, output_path):
    command = [
        "ffmpeg", # call ffmpeg from command line (asumes its installed) 
        "-y", # tells ffmpeg overwrite output file if it already exists 
        "-i", input_path, # specify input file 
        "-vcodec", "libx264", # tell ffmpeg to encode using H.264 codec
        "-pix_fmt", "yuv420p", # set pixel format to yuv420p (good for browser)
        "-acodec", "aac", # set audio codec to aac (good for browser)
        output_path # where ffmpeg saves the output file 
    ]
    # this line runs the ffmpeg command from python like in the terminal
    # check to raise exception if failed 
    # stdout captures standard output 
    # stder captures error output 
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # returns the path of the converted browser ready video 
    return output_path
    

# 1. Pose extraction and Overlay 
# ------------------------------------------------------------------------------
# input path to video
# indicate that the function returns a df

def extract_pose_dataframe_and_overlay_from_video(
    video_path: str,
    output_video_path: str,
    fps: float = FPS_DEFAULT,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> tuple[pd.DataFrame, str]:
    """
    Run MediaPipe Pose on a video once and:
    1. return a frame-level pose dataframe
    2. create an overlay video

    Returns:
        df: pose dataframe
        browser_ready_path: final overlay path if output_video_path is provided
    """
    # mp shortcuts 
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # open video 
    cap = cv2.VideoCapture(video_path)

    # get actual fps, if actual is greater than default overwrite
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps and actual_fps > 0:
        fps = actual_fps

    # get the og frame width and heigth  in pixels
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # intermediate file name since the video written by opencv might 
    # not be browser compatible so we will need to call convert_to_browser_mp4
    raw_output_path = output_video_path.replace(".mp4", "_raw.mp4")

    # set up the video writer 
    # define the codec identifier for opencv video writing 
    # the * unpack the string so opencv gets the chars individually 
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # create writer object
    # saves frames to raw_output_path using mp4v, fps, w and h 
    writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

    # create pose detector 
    pose = mp_pose.Pose(
        static_image_mode=False, # better for video 
        model_complexity=1, # heavier pose model
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    rows = []
    frame_idx = 0

    try: 
        # frame processing loop 
        while True:
            # read the next frame from the video 
            # ret - wheather the reading was success (when false we are done)
            # frame - the img data 
            ret, frame_bgr = cap.read()
            if not ret: # break when video ends 
                break

            h, w = frame_bgr.shape[:2]
            t = frame_idx / fps
    
            # opencv reads BGR but media pipe needs RGB so we convert 
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # send resutls into mediapipe 
            results = pose.process(frame_rgb)

            # init data for this frame 
            row = {
                "frame_idx": frame_idx,
                "time": t,
                "frame_width": w,
                "frame_height": h,
            }

            # prefill with missing values
            for name in JOINTS:
                row[f"{name}_x"] = np.nan
                row[f"{name}_y"] = np.nan
                row[f"{name}_vis"] = 0.0

            # if pose was detected 
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark # list of landmarks
                for name, joint_id in JOINT_IDS.items():
                    lm = landmarks[joint_id] # get values for that lm
                    row[f"{name}_x"] = lm.x * w
                    row[f"{name}_y"] = lm.y * h
                    row[f"{name}_vis"] = lm.visibility
                    
                if writer is not None:
                    mp_drawing.draw_landmarks(
                        frame_bgr, # img being drawn on
                        results.pose_landmarks,  # the detected body landmarks
                        mp_pose.POSE_CONNECTIONS,  # tell which landmarks connect by lines
                        mp_drawing.DrawingSpec(
                            color=(0, 180, 255),
                            thickness=4,
                            circle_radius=4,
                        ),  # style landmarks 
                        mp_drawing.DrawingSpec(
                            color=(255, 255, 255),
                            thickness=3,
                        ),  # style connections 
                    )

            # write frame to output video 
            # even if pose wasnt detected 
            if writer is not None:
                writer.write(frame_bgr)

            rows.append(row) # add frame to row list
            frame_idx += 1

    finally:
        cap.release() # close input video stream
        if writer is not None:
            writer.release() # close output video file 
        pose.close() # realease mp pose resources 

    df = pd.DataFrame(rows) # convert to df

    if raw_output_path is not None and output_video_path is not None:
        # convert video to browser friendly mp4
        browser_ready_path = convert_to_browser_mp4(raw_output_path, output_video_path)

    # return df and final video path
    return df, browser_ready_path, fps

# 2. Normalization
# ------------------------------------------------------------------------------
def normalize_pose_dataframe(
    df: pd.DataFrame,
    hip_thresh: float = 0.3,
    torso_thresh: float = 0.3,
    interp_limit: int = 10,
    eps: float = EPS,
) -> pd.DataFrame:
    """
    Compute pelvis-centered, torso-length normalized coordinates.
    """
    df = df.copy()

    # pelvis center
    df["pelvis_x"] = (df["LEFT_HIP_x"] + df["RIGHT_HIP_x"]) / 2
    df["pelvis_y"] = (df["LEFT_HIP_y"] + df["RIGHT_HIP_y"]) / 2

    # shoulder midpoint
    df["shoulder_mid_x"] = (df["LEFT_SHOULDER_x"] + df["RIGHT_SHOULDER_x"]) / 2
    df["shoulder_mid_y"] = (df["LEFT_SHOULDER_y"] + df["RIGHT_SHOULDER_y"]) / 2

    # torso length
    df["torso_len"] = np.sqrt(
        (df["shoulder_mid_x"] - df["pelvis_x"]) ** 2 +
        (df["shoulder_mid_y"] - df["pelvis_y"]) ** 2
    )

    valid_core = (
        (df["LEFT_HIP_vis"] >= hip_thresh) &
        (df["RIGHT_HIP_vis"] >= hip_thresh) &
        (df["LEFT_SHOULDER_vis"] >= torso_thresh) &
        (df["RIGHT_SHOULDER_vis"] >= torso_thresh)
    )

    # invalidate unreliable torso/pelvis estimates
    df.loc[~valid_core, ["pelvis_x", "pelvis_y", "torso_len"]] = np.nan

    median_len = df["torso_len"].median()
    if pd.notna(median_len):
        df.loc[df["torso_len"] < 0.4 * median_len, "torso_len"] = np.nan

    # interpolate short gaps
    df[["pelvis_x", "pelvis_y", "torso_len"]] = df[
        ["pelvis_x", "pelvis_y", "torso_len"]
    ].interpolate(limit=interp_limit, limit_direction="both")

    # normalized coordinates
    for j in JOINTS:
        df[f"{j}_xN_torso"] = (df[f"{j}_x"] - df["pelvis_x"]) / (df["torso_len"] + eps)
        df[f"{j}_yN_torso"] = (df[f"{j}_y"] - df["pelvis_y"]) / (df["torso_len"] + eps)

    return df


# 3. Stride detection
# -------------------------------------------------------------------------------

def detect_dominant_side(df: pd.DataFrame, min_vis: float = MIN_VIS) -> str:
    """
    Choose the side with better visibility across the run.
    Uses heel, ankle, knee, and hip visibility as the main side-view joints.
    Returns "LEFT" or "RIGHT".
    """
    left_cols = [
        "LEFT_HEEL_vis",
        "LEFT_ANKLE_vis",
        "LEFT_KNEE_vis",
        "LEFT_HIP_vis",
    ]
    right_cols = [
        "RIGHT_HEEL_vis",
        "RIGHT_ANKLE_vis",
        "RIGHT_KNEE_vis",
        "RIGHT_HIP_vis",
    ]

    # average fraction of frames where each landmark is sufficiently visible
    left_score = np.mean([
        (df[col] >= min_vis).mean() for col in left_cols if col in df.columns
    ])
    right_score = np.mean([
        (df[col] >= min_vis).mean() for col in right_cols if col in df.columns
    ])

    return "LEFT" if left_score >= right_score else "RIGHT"
    
def detect_strides_from_pose(
    df: pd.DataFrame,
    fps: float = FPS_DEFAULT,
    heel_col: str = "LEFT_HEEL_y",
    min_stride_time: float = 0.6,
    prominence_scale: float = 0.3,
    peak_width: int = 3,
    peak_shift: int = -1,
    drop_first_stride: bool = True,
    min_time: float = 0.5,
    max_time: float = 1.2,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Detect stride boundaries from heel vertical motion.
    Returns:
      peaks: heel-strike candidate frame indices
      strides_df: dataframe with columns [stride_id, start_frame, end_frame, duration]
    """
    df = df.copy()

    if heel_col not in df.columns:
        raise ValueError(f"Column '{heel_col}' not found in dataframe.")

    y = df[heel_col].astype(float)

    valid_y = y.dropna()
    if valid_y.empty:
        return np.array([], dtype=int), pd.DataFrame(
            columns=["stride_id", "start_frame", "end_frame"]
        )

    min_distance_frames = max(1, int(min_stride_time * fps))
    prominence = np.std(valid_y) * prominence_scale

    peaks, _ = find_peaks(
        y.ffill().bfill(),
        distance=min_distance_frames,
        prominence=prominence,
        width=peak_width,
    )

    peaks = peaks + peak_shift
    peaks = peaks[peaks >= 0]
    peaks = np.unique(peaks)

    stride_ranges = []
    for i in range(len(peaks) - 1):
        start = int(peaks[i])
        end = int(peaks[i + 1])
        if end > start:
            stride_ranges.append((start, end))

    if drop_first_stride and len(stride_ranges) > 1:
        stride_ranges = stride_ranges[1:]

    strides_df = pd.DataFrame(stride_ranges, columns=["start_frame", "end_frame"])
    
    if strides_df.empty:
        return peaks, pd.DataFrame(columns=["stride_id", "start_frame", "end_frame", "duration"])

    # duration in seconds
    strides_df["duration"] = (strides_df["end_frame"] - strides_df["start_frame"]) / fps

    # keep only realistic strides
    strides_df = strides_df[
        (strides_df["duration"] >= min_time) &
        (strides_df["duration"] <= max_time)
    ].reset_index(drop=True)

    # reassign stride ids after filtering
    strides_df.insert(0, "stride_id", range(1, len(strides_df) + 1))

    return peaks, strides_df

# -----------------------------------
# 4. Wrapper pipeline
# -----------------------------------

def process_video_pipeline(video_path, view_name, overlay_video_path=None, fps: float = FPS_DEFAULT):
    
    pose_df, overlay_path, actual_fps = extract_pose_dataframe_and_overlay_from_video(
        video_path,
        output_video_path=overlay_video_path
    )

    norm_df = normalize_pose_dataframe(pose_df)
    dominant_side = detect_dominant_side(norm_df)

    # use that side's heel for stride detection
    heel_col = f"{dominant_side}_HEEL_y"
    peaks, strides_df = detect_strides_from_pose(
        norm_df,
        fps=actual_fps,
        heel_col=heel_col
    )

    # use that same side for feature extraction
    features_df = compute_stride_features(
        norm_df,
        strides_df,
        fps=actual_fps,
        side=dominant_side
    )
    
    return {
        "pose_df": pose_df,
        "norm_df": norm_df,
        "peaks": peaks,
        "strides_df": strides_df,
        "overlay_video": overlay_path,
        "features_df": features_df
    }