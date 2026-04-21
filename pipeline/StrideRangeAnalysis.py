import numpy as np
import pandas as pd


def split_into_stride_dfs(norm_df: pd.DataFrame, strides_df: pd.DataFrame):
    """
    Split normalized frame-level dataframe into one dataframe per stride.
    Assumes strides_df has columns: stride_id, start_frame, end_frame
    """
    stride_dfs = []

    # loop through each stride
    for _, row in strides_df.iterrows():
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        stride_id = int(row["stride_id"])

        # select frames between start inclusive and end exclusive 
        # because end is where next stride starts to avoid overlap 
        stride_df = norm_df[
            (norm_df["frame_idx"] >= start) & (norm_df["frame_idx"] < end)
        ].copy()

        stride_df["stride_id"] = stride_id
        stride_df["stride_start"] = start
        stride_df["stride_end"] = end - 1

        stride_dfs.append(stride_df) # add stride to list 

    return stride_dfs


def angle_three_pts(a, b, c):
    """
    Given three points, compute angle at point b (the joint)
    # knee angle - (hip, knee, ankle)
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    # two vectors that start at the joint
    ba = a - b # vector b to a
    bc = c - b # vector a to c 

    # comput magnitudes 
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0: # if angle length 0 
        return np.nan

    # cos(θ) = (A·B) / (|A||B|)
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0) # clip to prevent errors
    return np.degrees(np.arccos(cos_angle)) # return angle in degrees 


def angle_to_vertical(top, bottom):
    """
    upper joint and lower joint

    Returns the angle in degrees between the vector top->bottom
    and the vertical downward axis.

    top, bottom are (x, y) points.

    ex trunk lean: shoulder - hip vs vertical
    """
    # convert to arrays
    top = np.array(top, dtype=float) 
    bottom = np.array(bottom, dtype=float)

    v = bottom - top # vector pointing, leg direction at contact 
    norm_v = np.linalg.norm(v) # magnitude

    if norm_v == 0:
        return np.nan

    # y increases downward so [0,1] = vertical downward
    vertical = np.array([0.0, 1.0])  # downward in image coords
    # angle between body segment and vertical axis
    cos_angle = np.dot(v, vertical) / norm_v
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cos_angle)) # return in degrees 


def compute_overstride_features_at_contact(stride_df, side="LEFT"):
    """
    Compute overstide features for one stride 
    """
    contact = stride_df.iloc[0] # take first row of df as contact 

    hip_x = contact[f"{side}_HIP_xN_torso"]
    hip_y = contact[f"{side}_HIP_yN_torso"]

    knee_x = contact[f"{side}_KNEE_xN_torso"]
    knee_y = contact[f"{side}_KNEE_yN_torso"]

    ankle_x = contact[f"{side}_ANKLE_xN_torso"]
    ankle_y = contact[f"{side}_ANKLE_yN_torso"]

    foot_x = contact[f"{side}_HEEL_xN_torso"]
    foot_y = contact[f"{side}_HEEL_yN_torso"]

    # horizontal distance between the heel and hip at contact
    foot_to_hip_distance_at_contact = abs(foot_x - hip_x)

    # compute angle at the knee 
    knee_angle_at_contact = angle_three_pts(
        (hip_x, hip_y),
        (knee_x, knee_y),
        (ankle_x, ankle_y)
    )

    # angle between the line from hip to foot and the vertical axis
    braking_angle = angle_to_vertical(
        (hip_x, hip_y),
        (foot_x, foot_y)
    )

    # angle between the shank segment (knee → ankle) and vertical
    shank_angle_at_contact = abs(
        angle_to_vertical(
            (knee_x, knee_y),
            (ankle_x, ankle_y)
        )
    )

    # Euclidean distance between hip and knee and shnak and knee 
    thigh_len = np.sqrt((hip_x - knee_x) ** 2 + (hip_y - knee_y) ** 2)
    shank_len = np.sqrt((knee_x - ankle_x) ** 2 + (knee_y - ankle_y) ** 2)
    leg_len = thigh_len + shank_len # total leg length 

    if leg_len > 0:
        # normalizes the horizontal foot-to-hip distance by leg length
        # raw distance alone still depends somewhat on body proportions
        # dividing by leg length, gives a more dimensionless measure
        foot_strike_ratio = abs(foot_x - hip_x) / leg_len
    else:
        foot_strike_ratio = np.nan

    n = len(stride_df) # stride length in frames 
    early_n = max(2, int(0.3 * n)) # early stance first 30% of frames
    early_df = stride_df.iloc[:early_n] # df of early stride 

    knee_angles = []
    # compute knee angle at each frame of early stance 
    for _, row in early_df.iterrows():
        hip = (row[f"{side}_HIP_xN_torso"], row[f"{side}_HIP_yN_torso"])
        knee = (row[f"{side}_KNEE_xN_torso"], row[f"{side}_KNEE_yN_torso"])
        ankle = (row[f"{side}_ANKLE_xN_torso"], row[f"{side}_ANKLE_yN_torso"])
        knee_angles.append(angle_three_pts(hip, knee, ankle))

    knee_angles = np.array(knee_angles)

    if len(knee_angles) > 0 and not np.all(np.isnan(knee_angles)):
        contact_angle = knee_angles[0] # first angle is contact 
        min_early = np.nanmin(knee_angles) # find smallest angle in early stance
        # how much the knee flexed after contact during early stance.
        knee_flexion_change_early_stance = contact_angle - min_early
    else:
        knee_flexion_change_early_stance = np.nan

    return {
        "contact_frame": int(contact["frame_idx"] + 1),
        "foot_to_hip_distance_at_contact": foot_to_hip_distance_at_contact,
        "knee_angle_at_contact": knee_angle_at_contact,
        "braking_angle": braking_angle,
        "shank_angle_at_contact": shank_angle_at_contact,
        "foot_strike_ratio": foot_strike_ratio,
        "knee_flexion_change_early_stance": knee_flexion_change_early_stance,
    }


def compute_bounce_features(stride_df):
    """
    Compute high bounce features for one stride 
    """
    # noramlize pelvis 
    pelvis_y = (stride_df["pelvis_y"] / stride_df["torso_len"]).values
    time = stride_df["time"].values # get time vals for each stride 

    # how much the body moves up and down
    hip_vertical_range = np.max(pelvis_y) - np.min(pelvis_y)

    dy = np.diff(pelvis_y) # how much pelvis height changes between consecutive frames
    dt = np.diff(time) # how much time passes between consecutive frames

    if len(dt) == 0 or np.any(dt == 0):
        vertical_velocity_peak = np.nan
    else:
        vertical_velocity = dy / dt # # frame-to-frame vertical speed of the pelvis
        # abs val and find peak
        vertical_velocity_peak = np.max(np.abs(vertical_velocity))

    return {
        "hip_vertical_range": hip_vertical_range,
        "vertical_velocity_peak": vertical_velocity_peak,
    }


def trunk_angle_to_vertical(pelvis, shoulder):
    """
    computes the angle between the trunk and vertical
    """
    # unpack the pelvis and shoulder coordinates into separate x and y values
    px, py = pelvis
    sx, sy = shoulder

    # build trunk vactor pelvis to shoulder 
    vx = sx - px
    vy = sy - py

    # x increases to the right and y down upward vertical is [0, -1]
    vert = np.array([0, -1])
    v = np.array([vx, vy])

    # |v| * |vert|
    denom = np.linalg.norm(v) * np.linalg.norm(vert)
    if denom == 0:
        return np.nan

    cos_angle = np.dot(v, vert) / denom
    cos_angle = np.clip(cos_angle, -1, 1)

    return np.degrees(np.arccos(cos_angle)) # in degrees 


def compute_trunk_features(stride_df):
    """
    Compute trunk lean features for one stride 
    """
    angles = []

    # get all trunk angles 
    for _, row in stride_df.iterrows():
        pelvis = (row["pelvis_x"], row["pelvis_y"])
        shoulder = (row["shoulder_mid_x"], row["shoulder_mid_y"])
        angles.append(trunk_angle_to_vertical(pelvis, shoulder))

    angles = np.array(angles)

    return {
        "mean_trunk_angle": np.nanmean(angles), # find mean
        "max_trunk_angle": np.nanmax(angles), # find max 
    }


def compute_cadence_features(stride_df, fps=30.0):
    """
    Compute cadence features for one stride 
    """
    n_frames = len(stride_df)
    # convert no. of frames to seconds 
    stride_time = n_frames / fps if fps > 0 else np.nan

    if pd.isna(stride_time) or stride_time <= 0:
        cadence = np.nan
    else:
        # steps per second = 2 / stride_time
        # steps per minute = (2 / stride_time) * 60 = 120 / stride_time
        cadence = 120 / stride_time

    return {
        "n_frames": n_frames,
        "stride_time": stride_time,
        "cadence": cadence,
    }


def compute_stride_features(norm_df: pd.DataFrame, strides_df: pd.DataFrame, fps=30.0, side="LEFT"):
    """
    Main feature builder for the dashboard.
    Returns one row per stride.
    """
    # split into stride specific df
    stride_dfs = split_into_stride_dfs(norm_df, strides_df)

    rows = []

    for stride_df in stride_dfs:
        if stride_df.empty:
            continue

        # extract stride id from df
        stride_id = int(stride_df["stride_id"].iloc[0])

        # compute features
        overstride_feats = compute_overstride_features_at_contact(stride_df, side=side)
        bounce_feats = compute_bounce_features(stride_df)
        trunk_feats = compute_trunk_features(stride_df)
        cadence_feats = compute_cadence_features(stride_df, fps=fps)

        row = {"stride_id": stride_id}
        # merge each into row 
        row.update(overstride_feats)
        row.update(bounce_feats)
        row.update(trunk_feats)
        row.update(cadence_feats)

        rows.append(row)

    # turn to df 
    features_df = pd.DataFrame(rows)

    return features_df
