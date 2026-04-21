# IMPORTS
# ---------------------------------------------------------------------------------------------------------------------------------------------
import streamlit as st 
import pandas as pd
import plotly.express as px
from pathlib import Path # to work with files
from streamlit_local_storage import LocalStorage
import base64
import tempfile
import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
import cv2
import mediapipe as mp
import subprocess
import hashlib
import joblib
import numpy as np
import io
import uuid
import json
import plotly.graph_objects as go

# external .py imports
from pipeline.DataProcessingPipeline import process_video_pipeline

# DB INIT
# -----------------------------------------------------------------------------------------------------------------------------------------------
from supabase import create_client


# Open a connection object to talk to Supabase. 


# read the values from Streamlit's secrets file 
url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]

# create a Supabase client using the url and key 
supabase = create_client(url, key)


# AUTHENTICATION HELPERS 
# -----------------------------------------------------------------------------------------------------------------------------------------------

def sign_up_user(email, password):
    """
    Call Supabase Auth's sign-up method and send a dict with email and password.
    Return results directly back to app.
    """
    # go to auth part of client and tell Supabase to create a new user
    return supabase.auth.sign_up({
        "email": email, # email credentials
        "password": password # password credentials
    })

def sign_in_user(email, password):
    """
    Send email and password to Supabase and attempts to authenticate user.
    Returns the login response.
    If credentials are wrong Supabase returns an error or failed response. 
    """
    # go to auth part of client and tell Supabase to authenticate
    return supabase.auth.sign_in_with_password({
        "email": email,
        "password": password
    })

def sign_out_user():
    """
    Log the user out.
    End current auth session on the Supabase side.
    """
    # call Supabase sign out method
    supabase.auth.sign_out()



# DB HELPERS
# ----------------------------------------------------------------------------------------------------------------------------------------------

# Storage bucket name where run files will be stored 
RUN_FILES_BUCKET = "run-files"

def make_run_paths(user_id: str, run_id: str):
    """
    Create a storage path for overlay video and stride features file and return it as a dict
    """
    # create base folder path, ex: pau/run1
    base = f"{user_id}/{run_id}"
    return {
        "overlay_video_path": f"{base}/overlay_video.mp4",
        "stride_features_path": f"{base}/stride_features.parquet",
    }

def upload_file_to_storage(bucket: str, path: str, file_bytes: bytes, content_type: str):
    """
    Upload a file to Supabase storage
    bucket: sotrage bucket name
    path: location inside the bucket
    file_bytes: actual file content
    content_type: MIME type
    """
    # tell db to go to this bucket and upload a file
    return supabase.storage.from_(bucket).upload(
        path=path,
        file=file_bytes, # content, not file name
        # upsert false so that we do not overwrite existing files 
        # if the same path exists throw error 
        file_options={"content-type": content_type, "upsert": "false"}
    )

def dataframe_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert df to a binary parquet file stored in memory
    We do this to avoid writing a temp file to disk 
    """
    # create in memory binary buffer (like a temp file in RAM)
    buffer = io.BytesIO()
    # wrtie df into that buffer as a parquet file 
    df.to_parquet(buffer, index=False)
    buffer.seek(0) # reset buffer back to the beginning 
    return buffer.read() # read entire buffer and return the bytes 


def save_run_to_supabase(
    user_id: str,
    title: str,
    source_view: str,
    overlay_video_local_path: str,
    features_df: pd.DataFrame,
    run_summary: dict,
    stride_probs_df: pd.DataFrame = None,
    filename: str = None,
    duration_seconds: float = None,
    fps: float = None,
    n_frames: int = None,
):
    """
    Pipeline to upload run info to db
    user_id: user performing run 
    title: run name 
    source_view: side/front/rear
    overlay_video_local_path: location of the overlay video
    features_df: stride features
    runc_summary: run-level predicitons
    stride_probs_df: optional stride probs
    filename: og filename
    duration_seconds: video duration
    fps: frames per second
    n_frames: no. of frames
    Upload artifacts to Storage, then insert rows into runs + predictions.
    """
    run_id = str(uuid.uuid4()) # generate a unique identifier (PK)
    # generate path for overlay and stride features
    paths = make_run_paths(user_id, run_id) 

    # open video file in binary mode 
    with open(overlay_video_local_path, "rb") as f:
        overlay_video_bytes = f.read() # read whole video into memory

    # create copy of og df
    features_to_save = features_df.copy()

    # merge predictions into features table if they exist 
    if stride_probs_df is not None and not stride_probs_df.empty:
        features_to_save = features_to_save.merge(stride_probs_df, on="stride_id", how="left")
        
    # convert features dataframe to parquet bytes
    parquet_bytes = dataframe_to_parquet_bytes(features_to_save)

    # 1) upload overlay video to storage
    upload_file_to_storage(
        RUN_FILES_BUCKET,
        paths["overlay_video_path"],
        overlay_video_bytes,
        "video/mp4"
    )

    # 2) upload stride features parquet
    upload_file_to_storage(
        RUN_FILES_BUCKET,
        paths["stride_features_path"],
        parquet_bytes,
        "application/octet-stream"
    )

    # 3) insert run db row 
    # dict that will become a row 
    run_payload = {
        "id": run_id,
        "user_id": user_id,
        "title": title,
        "source_view": source_view.lower(),
        "status": "completed",
        "overlay_video_path": paths["overlay_video_path"], # path not content
        "stride_features_path": paths["stride_features_path"], # path not content
        "analyzed_at": pd.Timestamp.utcnow().isoformat(), # time stamp
        "duration_seconds": duration_seconds,
        "fps": fps,
        "n_frames": n_frames,
        "notes": filename,
    }

    # insert row into runs table 
    run_response = (
        supabase.table("runs")
        .insert(run_payload)
        .execute()
    )

    # 4) insert predictions db row
    # dict that will become a row 
    prediction_payload = {
        "run_id": run_id,
        # given the binary probability, if it is greater than 0.7 set to true 
        "overstride": run_summary.get("Overstride", 0) >= 0.7,
        "trunk_lean": run_summary.get("Trunk Lean", 0) >= 0.7,
        "high_bounce": run_summary.get("High Bounce", 0) >= 0.7,
        "low_cadence": run_summary.get("Low Cadence", 0) >= 0.7,
        # add the actual calculated probability
        "overstride_prob": float(run_summary.get("Overstride", 0)),
        "trunk_lean_prob": float(run_summary.get("Trunk Lean", 0)),
        "high_bounce_prob": float(run_summary.get("High Bounce", 0)),
        "low_cadence_prob": float(run_summary.get("Low Cadence", 0)),
        "model_name": "side_binary_models",
        "model_version": "v1",
    }

    # insert to predictionstable
    (
        supabase.table("predictions")
        .insert(prediction_payload)
        .execute()
    )

    # return run id, the storage paths, and the database response 
    return run_id, paths, run_response

def get_user_runs(user_id: str):
    """
    Gets all runs from a specific user
    """
    response = (
        # query table runs 
        supabase.table("runs")
        .select("id, title, source_view, uploaded_at, overlay_video_path, stride_features_path")
        .eq("user_id", user_id) #where
        .order("uploaded_at", desc=True) # order by uploaded ascending 
        .execute()
    )

    # return result making sure it is a list 
    return response.data if response.data else []

def get_predictions_for_run(run_id: str):
    """
    Gets all predictions (predictions dict) from a specific run 
    """
    response = (
        # quiery table predictions 
        supabase.table("predictions")
        .select("*")
        .eq("run_id", run_id)
        .limit(1)
        .execute()
    )
    # return response if not null
    return response.data[0] if response.data else None

def download_parquet_from_storage(bucket: str, path: str) -> pd.DataFrame:
    """
    Download parquet bytes  from storage and convert to df 
    bucket: storage bucket name
    path: path of file in the bucket 
    """
    # access storage and that bucket and download file in binary content 
    file_bytes = supabase.storage.from_(bucket).download(path)
    # create in memory binary file like obj and wrap the raw bytes in it 
    buffer = io.BytesIO(file_bytes) # pretend file made from raw data 
    return pd.read_parquet(buffer) # read parquet data from buffer and convert to df 

def get_video_url(supabase, bucket, path):
    return supabase.storage.from_(bucket).get_public_url(path)

def rename_run(run_id: str, user_id: str, new_title: str):
    """
    Rename a specific run 
    """
    return (
        # update query un table runs 
        supabase.table("runs")
        .update({"title": new_title})
        .eq("id", run_id)
        .eq("user_id", user_id)
        .execute()
    )

def delete_storage_file(bucket: str, path: str):
    """
    Delete file from storage 
    """
    if path: # check path exists/not empty
        return supabase.storage.from_(bucket).remove([path]) # go to bucket and remove 
    return None # if missing path do nothing 

def delete_run_from_supabase(run_id: str, user_id: str, overlay_video_path: str = None, stride_features_path: str = None):
    """
    Delete an entire run 
    Delete storage files first, then delete prediction row, then run row.
    Overlay and stride_features paths are optional 
    """

    # 1) delete files from storage
    if overlay_video_path:
        delete_storage_file(RUN_FILES_BUCKET, overlay_video_path)

    if stride_features_path:
        delete_storage_file(RUN_FILES_BUCKET, stride_features_path)

    # 2) delete predictions tied to this run from predictions table
    supabase.table("predictions").delete().eq("run_id", run_id).execute()

    # 3) delete the run row itself from runs table 
    return (
        supabase.table("runs")
        .delete()
        .eq("id", run_id)
        .eq("user_id", user_id)
        .execute()
    )

def run_title_exists(user_id: str, title: str, exclude_run_id: str = None) -> bool:
    """
    Checks whether a run title already exists for a user
    exclude_run_id: ignore this run when checking for dups 
    """
    query = (
        # query table runs 
        supabase.table("runs")
        .select("id") # select only the id
        .eq("user_id", user_id) # from that user
        .eq("title", title) # with that title 
    )

    # execute query 
    response = query.execute()

    # if nothing was found we are good, no runs with that name exist
    if not response.data:
        return False

    # if we are doing a normal dup check, not rename, then we found a dup 
    if exclude_run_id is None:
        return True

    # look through all matching rows, if any has an id that is not the excluded run id
    # (meaining that some run other than itself has the same name) return true 
    return any(row["id"] != exclude_run_id for row in response.data)

def run_filename_exists(user_id: str, filename: str) -> bool:
    """
    Check whether a user already uploaded a file with the same original filename 
    Returns true if match was found, False if no match 
    """
    response = (
        # query runs table 
        supabase.table("runs")
        .select("id") # get run id 
        .eq("user_id", user_id) # of this user 
        .eq("notes", filename) # that matches this name 
        .limit(1)
        .execute()
    )
    return bool(response.data) # convert response to boolean 

# Page config
# -------------------------------------------------------------------------

# This sets up the basic config if the Streamlit page
st.set_page_config(
    page_title="StrideVision", # browser tab title
    page_icon="static/favicon.png", # icon in the browser tab
    layout="wide", # use wide page layout 
    initial_sidebar_state="expanded" # side bar opened by default
)

# function to convert image to base 64
# open img file, read its binary contents convert to base64 text
# input: file path of img
def get_image_base64(path):
    with open(path, "rb") as f: # read in binary mode
        # read the file as binary bytes, convert to base64 encoded bytes
        # finally, decode to turn into python string
        return base64.b64encode(f.read()).decode()

# inject custom css to stramlit for authentiction page
def auth_page_css():
    # inserting markdown content into the app 
    # allowed by unsafe_allow_html=True
    
    st.markdown("""
    <style>
        .stApp {
            background-color: #f3f6fb;
        }

        [data-testid="stSidebar"] {
            display: none;
        }

        .auth-logo-wrap {
            display: flex;
            justify-content: center;
            margin-top: 40px;
            margin-bottom: 18px;
        }

        .auth-title {
            text-align: center;
            font-size: 1.6rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.4rem;
        }

        .auth-subtext {
            text-align: center;
            color: #64748b;
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
        }

        .auth-link-text {
            text-align: center;
            color: #64748b;
            font-size: 0.95rem;
            margin-top: 1rem;
            margin-bottom: 0.4rem;
        }

        /* OUTER CARD */
        div[data-testid="stVerticalBlockBorderWrapper"]:first-of-type {
            background: white;
            border-radius: 24px;
            padding: 2rem 2rem 1.5rem 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.10);
            border: 1px solid rgba(37, 99, 235, 0.08);
        }

        /* INPUT FIELD STYLE */
        div[data-testid="stTextInput"] input,
        div[data-testid="stSelectbox"] > div,
        div[data-testid="stSelectbox"] div[role="combobox"] {
            border-radius: 12px !important;
        }

        div[data-testid="stButton"] > button,
        div[data-testid="stFormSubmitButton"] > button {
            border-radius: 12px;
            height: 44px;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)

# init page (default to login)
# if we havent stored which page the app is on
if "page" not in st.session_state: 
    st.session_state.page = "login" # set it to login 

# check if auth exists 
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False # if not set to false (user not loged by default)

# if not user is stored yet, init as none (no logged user yet)
if "user" not in st.session_state:
    st.session_state.user = None

# check if session has access token, if not set to none
# access token proves if user is authenticated 
if "access_token" not in st.session_state:
    st.session_state.access_token = None

# same for refresh 
# refresh help restore or renew the session 
if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = None

# create local storage helper 
# lets streamlit read and write data in browser local storage 
# session state resets when browser session changes or app reloads
local_storage = LocalStorage()

# if we dont have one or both tokens yet 
if st.session_state.access_token is None or st.session_state.refresh_token is None:
    # try to read the browser local storage 
    stored_auth = local_storage.getItem("sv_auth")
    # if something was found (saved login data exists)
    if stored_auth:
        try:
            # convert what was read from json to pythin dict 
            auth_data = json.loads(stored_auth)
            # pull access and refresh token and save it 
            st.session_state.access_token = auth_data.get("access_token")
            st.session_state.refresh_token = auth_data.get("refresh_token")
        except Exception:
            # if something goes wrong set tokens to none 
            st.session_state.access_token = None
            st.session_state.refresh_token = None

# if both tokens are present 
if st.session_state.access_token and st.session_state.refresh_token:
    try:
        # autoligin setup
        # tell supabase to use these tokens as the current session
        supabase.auth.set_session(
            st.session_state.access_token,
            st.session_state.refresh_token
        )
        # ask db who is the authenticated user 
        # to verify the session is real and valid 
        user_response = supabase.auth.get_user()
        # if a response exists and has a valid user 
        if user_response and user_response.user:
            # store the user object in session state (app knows who is loggen in)
            st.session_state.user = user_response.user
            # mark user as authenticated 
            st.session_state.authenticated = True
            # send the mto the dashboard
            st.session_state.page = "dashboard"
    except Exception:
        # if anything fails
        st.session_state.user = None # remove user oject
        st.session_state.authenticated = False # aithentication is false
        st.session_state.access_token = None # set tokens to none
        st.session_state.refresh_token = None # set tokens to none
        local_storage.deleteItem("sv_auth") # delete auth data from browser storage
        

def login_page():
    # call the css
    auth_page_css()

    # read logo and convert to base 64 to prepare logo for display
    img_b64 = get_image_base64("static/strideVision_nobg.png")

    # inject css again for logo display
    st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: center;
            margin-top: -50px;
            margin-bottom: -10px;
            padding-right: 0px;
        ">
            <img src="data:image/png;base64,{img_b64}" width="360">
        </div>
        """,
        unsafe_allow_html=True
    )

    # create three columns in the layout with relative widths 
    left, center, right = st.columns([1.3, 0.9, 1.3])

    # inside center 
    with center:
        # create streamlit container 
        with st.container():
            st.markdown('<div class="auth-title">Sign in with email</div>', unsafe_allow_html=True) # display title
            st.markdown(
                '<div class="auth-subtext">Access your Stride Vision dashboard</div>',
                unsafe_allow_html=True
            ) # display sub text 

            # create login form 
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                c1, c2, c3 = st.columns([1,1,1])
                with c2:
                    enter_clicked = st.form_submit_button("Enter", use_container_width=True) # button to try to login
    
            if enter_clicked: # clicked
                if not email or not password:
                    st.error("Please enter your email and password.")
                else:
                    try:
                        response = sign_in_user(email, password) # try sign in

                        # if user exists login was successful 
                        if response.user is not None:
                            st.session_state.user = response.user # save auth user
                            st.session_state.authenticated = True # mark as auth 
                            st.session_state.page = "dashboard" # go to daashbpard
                        
                            # if session info exists
                            if response.session:
                                st.session_state.authenticated = True 
                                st.session_state.user = response.user
                                st.session_state.access_token = response.session.access_token # save access tokens 
                                st.session_state.refresh_token = response.session.refresh_token # save refresh tokens 
                            
                                # create json with tokens
                                auth_payload = json.dumps({
                                    "access_token": response.session.access_token,
                                    "refresh_token": response.session.refresh_token
                                })

                                # save tokens in local browser 
                                local_storage.setItem("sv_auth", auth_payload)
                                st.rerun() # force refresh 
                        else:
                            # error message 
                            st.error("Login failed. Please try again.")

                    # if anything crashes show error message 
                    except Exception as e:
                        st.error(f"Login failed: {str(e)}")
    
            st.markdown('<div class="auth-link-text">New here?</div>', unsafe_allow_html=True) 

            # create centered layout again 
            c1, c2, c3 = st.columns([1,1,1])
            with c2: # in middle column
                # register buttin 
                if st.button("Register", key="go_register", use_container_width=True):
                    # change to register page
                    st.session_state.page = "register"
                    st.rerun() # force rerun to load that page 

# same technique as login page 
def register_page():
    auth_page_css()

    img_b64 = get_image_base64("static/strideVision_nobg.png")

    st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: center;
            margin-top: -50px;
            margin-bottom: -80px;
            padding-right: 0px;
        ">
            <img src="data:image/png;base64,{img_b64}" width="360">
        </div>
        """,
        unsafe_allow_html=True
    )


    left, center, right = st.columns([1.3, 0.9, 1.3])

    with center:
        with st.container():
            st.markdown('<div class="auth-title">Register</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="auth-subtext">Create your Stride Vision account</div>',
                unsafe_allow_html=True
            )
    
            with st.form("register_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Repeat Password", type="password")
                c1, c2, c3 = st.columns([1,1,1])
                with c2:
                    register_clicked = st.form_submit_button("Register", use_container_width=True)
    
            if register_clicked:
                if not email or not password or not confirm_password:
                    st.error("Please fill out all fields.")
                elif password != confirm_password:
                    st.error("Passwords do not match.")
                elif len(password) < 12:
                    st.error("Password must be at least 12 characters.")
                else:
                    try:
                        # call sign up function 
                        response = sign_up_user(email, password)
            
                        if response.user is not None:
                            st.success("Account created. Check your email for confirmation if required, then sign in.")
                            st.session_state.page = "login" # go to login 
                            st.rerun() # force rerun 
                        else:
                            # if something fails throw error 
                            st.error("Registration failed. Please try again.")

                    # if something fails throw error 
                    except Exception as e:
                        st.error(f"Registration failed: {str(e)}")
    
            st.markdown(
                '<div class="auth-link-text">Already have an account?</div>',
                unsafe_allow_html=True
            )
    
            c1, c2, c3 = st.columns([1,1,1])
            with c2:
                if st.button("Sign in", key="go_login", use_container_width=True):
                    st.session_state.page = "login" # go to login 
                    st.rerun() # force rerun 

# check crrent page stored in session state
# if login call login function 
if st.session_state.page == "login":
    login_page()
    st.stop() # stop running the rest of the script right now 
    
# if register call register funciton 
elif st.session_state.page == "register":
    register_page()
    st.stop() # stop running the rest of the script right now 

# if not aut of user is none (user not properly logged in) 
# cant reach dashboard wihtout being authenticated 
if not st.session_state.authenticated or st.session_state.user is None:
    st.session_state.page = "login" # force back to login 
    st.warning("Please sign in to access the dashboard.")
    st.stop() # stop execution 

# takes the logged in user object and extract uniqueid 
user_id = st.session_state.user.id

# Custom CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .main {
        background-color: #f8fbff;
    }

    .hero-box {
        background: linear-gradient(155deg, #062163, #3B98FF);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.95;
        line-height: 1.6;
    }

    .metric-card {
        background-color: white;
        padding: 1.2rem 1rem;
        border-radius: 18px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        border-left: 6px solid #2563eb;
        margin-bottom: 1rem;
    }

    .metric-label {
        color: #475569;
        font-size: 0.95rem;
        margin-bottom: 0.2rem;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0f172a;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }

    .small-note {
        color: #64748b;
        font-size: 0.95rem;
    }

    [data-testid="stSidebar"] {
        background-color: #eaf2ff;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0rem;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
        padding: 10px 18px;
    }
    
    .prediction-row {
        display: grid;
        grid-template-columns: 120px 1fr 50px;
        align-items: center;
        gap: 10px;
        padding: 8px 0;
        font-size: 0.95rem;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .prediction-row:last-child {
        border-bottom: none;
    }

</style>
""", unsafe_allow_html=True)

# HELPER FUNCTIONS 
# ------------------------------------------------------------------------------

HISTORY_FEATURE_COLS = [
    "cadence",
    "mean_trunk_angle",
    "max_trunk_angle",
    "hip_vertical_range",
    "vertical_velocity_peak",
    "foot_to_hip_distance_at_contact",
    "knee_angle_at_contact",
    "braking_angle",
    "shank_angle_at_contact",
    "foot_strike_ratio",
    "knee_flexion_change_early_stance",
    "stride_time",
]

METRIC_LABELS = {
    "shank_angle_at_contact": "Lower Leg Angle (Contact)",
    "foot_strike_ratio": "Foot Landing Position",
    "knee_flexion_change_early_stance": "Early Stance Knee Flexion",
    "hip_vertical_range": "Vertical Oscillation",
    "vertical_velocity_peak": "Peak Vertical Speed",
    "mean_trunk_angle": "Average Trunk Lean",
    "max_trunk_angle": "Maximum Trunk Lean",
    "n_frames": "Frames per Stride",
    "stride_time": "Stride Duration",
    "cadence": "Cadence",
    "contact_frame": "Foot Strike Frame",
    "foot_to_hip_distance_at_contact": "Foot-Hip Distance (Contact)",
    "knee_angle_at_contact": "Knee Angle at Foot Strike",
    "braking_angle": "Braking Angle",
}

METRIC_TOOLTIPS = {
    "cadence": "Number of steps taken per minute during running.",
    "stride_time": "Duration of a full stride cycle, measured from one foot strike to the next strike of the same foot.",
    "foot_strike_ratio": "Relative position of the foot at ground contact. Higher values indicate the foot lands farther in front of the body.",
    "shank_angle_at_contact": "Angle of the lower leg relative to vertical at the moment of foot strike.",
    "knee_angle_at_contact": "Knee joint angle at initial ground contact.",
    "knee_flexion_change_early_stance": "Amount of knee flexion that occurs during the early stance phase after foot strike.",
    "foot_to_hip_distance_at_contact": "Horizontal distance between the foot and the hip at ground contact.",
    "braking_angle": "Angle of the leg relative to the body at ground contact that contributes to braking forces.",
    "hip_vertical_range": "Vertical displacement of the hips during a stride cycle.",
    "vertical_velocity_peak": "Maximum vertical velocity of the body during a stride cycle.",
    "mean_trunk_angle": "Average forward lean of the torso relative to vertical during the stride.",
    "max_trunk_angle": "Maximum forward trunk lean observed during the stride cycle.",
    "n_frames": "Number of video frames used to represent a single stride.",
    "contact_frame": "Frame index corresponding to foot strike."
}

def get_file_hash(file_bytes):
    """
    Take raw bytes of a file and create a finger print 
    """
    # use md5 hashing 
    # same file content - same hash 
    # different file content - different hash 
    # then covnert to a readable hexadecimal 
    return hashlib.md5(file_bytes).hexdigest()

def ensure_cache_dir():
    """
    Make sure a folder called cache_videos exists 
    """
    # path object pointing to folder cache_videos
    cache_dir = Path("cache_videos")
    # tries to create the folder if doesnt exists
     # if it does, exist_ok=makes it safe to call again 
    cache_dir.mkdir(exist_ok=True)
    return cache_dir # return path for that folder 
    
def render_prediction_panel(predictions, title="Model Predictions"):
    """
    Show the prdeiction resutls
    Takes in  dict of predictions and renders row for each one
    predictions: dict with each label and the avg prob ex. Overstride: 0.87
    """ 
    
    st.markdown(f"#### {title}")

    # loop through predictions 
    # label - pred name 
    # prob - probability 
    for label, prob in predictions.items():
        percent = int(prob * 100) # turn prob to % 

        bar_length = int(prob * 20)  #scale prob to a bar length of 20 chars 
        # fill solid to bar length and the rest as dotted bar 
        bar = "█" * bar_length + "░" * (20 - bar_length)

        # display a pred row using html 
        st.markdown(
            f"""
            <div class="prediction-row">
                <span>{label}</span>
                <span style="font-family:monospace">{bar}</span>
                <span><b>{percent}%</b></span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
def predict_side_run(features_df):
    """
    features_df: one row per stride with the same feature columns used in training
    returns:
    stride_probs_df: probabilities per stride
    run_summary: average probability per cue across the run
    """

    # select only the cols used during training  
    X = features_df[SIDE_FEATURE_COLS].copy()

    stride_probs = {} # creaty empty dict

    # predict all 4 labels using the binary models 
    for label, model in SIDE_BINARY_MODELS.items():
        # predict_proba returns [P(class=0), P(class=1)] for each stride
        # we take all rows but take col 1 not col 0 P(bad form present)
        stride_probs[label] = model.predict_proba(X)[:, 1]

    # save in df, each row, one stride
    stride_probs_df = pd.DataFrame(stride_probs)
    # add stride id column at pos 0 to df from features df  
    stride_probs_df.insert(0, "stride_id", features_df["stride_id"].values)

    # calculate the final prediction averaging results 
    # take mean of all stride probs return a float 
    run_summary = {
        "Overstride": float(stride_probs_df["overstride_label"].mean()),
        "Trunk Lean": float(stride_probs_df["trunk_lean_label"].mean()),
        "High Bounce": float(stride_probs_df["high_bounce_label"].mean()),
        "Low Cadence": float(stride_probs_df["low_cadence_label"].mean()),
    }

    return stride_probs_df, run_summary

def compute_side_summary_metrics(features_df, run_summary):
    """
    Build the 4 main summary box values for Side View.
    """
    # if features is empty return nan for boxes 
    if features_df is None or features_df.empty:
        return {
            "cadence": np.nan,
            "mean_trunk_angle": np.nan,
            "vertical_oscillation": np.nan,
            "primary_flag": "N/A",
        }

    # compute avg cadence if exists if not set to nan
    cadence = float(features_df["cadence"].mean()) if "cadence" in features_df.columns else np.nan 
    # compute max trunk angle if exists if not set to nan
    mean_trunk_angle = float(features_df["mean_trunk_angle"].mean()) if "mean_trunk_angle" in features_df.columns else np.nan
    # compute vertical oscillation angle if exists if not set to nan
    vertical_oscillation = float(features_df["hip_vertical_range"].mean()) if "hip_vertical_range" in features_df.columns else np.nan

    # if we got run summary 
    if run_summary:
        # set primary flag to the one with max prob
        primary_flag = max(run_summary, key=run_summary.get)
    else:
        primary_flag = "N/A"

    # return all the values 
    return {
        "cadence": cadence,
        "mean_trunk_angle": mean_trunk_angle,
        "vertical_oscillation": vertical_oscillation,
        "primary_flag": primary_flag,
    }

@st.cache_data(show_spinner=False) # apply streamlit caching (dont show spinner)
def get_all_user_history_overview(user_id: str):
    """
    If function is called again with the same input, stramlit can reuse prev result
    returns a table where each row is one run, the cols are metadata, predictions, avg features 
    """
    # get all runs from this user 
    runs = get_user_runs(user_id)
    rows = []

    for run in runs:
        # get predictions from that run from the db 
        pred = get_predictions_for_run(run["id"])

        # craete dict for the current run 
        row = {
            "run_id": run["id"],
            "title": run["title"],
            "source_view": run["source_view"],
            "uploaded_at": run["uploaded_at"],
            "overlay_video_path": run["overlay_video_path"],
            "stride_features_path": run["stride_features_path"],
        }

        # add run level prediction fields if exists
        if pred:
            row.update({
                "overstride_prob": pred.get("overstride_prob"),
                "trunk_lean_prob": pred.get("trunk_lean_prob"),
                "high_bounce_prob": pred.get("high_bounce_prob"),
                "low_cadence_prob": pred.get("low_cadence_prob"),
                # flags: T/F 
                "overstride": pred.get("overstride"),
                "trunk_lean": pred.get("trunk_lean"),
                "high_bounce": pred.get("high_bounce"),
                "low_cadence": pred.get("low_cadence"),
            })

        # add averaged feature summaries from saved parquet
        stride_path = run.get("stride_features_path") # get the path were features are saved 
        if stride_path: # if path saved 
            try:
                # download from storage 
                saved_features_df = download_parquet_from_storage(RUN_FILES_BUCKET, stride_path)
                # add summary metrics into the current row 
                row.update(summarize_run_features(saved_features_df))
            except Exception:
                # if anything fails, skip that run's feature summary 
                pass

        # append row to the list 
        rows.append(row)

    # convert to df
    return pd.DataFrame(rows)

def summarize_run_features(features_df: pd.DataFrame) -> dict:
    """
    Convert stride level features for one run into a single run-level summary row
    Uses the mean across strides for numeric metrics.
    """

    if features_df is None or features_df.empty:
        return {}

    summary = {}

    # for every col we want to summarize 
    for col in HISTORY_FEATURE_COLS:
        # if col exists 
        if col in features_df.columns:
            # convert col to numeric 
            vals = pd.to_numeric(features_df[col], errors="coerce")
            # if there is at least one valid num
            if vals.notna().any():
                # get the mean of that columns 
                summary[col] = float(vals.mean())

    # compute a real percentage flagged strides if label columns exist
    label_cols = [
        "overstride_label",
        "trunk_lean_label",
        "high_bounce_label",
        "low_cadence_label",
    ]

    # keep only the labels that do exist in the df 
    available_label_cols = [c for c in label_cols if c in features_df.columns]
    # if any label is there 
    if available_label_cols:
        # creat df of labels and convert to numeric, replace missing vals w 0 
        label_df = features_df[available_label_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        # for eac hstride check if any of the label columns is >0.7
        any_flagged = (label_df > 0.7).any(axis=1)
        # compute pecetnage of strides that were flagged and store it in summary
        # if 3 true (1) out of 10 3/10=0.3 * 100 = 30%
        summary["pct_flagged_strides"] = float(any_flagged.mean() * 100)

        # loop through each column
        for c in available_label_cols:
            # compute percentage of strides flagged for each label 
            summary[f"{c.replace('_label', '')}_stride_pct"] = float((label_df[c] > 0.7).mean() * 100)

    return summary

def render_side_detail_section(
    features_df: pd.DataFrame,
    run_summary: dict,
    key_prefix: str,
    stride_probs_df: pd.DataFrame,
):
    """
    Reusable function to generate the section with summary boxes and metric explorer 
    Recieves the features df, avg probs per cue, and stride level probs 
    """

    # SUMMARY BOXES
    # --------------------------------------------------------------------------------------------
    st.markdown("#### Run Summary")

    # compute summary metrics 
    summary_metrics = compute_side_summary_metrics(features_df, run_summary)

    # create 4 cols 
    b1, b2, b3, b4 = st.columns(4)

    # col 1 where box 1 will be 
    with b1:
        # -- if no val or the actual avg if its there 
        cadence_text = "--" if pd.isna(summary_metrics["cadence"]) else f"{summary_metrics['cadence']:.1f} spm"
        # create box 
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Cadence</div>
            <div class="metric-value">{cadence_text}</div>
        </div>
        """, unsafe_allow_html=True)

    # same for box 2
    with b2:
        trunk_text = "--" if pd.isna(summary_metrics["mean_trunk_angle"]) else f"{summary_metrics['mean_trunk_angle']:.1f}°"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Mean Trunk Angle</div>
            <div class="metric-value">{trunk_text}</div>
        </div>
        """, unsafe_allow_html=True)

    # same for box 3
    with b3:
        vo_text = "--" if pd.isna(summary_metrics["vertical_oscillation"]) else f"{summary_metrics['vertical_oscillation']:.2f}"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Vertical Oscillation</div>
            <div class="metric-value">{vo_text}</div>
        </div>
        """, unsafe_allow_html=True)

    # same for box 4
    with b4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Primary Flag</div>
            <div class="metric-value">{summary_metrics['primary_flag']}</div>
        </div>
        """, unsafe_allow_html=True)

    # METRIC EXPLORER
    # --------------------------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Metric Explorer")

    # if there are no features show message 
    if features_df is None or features_df.empty:
        st.info("No stride features available for this run.")
        return

    # select columns that are numeric for analysis 
    numeric_cols = features_df.select_dtypes(include="number").columns.tolist()
    # exclude these columns cause they dont give analyzable values 
    exclude_cols = ["stride_id", "contact_frame", "overstride_label", "trunk_lean_label", "high_bounce_label", "low_cadence_label"]
    # select numeric cols that shouldnt be excluded 
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    # select all the columns that are in the METRIC_LABELS list 
    # create a dict with friendly label name
    # ex. "Mean Trunk Angle": "mean_trunk_angle"
    metric_options = {METRIC_LABELS.get(col, col): col for col in numeric_cols}

    # create dropdwon from the metric options 
    # key gives the widget a unique key to prevent collisions 
    selected_label = st.selectbox(
        "Select a stride metric",
        list(metric_options.keys()),
        key=f"{key_prefix}_metric_explorer"
    )

    # the real name column 
    selected_metric = metric_options[selected_label]
    # look the description of the metric in METRIC_TOOLTIPS
    tooltip_text = METRIC_TOOLTIPS.get(selected_metric, "No description available.")
    st.info(f"ℹ️ {tooltip_text}") # display it 

    # smaller df with the stride id and selected metric 
    plot_df = features_df[["stride_id", selected_metric]].copy()
    # force the slected metric to numeric 
    plot_df[selected_metric] = pd.to_numeric(plot_df[selected_metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[selected_metric]) # drop na of selected metric 

    # if empty show message 
    if plot_df.empty:
        st.warning(f"No valid values available for {selected_label}.")
    else:
        # if stride probs are not none and not empty 
        if stride_probs_df is not None and not stride_probs_df.empty:
            # merges stride probs for each stride 
            plot_df = plot_df.merge(stride_probs_df, on="stride_id", how="left")
            # chose the label column with the highest average prob (primary flag)
            flag_col = max(
                ["overstride_label", "trunk_lean_label", "high_bounce_label", "low_cadence_label"],
                key=lambda c: stride_probs_df[c].mean() if c in stride_probs_df.columns else 0
            )
            # add a column and copies the prob values from the primary flag
            plot_df["flag_prob"] = plot_df[flag_col]
            # col that will be used for coloring 
            color_arg = "flag_prob"
            # what label to show on color legend 
            color_label = "Primary Flag Probability"
        else:
            # else do not color the plot 
            color_arg = None
            color_label = None

        st.markdown("#### Metric Across Strides")
        # scatter plot 
        fig_line = px.scatter(
            plot_df,
            x="stride_id",
            y=selected_metric,
            color=color_arg,
            color_continuous_scale="RdYlGn_r" if color_arg else None, # color scale 
            labels={color_arg: color_label} if color_arg else {}, # if color rename legend
            title=f"{selected_label} per stride"
        )
        # add another trace to connect the points 
        fig_line.add_scatter(
            x=plot_df["stride_id"],
            y=plot_df[selected_metric],
            mode="lines",
            line=dict(color="lightgrey", width=1.5),
            showlegend=False # so no extra line in legend
        )

        # target ranges for specific metrics 
        TARGET_RANGES = {
            "foot_to_hip_distance_at_contact": (0.10, 0.18),
            "foot_strike_ratio": (0.9, 1.1),
            "braking_angle": (5, 15),
            "shank_angle_at_contact": (5, 20),
            "knee_angle_at_contact": (160, 175),
            "knee_flexion_change_early_stance": (10, 20),
            "mean_trunk_angle": (5, 12),
            "max_trunk_angle": (8, 18),
            "hip_vertical_range": (0.04, 0.08),
            "vertical_velocity_peak": (0.4, 0.9),
            "stride_time": (0.55, 0.75),
            "cadence": (160, 185)
        }

        # if metric has a range 
        if selected_metric in TARGET_RANGES:
            # save low and high bound 
            low, high = TARGET_RANGES[selected_metric]
            # add horizontal rectangle with height of low and high 
            fig_line.add_hrect(
                y0=low,
                y1=high,
                fillcolor="green",
                opacity=0.12,
                line_width=0,
                layer="below" # place rectangel below data points 
            )

        # make markers bigger 
        fig_line.update_traces(marker=dict(size=10), selector=dict(mode="markers"))
        # white bg and chart height and rename axis
        fig_line.update_layout(
            template="plotly_white", 
            height=350,
            xaxis_title="Stride",
            yaxis_title=selected_label)
        # display figure in streamlit 
        st.plotly_chart(fig_line, use_container_width=True, key=f"{key_prefix}_line_{selected_metric}")

        # if target range show text with optimal range and disclaimer 
        if selected_metric in TARGET_RANGES:
            st.caption(f"Optimal range: {low:.2f} – {high:.2f}")
            st.caption("Target ranges are based on typical recreational running biomechanics and may vary depending on speed, experience, and individual morphology.")

        # ROW 1: GOOD/BAD | METRIC TREND
        # ---------------------------------------------------------------------
        
        # create two colmns 
        col_left, col_right = st.columns(2)

        with col_left:
            """
            Comparison plot 
            """
            st.markdown("#### Good vs Bad Stride Comparison")

            # if stride probs is not empty and not none and there is a flag col
            if stride_probs_df is not None and not stride_probs_df.empty and flag_col in plot_df.columns:

                # subset of df with strides with probs greater then 0.7
                flagged = plot_df[plot_df["flag_prob"] >= 0.7]
                normal = plot_df[plot_df["flag_prob"] < 0.7] # less then 0.7

                # stack both df into one comp_df 
                comp_df = pd.concat([
                    # take flagged df and adds a new col Flagged 
                    flagged.assign(group="Flagged"),
                    # take normal df and add new col Normal 
                    normal.assign(group="Normal")
                ])

                # violin chart using comp_df 
                fig_compare = px.violin(
                    comp_df,
                    y=selected_metric,
                    x="group",
                    box=False,
                    points="all",
                    color="group",
                    color_discrete_map={
                        "Flagged": "#ef4444",
                        "Normal": "#22c55e"
                    }
                )

                # set bg and height
                fig_compare.update_layout(
                    template="plotly_white", 
                    height=350, 
                    showlegend=False, 
                    yaxis_title=selected_label,
                    xaxis_title="Group"
                    )

                # redner plot 
                st.plotly_chart(
                    fig_compare,
                    use_container_width=True,
                    key=f"{key_prefix}_goodbad_{selected_metric}"
                )

                # if both groups have at least one stride 
                if len(flagged) > 0 and len(normal) > 0:
                    # compute percentage difference avg flagged mean - avg normal mean / normal mean * 100
                    # ex normal mean = 10 and flagged mean = 12, (12 - 10) / 10 * 100 = 20%
                    diff = ((flagged[selected_metric].mean() - normal[selected_metric].mean()) /
                            normal[selected_metric].mean()) * 100

                    # show abs differnce , display higher is diff is > 0 else lower 
                    st.info(
                        f"Flagged strides from primary flag show **{abs(diff):.1f}% "
                        f"{'higher' if diff > 0 else 'lower'}** {selected_label} compared to normal strides."
                    )

        with col_right:

            # Smooth version of metric over the run 
            st.markdown("#### Metric Trend During Run")

            # chose a rolling avg window size 
            # number of strides, divide by 8, but cap balue at 11 and ensure value is at least 9
            # so 9 10 or 11
            window = max(9, min(11, len(plot_df) // 8))
            # take metric col create rolling window, center window around each stride allow calculation at edges take the mean of each rolling window 
            rolling = plot_df[selected_metric].rolling(window=window, center=True, min_periods=1).mean()

            # empty plotly grph object 
            fig_roll = go.Figure()
            
            # raw metric line
            fig_roll.add_trace(go.Scatter(
                x=plot_df["stride_id"],
                y=plot_df[selected_metric],
                mode="lines",
                name="Raw stride values",
                line=dict(color="lightgray", width=1.5)
            ))
            
            # rolling average on top
            fig_roll.add_trace(go.Scatter(
                x=plot_df["stride_id"],
                y=rolling,
                mode="lines",
                name=f"{window}-stride rolling average",
                line=dict(color="#2563eb", width=3)
            ))
            
            fig_roll.update_layout(
                template="plotly_white",
                height=320,
                xaxis_title="Stride",
                yaxis_title=selected_label,
                # horizontal legend  above the cahrt near top right  
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            # display the chart 
            st.plotly_chart(
                fig_roll,
                use_container_width=True,
                key=f"{key_prefix}_rolling_{selected_metric}"
            )
            
            st.caption(
                f"Blue line shows the smoothed {window}-stride trend. Grey line shows the raw stride values."
            )

        # ROW 2: STRIDE STABILITY | METRIC BY SEGMENT
        # --------------------------------------------------------------------------------------------
        col_left2, col_right2 = st.columns(2)
    
        with col_left2:
            st.markdown("#### Stride Stability")

            # if metric int ranges
            if selected_metric in TARGET_RANGES:
                # set low to high to optimal range of that metric
                low, high = TARGET_RANGES[selected_metric]
        
                # create df with just the rows where the metric is in the target range 
                stable = plot_df[
                    (plot_df[selected_metric] >= low) &
                    (plot_df[selected_metric] <= high)
                ]
                
                # calculate percentage of strides within that range 
                stability = len(stable) / len(plot_df) * 100
        
                st.markdown(
                    f"""
                    <div style="text-align:center; margin-bottom:-10px;">
                        <div style="font-size:2.2rem; font-weight:700; color:#1f2937;">
                            {stability:.0f}%
                        </div>
                        <div style="font-size:0.95rem; color:#6b7280;">
                            of strides within target range
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # gauge indicator figure 
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge",
                    value=stability, # number shown by gauge is stability percentage
                    gauge={
                        "axis": {
                            "range": [0, 100], # 0 to 100
                            "tickwidth": 1,
                            "tickcolor": "#94a3b8",
                            "tickfont": {"color": "#64748b", "size": 12},
                        },
                        "bar": {"color": "#3b82f6", "thickness": 0.25},
                        "bgcolor": "white",
                        "borderwidth": 0,
                        # color ranges 
                        "steps": [
                            {"range": [0, 40],  "color": "#f87171"},
                            {"range": [40, 60], "color": "#fb923c"},
                            {"range": [60, 80], "color": "#facc15"},
                            {"range": [80, 100], "color": "#4ade80"},
                        ],
                    },

                    domain={"x": [0, 1], "y": [0, 1]} # position in chart 
                ))
        
                fig_gauge.update_layout(
                    template="plotly_white",
                    height=280,
                    margin=dict(l=30, r=30, t=60, b=10),
                    font={"color": "#374151", "family": "sans-serif"}
                )

                # render 
                st.plotly_chart(
                    fig_gauge,
                    use_container_width=True,
                    key=f"{key_prefix}_stability_gauge_{selected_metric}"
                )
            else:
                st.info("No target range defined for this metric.")
                
        with col_right2: 
            st.markdown("#### Metric by Run Segment")

            # split run into 3 equal segments
            segment_df = plot_df.copy()
            # create new col segment and call quantile cut q=3
            segment_df["segment"] = pd.qcut(
                segment_df["stride_id"],
                q=3,
                labels=["Start", "Middle", "End"]
            )
            
            # compute average metric per segment
            segment_summary = (
                segment_df.groupby("segment", as_index=False, observed=False)[selected_metric]
                .mean()
            )
            
            fig_segment = px.bar(
                segment_summary,
                x="segment",
                y=selected_metric,
                text=selected_metric, # display numeric val on top 
                color="segment", # color by segment 
                category_orders={"segment": ["Start", "Middle", "End"]}, # force order 
                labels={
                    "segment": "Run Segment", # axis label 
                    selected_metric: selected_label # metric label 
                },
                color_discrete_map={
                    "Start": "#75dbfa",     
                    "Middle": "#5493ff",   
                    "End": "#3936e3"        
                }
            )
            
            fig_segment.update_traces(
                texttemplate="%{text:.3f}",
                textposition="outside", # text above bars 
                cliponaxis=False # render outside plotting area if needed 
            )
            
            fig_segment.update_layout(
                template="plotly_white",
                height=340,
                margin=dict(l=30, r=30, t=80, b=40),
                showlegend=False,
                xaxis_title="Run Segment",
                yaxis_title=selected_label,
                yaxis=dict(range=[0, segment_summary[selected_metric].max()*1.15]) # axis upper bound bigber to add extra headroom so value labels above the bars do not get cut off 
            )

            # render 
            st.plotly_chart(
                fig_segment,
                use_container_width=True,
                key=f"{key_prefix}_segment_{selected_metric}"
            )
            
            # insight text
            # ex of segment summary
            # Start	175.03337403001618
            # Middle	183.7146046314263
            # End	183.68053451200825
            start_val = segment_summary.loc[segment_summary["segment"] == "Start", selected_metric].iloc[0]
            mid_val = segment_summary.loc[segment_summary["segment"] == "Middle", selected_metric].iloc[0]
            end_val = segment_summary.loc[segment_summary["segment"] == "End", selected_metric].iloc[0]

            # percetnage change from start to end (safe check to avoid dividing by 0)
            change_pct = ((end_val - start_val) / start_val) * 100 if start_val != 0 else 0
            
            # interpretation 
            if abs(change_pct) < 5:
                trend_text = "stayed relatively stable across the run" # stable is less than 5% change
            elif change_pct > 0:
                trend_text = f"increased by {abs(change_pct):.1f}% from start to end" # by how much increase
            else:
                trend_text = f"decreased by {abs(change_pct):.1f}% from start to end" # by how much decrease 
            
            st.info(
                f"{selected_label} {trend_text}. "
                f"Start: {start_val:.3f}, Middle: {mid_val:.3f}, End: {end_val:.3f}."
            )


        st.markdown("---")

        # METRIC SUMMARY 
        # --------------------------------------------------------------------------------------------

        st.markdown("#### Metric Summary")
        mean_val = plot_df[selected_metric].mean()
        std_val = plot_df[selected_metric].std()
        min_val = plot_df[selected_metric].min()
        max_val = plot_df[selected_metric].max()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", f"{mean_val:.3f}")
        c2.metric("Std Dev", f"{std_val:.3f}")
        c3.metric("Min", f"{min_val:.3f}")
        c4.metric("Max", f"{max_val:.3f}")

# CACHED VIDEO PROCESSING 
# -------------------------------------------------------------------------------
@st.cache_data 
def process_and_cache_video(file_hash, view_name, original_filename, file_bytes):
    """
    Apply caching to reuse previous results
    Takes in the raw bin contetns of the uploaded file 
    Which view this video belongs to 
    Og filename 
    """

    # extract file exension and make it lower case 
    suffix = Path(original_filename).suffix.lower()

    file_hash = get_file_hash(file_bytes) # generate unique hashing 
    cache_dir = ensure_cache_dir() # ensure cache folder exists and points to that folder 
    base_name = f"{view_name.lower().replace(' ', '_')}_{file_hash}"

    # creates a path were the uploaded video will be saved 
    # cache_dir/view_name(lowercase, no space, use _ )_unique_hash.suffix(.mp4)
    input_video_path = cache_dir / f"{base_name}{suffix}"
    overlay_video_path = cache_dir / f"{base_name}_overlay.mp4"
    pose_df_path = cache_dir / f"{base_name}_pose.parquet"
    norm_df_path = cache_dir / f"{base_name}_norm.parquet"
    strides_df_path = cache_dir / f"{base_name}_strides.parquet"
    features_df_path = cache_dir / f"{base_name}_features.parquet"
    peaks_path = cache_dir / f"{base_name}_peaks.json" 

    # check if cached input exists 
    # if yes we dont need to save it again 
    if not input_video_path.exists(): # if no, save the uploaded bytes 
        # open target file in binary mode and write the uploaded bytes into it 
        with open(input_video_path, "wb") as f:
            f.write(file_bytes) # write file content to disk 

    # check if all processed outputs already exist
    cache_hit = (
        overlay_video_path.exists()
        and pose_df_path.exists()
        and norm_df_path.exists()
        and strides_df_path.exists()
        and features_df_path.exists()
        and peaks_path.exists()
    )

    if cache_hit:
        processed_now = False

        # if all exists read them 
        pose_df = pd.read_parquet(pose_df_path)
        norm_df = pd.read_parquet(norm_df_path)
        strides_df = pd.read_parquet(strides_df_path)
        features_df = pd.read_parquet(features_df_path)

        # load json as numpy array
        with open(peaks_path, "r") as f:
            peaks = np.array(json.load(f))
            
    else:
        processed_now = True

        # call function if its a not saved run 
        pipeline_results = process_video_pipeline(
            str(input_video_path),
            view_name,
            overlay_video_path=str(overlay_video_path)
        )

        # retrieve df from pipeline 
        pose_df = pipeline_results["pose_df"]
        norm_df = pipeline_results["norm_df"]
        peaks = pipeline_results["peaks"]
        strides_df = pipeline_results["strides_df"]
        features_df = pipeline_results["features_df"]

        # save outputs in cache dir for future reuse
        pose_df.to_parquet(pose_df_path, index=False)
        norm_df.to_parquet(norm_df_path, index=False)
        strides_df.to_parquet(strides_df_path, index=False)
        features_df.to_parquet(features_df_path, index=False)

        # convert to list and save as json 
        with open(peaks_path, "w") as f:
            json.dump(peaks.tolist(), f)

    return {
        "input_video": str(input_video_path),  # path to cached input
        "overlay_video": str(overlay_video_path), # path to overaly video 
        "filename": original_filename, # og filename
        "processed_now": processed_now, # whether video was cached or  just generated
        "pose_df": pose_df, # pose df
        "norm_df": norm_df,  # nromalized df
        "peaks": peaks,  # peaks found 
        "strides_df": strides_df,  # stride df  
        "features_df": features_df, # features df
    }

# SIDEBAR
# -----------------------------------------------------------------------------------------------------------------------------------------------

logo_path = Path("static/strideVision_nobg.png")

# show logo 
if logo_path.exists():
    st.sidebar.image(str(logo_path))

# if we have a logged in user 
if st.session_state.user is not None:

    st.sidebar.write(f"Signed in as: **{st.session_state.user.email}**")

    # if click button log out 
    if st.sidebar.button("Log out"):
        # sign out even if fails 
        try:
            sign_out_user()
        except Exception:
            pass

        # delete saved tokens from browser local storage 
        local_storage.deleteItem("sv_auth")

        st.session_state.user = None # remove user object 
        st.session_state.authenticated = False # mark user as loged out 
        st.session_state.access_token = None # clear tokens
        st.session_state.refresh_token = None # clear tokens 
        st.session_state.page = "login" # set page back to login 
        st.rerun() # rerun 

st.sidebar.markdown("## Upload Videos")
st.sidebar.markdown("Upload the video(s) available for this session.")
st.sidebar.markdown(".mp4 is the only supported format.")

# side video file upload widget 
side_file = st.sidebar.file_uploader(
    "Side View Video", # label shown to user 
    type=["mp4"], # only accept mp4
    key="side_upload"
)

# front video (disabled) 
st.sidebar.file_uploader(
    "Front View Video (coming soon)",
    type=["mp4"],
    disabled=True,
    key="front"
)

# rear video (disabled) 
st.sidebar.file_uploader(
    "Rear View Video (coming soon)",
    type=["mp4"],
    disabled=True,
    key="rear"
)

# clear cache button clears local cache files
if st.sidebar.button("Clear video cache"):
    cache_dir = Path("cache_videos") # poit to cache folder 
    if cache_dir.exists():
        for file in cache_dir.iterdir(): # loop through every file 
            if file.is_file(): # delete files not folders 
                file.unlink() # delete from disk 

    st.cache_data.clear() # clear all cached results 
    st.sidebar.success("Cache cleared.")

# Hero section
# ------------------------------------------------------------------------------------------


st.markdown("""
<div class="hero-box">
    <div class="hero-title">Stride Vision</div>
    <div class="hero-subtitle">
        A running gait analysis dashboard built to transform 2D treadmill video into
        interpretable biomechanical insights. Explore stride-level metrics, compare
        movement patterns across views, and identify potential bad-form indicators.
    </div>
</div>
""", unsafe_allow_html=True)

# project overview
# -----------------------------------------------------------------------------------------
st.markdown('<div class="section-title">Project Overview</div>', unsafe_allow_html=True)
st.markdown("""
This dashboard is designed to support the analysis of running form using extracted stride-level
features. The goal is not only to flag potential issues, but also to present clear and interpretable
metrics such as cadence, stride time, trunk angle, and vertical oscillation.

Use the tabs below to explore results by camera view:
- **Side View** for metrics like overstride, trunk lean, and vertical motion
- **Front View** for symmetry and crossover-related patterns
- **Rear View** for step width, toe-out, and arm asymmetry observations
""")

# LOAD MODELS 
# ----------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_resource
def load_side_models():
    """
    Load this once, keep in memory and reuse across runs 
    Load all side view models 
    """
    binary_models = joblib.load("models/side_binary_models.joblib") # dict of models 
    feature_cols = joblib.load("models/side_feature_cols.joblib") # features used in training 
    label_cols = joblib.load("models/side_label_cols.joblib") # label names 
    return binary_models, feature_cols, label_cols

SIDE_BINARY_MODELS, SIDE_FEATURE_COLS, SIDE_LABEL_COLS = load_side_models() # save as global vars  
          
# Process uploaded videos
# -----------------------------------------------------------------------------------------------------------------------------------------------
uploaded_views = {}

# possible uploaded files (only side view active rn
view_files = {
    "Side View": side_file,
    # "Front View": front_file,
    # "Rear View": rear_file
}

# for view and file in view files 
for view_name, uploaded_file in view_files.items():
    # if something was uploaded 
    if uploaded_file is not None:
        # reads file into raw binary bytes 
        file_bytes = uploaded_file.getvalue()
        file_hash = get_file_hash(file_bytes)

        # display loading spinner s
        with st.spinner(f"Processing {view_name} video..."):
            # call processing function and stores resutls in uploaded views 
            uploaded_views[view_name] = process_and_cache_video(
                file_hash=file_hash,
                file_bytes=file_bytes, 
                view_name=view_name,
                original_filename=uploaded_file.name
            )

def render_view_tab(view_name, view_data):
    """
    Render UI for one uploaded view 
    """
    st.markdown(f"**Uploaded file:** {view_data['filename']}")

    if view_data["processed_now"]:
        st.success("Pose overlay generated successfully.") # message if just processed
    else:
        st.info("Pose overlay loaded from cache.") # message is loaded from cache 

    model_run_summary = None
    stride_probs_df = None

    # if side view
    if view_name == "Side View":
        features_df = view_data["features_df"].copy() # get features df 
        model_features_df = features_df.copy() # copy of that df 

        # check if model is trained on foot_to_hip_distance_at_contact
        if (
            "foot_to_hip_distance_at_contact" in model_features_df.columns
            and "foot_to_hip_distance_at_contact" not in SIDE_FEATURE_COLS
        ):
            model_features_df = model_features_df.drop(
                columns=["foot_to_hip_distance_at_contact"]
            ) # if not, drop the col

        # get stride level probs and run summary 
        stride_probs_df, model_run_summary = predict_side_run(model_features_df)
        view_data["stride_probs_df"] = stride_probs_df# add stride level data to dict 

    # TOP SECTION
    # --------------------------------------------------------------------------------------------
    video_col, side_col = st.columns([2, 1]) # two cols, left twice ads big 

    with video_col:
        st.markdown("#### Pose Overlay Video")
        # open saved video and read it as bytes 
        with open(view_data["overlay_video"], "rb") as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes) # show video 

    with side_col:
        if view_name == "Side View": # if side view
            # show model predictions 
            render_prediction_panel(model_run_summary, title="Model Predictions")

            st.markdown("#### Actions")
            # button to save
            if st.button("Save this run", key=f"save_{view_name}_{view_data['filename']}"):
                try:
                    # check if user has a run saved with this name 
                    existing_filename = run_filename_exists(user_id, view_data["filename"])

                    # if yes, error 
                    if existing_filename:
                        st.error("A saved run with this video filename already exists. Please rename the file before uploading or delete the previous run.")
                    else:
                        # save run to database 
                        save_run_to_supabase(
                            user_id=user_id,
                            title=Path(view_data["filename"]).stem,
                            source_view="side",
                            overlay_video_local_path=view_data["overlay_video"],
                            features_df=view_data["features_df"],
                            run_summary=model_run_summary,
                            stride_probs_df=stride_probs_df,
                            filename=view_data["filename"],
                            n_frames=int(len(view_data["pose_df"])) if view_data.get("pose_df") is not None else None,
                        )
                        st.cache_data.clear() # clear cache data 
                        st.success("Run saved successfully.")
            
                except Exception as e:
                    st.error(f"Failed to save run: {e}")
        else:
            st.info("Model predictions currently available for Side View only.")


    # RUN SUMMARY, METRIC EXPLORER AND PLOTS 
    # -----------------------
    if view_name == "Side View":
        # call function 
        render_side_detail_section(
            features_df=view_data["features_df"],
            run_summary=model_run_summary,
            key_prefix=f"current_{view_name}_{view_data['filename']}",
            stride_probs_df=stride_probs_df,
        )
        
# -----------------------------------
# Dynamic tabs
# -----------------------------------
if uploaded_views:
    st.markdown("## Current Run Analysis")

    view_names = list(uploaded_views.keys()) # collect names of uploeaded views
    tabs = st.tabs(view_names) # crete tabs for each view

    # loop through tabs and render each view
    for tab, view_name in zip(tabs, view_names):
        with tab:
            # inside each tab render function that views processed data
            render_view_tab(view_name, uploaded_views[view_name])
else:
    st.info("Upload at least one video from the sidebar to begin.")


# Past Runs 
# ----------------------------------------------------------------------------------------------------------------------------------------------

st.markdown("## Past Runs")

history_tab, detail_tab = st.tabs(["Overview", "Run Details"]) # create two tabs 

with history_tab: # in history tab 
    history_df = get_all_user_history_overview(user_id)

    if history_df.empty:
        st.info("No saved runs yet.")
    else:
        # convert uploaded to pandas datetime 
        history_df["uploaded_at"] = pd.to_datetime(history_df["uploaded_at"])
        # sort oldest to newest 
        history_df = history_df.sort_values("uploaded_at").reset_index(drop=True)

        # TOP SUMMARY CARDS
        # ---------------------------------------------------------------------------
        total_runs = len(history_df) # number of saved runs 

        # run prob cols 
        prob_cols = [
            "overstride_prob",
            "trunk_lean_prob",
            "high_bounce_prob",
            "low_cadence_prob",
        ]
        # check if theyre there
        available_prob_cols = [c for c in prob_cols if c in history_df.columns]

        # change names to readable ones
        flag_label_map = {
            "overstride_prob": "Overstride",
            "trunk_lean_prob": "Trunk Lean",
            "high_bounce_prob": "High Bounce",
            "low_cadence_prob": "Low Cadence",
        }

        # if prob cols 
        if available_prob_cols:
            # get mean of each 
            mean_probs = history_df[available_prob_cols].mean()
            # select largest one 
            most_common_flag = flag_label_map[mean_probs.idxmax()]
        else:
            most_common_flag = "N/A"

        # display metric cards 
        c1, c2 = st.columns(2)
        c1.metric("Saved Runs", total_runs)
        c2.metric("Most Common Flag", most_common_flag)

        # ROW 1: PIE + RADAR SIDE BY SIDE
        # ----------------------------------------------------------------------------------------------------
        if available_prob_cols:
            # new col primary flag
            # take prob cols and for each row find which label is largest and assign it
            history_df["primary_flag"] = history_df[available_prob_cols].idxmax(axis=1).map(flag_label_map)

        col_pie, col_radar = st.columns(2) # two cols 

        with col_pie:
            st.markdown("### Primary Flag Distribution")
            if available_prob_cols:
                # take primary flag col count how many times each appears  
                pie_counts = (
                    history_df["primary_flag"]
                    .value_counts()
                    .reset_index() # convert to df
                )
                pie_counts.columns = ["flag", "count"] # df column names 

                # pie chart 
                fig_pie = px.pie(
                    pie_counts,
                    names="flag", # label by flag  
                    values="count", # size by count 
                    hole=0.35 #donut intstead of pie 
                )
                fig_pie.update_layout(template="plotly_white", height=380) # bg & height
                st.plotly_chart(fig_pie, use_container_width=True) # render 

        with col_radar:
            st.markdown("### Run Profile Radar")

            # metrics for radar 
            radar_metrics = [
                "cadence",
                "hip_vertical_range",
                "mean_trunk_angle",
                "foot_to_hip_distance_at_contact",
                "knee_angle_at_contact",
            ]
            # make sure they are in df 
            radar_metrics = [m for m in radar_metrics if m in history_df.columns]

            if len(radar_metrics) >= 3: # if at least 3 metrics are available 
                latest_run = history_df.iloc[-1] # last run is last row 
                historical_avg = history_df[radar_metrics].mean() # avg val of all saved
                # convert metrics to readable name 
                radar_labels = [METRIC_LABELS.get(m, m) for m in radar_metrics]

                def normalize_series(series):
                    """
                    normalize mterics to a 0-100 scale 
                    """
                    min_val = series.min()
                    max_val = series.max()
                    if max_val == min_val: # avoid dividing by 0 so return 50 
                        return series * 0 + 50
                    # min becomes 0 max becomes 100 
                    return (series - min_val) / (max_val - min_val) * 100

                # apply to metrics col by col 
                normalized_df = history_df[radar_metrics].apply(normalize_series)
                norm_latest = normalized_df.iloc[-1] # last run is last row
                norm_avg = normalized_df.mean() # mean of all 

                fig_radar = go.Figure() # enmpty figure

                # add first radar polygon 
                fig_radar.add_trace(go.Scatterpolar(
                    r=[norm_latest[m] for m in radar_metrics], # norm val for latest run 
                    theta=radar_labels, # axis labels 
                    fill="toself", # fill polygon 
                    name=f"Latest Run ({latest_run['title']})", # label trace 
                    hovertemplate=[
                        f"{METRIC_LABELS.get(m, m)}<br>Raw: {latest_run[m]:.3f}<br>Normalized: {norm_latest[m]:.1f}/100"
                        for m in radar_metrics
                    ] # hover text metric name raw val and norm val 
                ))

                # add second radar 
                fig_radar.add_trace(go.Scatterpolar(
                    r=[norm_avg[m] for m in radar_metrics], # norm avg 
                    theta=radar_labels,
                    fill="toself",
                    name="Historical Average",
                    hovertemplate=[
                        f"{METRIC_LABELS.get(m, m)}<br>Avg: {historical_avg[m]:.3f}<br>Normalized: {norm_avg[m]:.1f}/100"
                        for m in radar_metrics
                    ] # hover text metric name raw val and norm val 
                ))

                
                fig_radar.update_layout(
                    template="plotly_white",
                    height=380,
                    margin=dict(l=120, r=80, t=80, b=80), # increase left margin
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100], # axis goes 0 - 100
                            tickvals=[0, 25, 50, 75, 100], # tick marks 
                            ticktext=["0", "25", "50", "75", "100"] # text 
                        )
                    )
                )

                # renader 
                st.plotly_chart(fig_radar, use_container_width=True)
                st.caption(
                    "All metrics are normalized to a 0–100 scale relative to your personal min/max across all runs. "
                    "Hover over each point to see the original raw value."
                )

        st.markdown("---")

        # IMPROVEMENT TRACKER
        # ----------------------------------------------------------------------------------------------------
        st.markdown("### Improvement Tracker")
        
        improvement_metrics = [
            "overstride_prob",
            "trunk_lean_prob",
            "high_bounce_prob",
            "low_cadence_prob",
        ]
        
        improvement_labels = {
            "overstride_prob": "Overstride",
            "trunk_lean_prob": "Trunk Lean",
            "high_bounce_prob": "High Bounce",
            "low_cadence_prob": "Low Cadence",
        }

        # check if available 
        available_metrics = [m for m in improvement_metrics if m in history_df.columns]
        
        if len(history_df) >= 2 and available_metrics: # if at least two saved runs 

            run_titles = history_df["title"].tolist() # convert titles to list 
        
            c1, c2 = st.columns(2)
        
            with c1:
                # select box 1 
                baseline_run_title = st.selectbox(
                    "Baseline run",
                    run_titles,
                    index=0,
                    key="baseline_run_select"
                )
        
            with c2:
                # select box 2 
                compare_run_title = st.selectbox(
                    "Compare run",
                    run_titles,
                    index=len(run_titles)-1,
                    key="compare_run_select"
                )

            # retrieve selected rows 
            baseline_run = history_df[history_df["title"] == baseline_run_title].iloc[0]
            compare_run = history_df[history_df["title"] == compare_run_title].iloc[0]
        
            improvement_rows = []
        
            for metric in available_metrics: # for each metric 

                # get the selected probs and convert to numeric 
                baseline_val = pd.to_numeric(baseline_run.get(metric), errors="coerce")
                compare_val = pd.to_numeric(compare_run.get(metric), errors="coerce")

                # if baseline or compare are not na and baseline is not 0 to avoid div by 0
                if pd.notna(baseline_val) and pd.notna(compare_val) and baseline_val != 0:

                    # calculate percentage change 
                    change_pct = ((compare_val - baseline_val) / baseline_val) * 100
        
                    # for form cues lower is better
                    # a decrese in % is imporvement 
                    # so reverse sign 
                    if metric in [
                        "overstride_prob",
                        "trunk_lean_prob",
                        "high_bounce_prob",
                        "low_cadence_prob"
                    ]:
                        display_change = -change_pct
                    else:
                        display_change = change_pct

                    # add to dict 
                    improvement_rows.append({
                        "metric": improvement_labels[metric],
                        "change": display_change
                    })
        
            improvement_df = pd.DataFrame(improvement_rows) # turn dict to df 
            
            if not improvement_df.empty:

                # color green if positive else color red 
                bar_colors = [
                    "#22c55e" if v > 0 else "#ef4444"
                    for v in improvement_df["change"]
                ]
        
                fig = go.Figure() # create fig 

                # horizontal bar chart 
                fig.add_trace(go.Bar(
                    x=improvement_df["change"], # bar length improve %
                    y=improvement_df["metric"], # category is metric name 
                    orientation="h",
                    marker_color=bar_colors, # color by 
                    text=[f"{v:+.1f}%" for v in improvement_df["change"]], # show % label with sign 
                    textposition="outside" # labels outside bars 
                ))

                # add vertical dash line at x = 0 
                fig.add_vline(
                    x=0,
                    line_width=1.5,
                    line_dash="dash",
                    line_color="gray"
                )


                # get largest abs val, compare to 10 to have min width 
                max_abs = max(abs(improvement_df["change"]).max(), 10)
        
                fig.update_layout(
                    template="plotly_white",
                    height=320,
                    xaxis_title="Improvement (%)",
                    yaxis_title="",
                    xaxis=dict(range=[-max_abs*1.25, max_abs*1.25]),
                    margin=dict(l=40, r=40, t=20, b=40)
                )

                st.plotly_chart(fig, use_container_width=True) #render 
        
                st.caption(
                    "Positive values indicate improvement relative to the baseline run. "
                    "For form cues, improvement means the probability decreased. "
                )
        
        else:
            st.info("Save at least two runs to compare improvements.")

        st.markdown("---")

        # ROW 2: FEATURE TRENDS 
        # ----------------------------------------------------------------------------------------------------
        st.markdown("### Feature Trends Across Runs")

        feature_candidates = [
            "cadence", "mean_trunk_angle", "max_trunk_angle", "hip_vertical_range",
            "vertical_velocity_peak", "foot_to_hip_distance_at_contact",
            "knee_angle_at_contact", "braking_angle", "shank_angle_at_contact",
            "foot_strike_ratio", "knee_flexion_change_early_stance", "stride_time",
        ]

        # make sure metrics are in df 
        available_feature_cols = [c for c in feature_candidates if c in history_df.columns]
 
        if available_feature_cols: # if any col available 
            # # sort values by upload time 
            history_df = history_df.sort_values("uploaded_at").reset_index(drop=True)
            # add col run number 
            history_df["run_number"] = range(1, len(history_df) + 1)
            # add col run label 
            history_df["run_label"] = "Run " + history_df["run_number"].astype(str)

            # convert cols to readable 
            feature_options = {METRIC_LABELS.get(col, col): col for col in available_feature_cols}

            # select box of the available readable cols 
            selected_feature_label = st.selectbox(
                "Select a feature to compare across runs",
                list(feature_options.keys()),
                key="history_feature_select"
            )

            selected_feature = feature_options[selected_feature_label] # store selected feature

            fig = go.Figure() # create figure 

            
            fig.add_trace(go.Scatter(
                x=history_df["run_label"],
                y=history_df[selected_feature],
                mode="lines",
                line=dict(color="lightgray", width=1.5),
                name="Trend",
                showlegend=False
            ))

            flag_colors = {
                "Overstride": "#6366f1",
                "High Bounce": "#6ec1e4",
                "Trunk Lean": "#ef4444",
                "Low Cadence": "#10b981",
            }

            # list of marker colors one color per run 
            # for each run's primary flag 
            marker_colors = [
                flag_colors.get(flag, "#444444") # default to dark gray 
                for flag in history_df["primary_flag"].tolist()
            ]

            fig.add_trace(go.Scatter(
                x=history_df["run_label"],
                y=history_df[selected_feature],
                mode="markers",
                marker=dict(size=11, color=marker_colors),
                hovertext=history_df["title"],
                name="Run",
                showlegend=False
            ))

            # get the flags that are present in these runs 
            flags_present = history_df["primary_flag"].unique()
            
            for flag, color in flag_colors.items():
                # if flag present 
                if flag in flags_present: 
                    # manual legend 
                    fig.add_trace(go.Scatter(
                        x=[None], y=[None], # dont show points on plot 
                        mode="markers",
                        marker=dict(size=10, color=color), # create fake marker 
                        name=flag, # label wit hflag name 
                        showlegend=True
                    ))

            fig.update_layout(
                template="plotly_white",
                height=380,
                xaxis_title="Run",
                yaxis_title=selected_feature_label,
                showlegend=True,
                legend=dict(
                    title="Primary Flag",
                    orientation="v",
                    yanchor="top", y=1,
                    xanchor="left", x=1.02
                )
            )

            st.plotly_chart(fig, use_container_width=True) # render 

        st.markdown("---")

        # ROW 3: Running Form Quality Across Runs 
        # -------------------------------------------------------------------------------------------------
        st.markdown("### Running Form Quality Across Runs")

        if available_prob_cols:
            def compute_form_score(row):
                # row has pct_flagged_strides do 100 - that 
                if pd.notna(row.get("pct_flagged_strides")):
                    return 100 - row["pct_flagged_strides"]
                else:
                    # collect all prob vals for this run 
                    prob_vals = pd.to_numeric(
                        pd.Series([row.get(c) for c in available_prob_cols]), errors="coerce"
                    )
                    # if at least one 
                    if prob_vals.notna().any():
                        # 100 - the avg prob across cues 
                        return 100 - (prob_vals.mean() * 100)
                    return np.nan

            # new column form_score aply function 
            history_df["form_score"] = history_df.apply(compute_form_score, axis=1)
        else:
            history_df["form_score"] = np.nan

        # if at least one score 
        if history_df["form_score"].notna().any():
            best_idx = history_df["form_score"].idxmax() # get max store and save 
            form_colors = ["#22c55e" if i == best_idx else "#93c5fd" for i in history_df.index] # color for best run 

            fig_form = go.Figure() # create fig 
            fig_form.add_trace(go.Bar(
                x=history_df["title"],
                y=history_df["form_score"],
                marker_color=form_colors,
                text=history_df["form_score"].round(1), # display text outside of bar 
                textposition="outside"
            ))

            fig_form.update_layout(
                template="plotly_white",
                height=380,
                xaxis_title="Run",
                yaxis_title="Form Score",
                yaxis=dict(range=[0, 110])
            )

            st.plotly_chart(fig_form, use_container_width=True) # render 
            st.caption(
                "📊 **Form Score** = 100 - percentage of strides where at least one form issue was detected. "
                "A score of 100 means no strides were flagged; a score of 0 means every stride had at least one issue. "
                "A stride is flagged when the model is more than 70% confident a form issue is present."
            )

            best_run = history_df.loc[history_df["form_score"].idxmax(), "title"] # title of best run 
            best_score = history_df["form_score"].max() # get best score
            st.info(f"🏆 Your best run was **{best_run}** with a form score of **{best_score:.1f}**.")

        st.markdown("---")

        # ROW 4: FLAG HEATMAP 
        # ----------------------------------------------------------------------------------------------------
        st.markdown("### Flag Combination Heatmap")

        heatmap_cols = [
            "overstride_prob", "trunk_lean_prob", "high_bounce_prob", "low_cadence_prob",
        ]
        heatmap_cols = [c for c in heatmap_cols if c in history_df.columns] # check existence

        if heatmap_cols:
            heatmap_df = history_df[["title"] + heatmap_cols].copy() # title + cols 
            # rename for clarity 
            heatmap_df = heatmap_df.rename(columns={
                "overstride_prob": "Overstride",
                "trunk_lean_prob": "Trunk Lean",
                "high_bounce_prob": "High Bounce",
                "low_cadence_prob": "Low Cadence",
            })

            # rows = title cols = cues 
            heatmap_matrix = heatmap_df.set_index("title")

            # create heatmap 
            fig_heat = px.imshow(
                heatmap_matrix,
                text_auto=".0%", # display as percentage 
                aspect="auto",
                color_continuous_scale=[
                    [0.0, "#f3f4f6"],   # light gray (neutral)
                    [0.25, "#fde68a"],  # soft yellow
                    [0.5, "#FBC000"],   # stronger yellow/orange
                    [0.75, "#f7921b"],  # orange
                    [1.0, "#d00202"]    # red
                ],
                labels=dict(x="Cue", y="Run", color="Severity")
            )

            fig_heat.update_layout(
                template="plotly_white",
                height=max(350, 40 * len(heatmap_matrix))
            )

            st.plotly_chart(fig_heat, use_container_width=True) # render 

with detail_tab:
    history_df = get_all_user_history_overview(user_id)

    if history_df.empty:
        st.info("No saved runs yet.")
    else:
        # build run title to display on select box 
        history_df["display_name"] = (
            history_df["title"].fillna("Untitled")
            + " • "
            + history_df["uploaded_at"].astype(str).str[:10]
        )

        # dict that maps display name to run id 
        run_options = dict(zip(history_df["display_name"], history_df["run_id"]))

        # select box list run options keys 
        selected_display = st.selectbox(
            "Select a saved run", # title 
            list(run_options.keys())
        )

        selected_run_id = run_options[selected_display] # get run id  
        # find selected and save it 
        selected_row = history_df[history_df["run_id"] == selected_run_id].iloc[0] 

        # RUN MANAGEMENT ACTIONS
        # --------------------------------------------------------------------------------------------
        st.markdown("#### Manage Selected Run")

        action_col1, action_col2 = st.columns(2)

        with action_col1:
            with st.expander("Rename run"): # expander for rename 
                # recieve text input 
                new_title = st.text_input(
                    "New run title",
                    value=selected_row["title"], # show current name 
                    key=f"rename_input_{selected_run_id}"
                )

                # if clicked strip title 
                if st.button("Save new title", key=f"rename_btn_{selected_run_id}"):
                    cleaned_title = new_title.strip()

                    if not cleaned_title:
                        st.error("Title cannot be empty.")
                    else:
                        try:
                            if run_title_exists(user_id, cleaned_title, exclude_run_id=selected_run_id):
                                # check for dupicate titles 
                                st.error("You already have another run with that title.")
                            else:
                                # rename the run 
                                rename_run(selected_run_id, user_id, cleaned_title)
                                st.cache_data.clear() # clear cache 
                                st.success("Run renamed successfully.")
                                st.rerun() # rerun 
                        except Exception as e:
                            st.error(f"Failed to rename run: {e}")

        with action_col2:
            with st.expander("Delete run"):
                st.warning("This will permanently delete the saved run, overlay video, and saved features.")

                confirm_delete = st.checkbox(
                    "I understand this action cannot be undone",
                    key=f"confirm_delete_{selected_run_id}"
                ) # confrim delete 

                if st.button("Delete this run", key=f"delete_btn_{selected_run_id}"):
                    if not confirm_delete:
                        st.error("Please confirm deletion first.")
                    else:
                        try:
                            # call functio nto delete 
                            delete_run_from_supabase(
                                run_id=selected_run_id,
                                user_id=user_id,
                                overlay_video_path=selected_row.get("overlay_video_path"),
                                stride_features_path=selected_row.get("stride_features_path"),
                            )
                            st.cache_data.clear() # clear cache 
                            st.success("Run deleted successfully.")
                            st.rerun() # rerun 
                        except Exception as e:
                            st.error(f"Failed to delete run: {e}")

        # download features from storage 
        saved_features_df = download_parquet_from_storage(
            RUN_FILES_BUCKET,
            selected_row["stride_features_path"]
        )

        # download overlay from storage 
        video_url = get_video_url(
            supabase,
            RUN_FILES_BUCKET,
            selected_row["overlay_video_path"]
        )
        # rebuild saved run summary 
        saved_run_summary = {
            "Overstride": float(selected_row.get("overstride_prob", 0) or 0),
            "Trunk Lean": float(selected_row.get("trunk_lean_prob", 0) or 0),
            "High Bounce": float(selected_row.get("high_bounce_prob", 0) or 0),
            "Low Cadence": float(selected_row.get("low_cadence_prob", 0) or 0),
        }

        # define which cols should exist 
        saved_stride_prob_cols = [
            "stride_id",
            "overstride_label",
            "trunk_lean_label",
            "high_bounce_label",
            "low_cadence_label",
        ]

        # if all cols are thede 
        if all(col in saved_features_df.columns for col in saved_stride_prob_cols):
            # save df with stride id and four label prob cols  
            saved_stride_probs_df = saved_features_df[saved_stride_prob_cols].copy()
        else:
            saved_stride_probs_df = None

        # TOP SECTION
        # --------------------------------------------------------------------------------------------
        video_col, side_col = st.columns([2, 1])

        with video_col:
            st.markdown("#### Pose Overlay Video")
            st.video(video_url) # show video 

        with side_col:
            render_prediction_panel(saved_run_summary, title="Saved Predictions") # show perds

            st.markdown("<br>", unsafe_allow_html=True)
            
            with st.expander("How are these predictions calculated?"):
                st.markdown("""
                The model analyzes your running form **stride by stride** using biomechanical features extracted from the video.
            
                For each stride, the model predicts a **probability** that a specific form cue is present.  
                These probabilities are then **averaged across all detected strides in the run**.
            
                The value shown in the progress bars represents this **average probability across the entire run**.
                """)

            st.markdown("#### Run Actions")
            # temp file in mem
            parquet_bytes = io.BytesIO() # create in memory binary buffer 
            # take stride features and write as parquet file 
            saved_features_df.to_parquet(parquet_bytes, index=False)

            # download button 
            st.download_button(
                "Download stride features",
                data=parquet_bytes.getvalue(), # get raw bytes from buffer to download 
                file_name=f"{selected_row['title']}_stride_features.parquet",
                mime="application/octet-stream",
                key=f"download_{selected_run_id}"
            )

        # RENDER RUN SUMMARY AND METRICS
        # --------------------------------------------------------------------------------------------
        render_side_detail_section(
            features_df=saved_features_df,
            run_summary=saved_run_summary,
            key_prefix=f"saved_{selected_run_id}",
            stride_probs_df=saved_stride_probs_df
        )
        
# FOOTER
# ----------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.caption("Stride Vision • Senior Project Dashboard v1")
