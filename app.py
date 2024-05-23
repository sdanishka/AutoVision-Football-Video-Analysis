import streamlit as st
from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from .team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import os


def process_video(video_file, stub_paths):
    # Save uploaded video to a temporary location
    with open('temp_video.mp4', 'wb') as f:
        f.write(video_file.getvalue())

    # Read Video
    video_frames = read_video('temp_video.mp4')

    # Initialize Tracker
    tracker = Tracker('models/yolov5xu.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path=stub_paths['track'])
    # Get object positions
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path=stub_paths['camera_movement']
                                                                              )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # Draw output
    # Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save video
    output_video_path = 'output_videos/output_video.avi'
    save_video(output_video_frames, output_video_path)

    # Delete temporary video file
    os.remove('temp_video.mp4')

    return output_video_path


def main():
    st.title("Football Video Analysis")

    st.write("Upload a video file..")

    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    stub_track_path = st.text_input("Stub Track Path", 'stubs/track_stubs.pkl')
    stub_camera_movement_path = st.text_input("Stub Camera Movement Path", 'stubs/camera_movement_stub.pkl')

    if video_file is not None and stub_track_path and stub_camera_movement_path:
        stub_paths = {
            'track': stub_track_path,
            'camera_movement': stub_camera_movement_path
        }
        with st.spinner('Processing video...'):
            output_video_path = process_video(video_file, stub_paths)
            st.success('Processing complete!')

        st.download_button(label="Download Processed Video", data=open(output_video_path, 'rb').read(),
                           file_name='processed_video.avi')


if __name__ == '__main__':
    main()
