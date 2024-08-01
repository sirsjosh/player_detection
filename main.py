from utils import read_video,save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np

def main():
    #read video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    #INITIALIZE TRACKER
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    #interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    #assign player teams
    team_assiner = TeamAssigner()
    team_assiner.assign_team_color(video_frames[0], tracks['players'][0])

    for frrame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assiner.get_player_team(video_frames[frrame_num],
                                                track['bbox'],
                                                player_id)

            tracks['players'][frrame_num][player_id]['team'] = team
            tracks['players'][frrame_num][player_id]['team_color'] = team_assiner.team_colors[team]

    #assign ball aqcuisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_box = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_box)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # #save cropped image
    # for track_id, player in tracks['players'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]

    #     #crop bbox from frame
    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    #     #saved the cropped image
    #     cv2.imwrite(f'output_videos/cropped_img.jpg', cropped_image)
    #     break

    #draw output
    #draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    #save video
    save_video(output_video_frames, 'output_videos/output_video.mp4')

if __name__ == '__main__':
    main()
