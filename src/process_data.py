import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import os
import av
import argparse

av.logging.set_level(av.logging.VERBOSE)

def get_closest_frame_timestamp_index(target_timestamp, frame_timestamps):
    idx = np.searchsorted(frame_timestamps, target_timestamp)
    if idx == 0:
        return 0
    if idx == len(frame_timestamps):
        return len(frame_timestamps) - 1
    before = frame_timestamps[idx - 1]
    after = frame_timestamps[idx]
    if abs(after - target_timestamp) < abs(before - target_timestamp):
        return idx
    else:
        return idx - 1


def process_trajectory(data_folder, output_folder, num_trajectory, context_window = 0):

    ## first grab the start time of the first frame in monotonic time from the linux clock.
    base_time_path = f'{data_folder}/start_time.txt'
    with open(base_time_path, 'r') as f:
        line = f.read().strip()
        base_time = int(line)

    start_time_path = f'{data_folder}/timestamp_log.txt'
    with open(start_time_path, 'r') as f:
        line = f.read().strip()
        start_time = int(line)

    actual_start_time = start_time + base_time

    ## create output folder for this trajectory
    output_folder = f'{output_folder}/trajectory_{num_trajectory}'
    os.makedirs(output_folder, exist_ok=True)

    ## first, load in trajectory data and remove all instances where robot is not moving.
    df = pd.read_csv(f'{data_folder}/rollout_data.csv')
    non_zero_index = df[df['speed'] >= 0.05].index
    start_index = max(0, non_zero_index[0] -1 - context_window)
    end_index = min(len(df)-1, non_zero_index[-1] + 1)
    trimmed = df.iloc[start_index:end_index+1]
    print(f'Start index: {start_index}, End index: {end_index}, Original length: {len(df)}, Trimmed length: {len(trimmed)}')

    ## next, load in video frames and match timestamps to trajectory data
    timestamps = np.sort(trimmed['timestamp'].values)
    first_timestamp = timestamps[0]
    last_timestamp = timestamps[-1]

    path = f'{data_folder}/recording.mp4'
    container = av.open(path)
    frame_index = 0
    pose_indices = []
    # decodes at 30 fps.

    for index, frame in enumerate(container.decode(video=0)):
        t_sec = float(frame.pts * frame.time_base)
        t_ns = int(t_sec * 1e9)
        monotonic_time = t_ns + actual_start_time

        ## 30fps so take every 6th frame to get 5fps
        if (index % 6 != 0) or ((monotonic_time < first_timestamp)):
            continue

        if (monotonic_time > last_timestamp):
            break

        pose_index = get_closest_frame_timestamp_index(monotonic_time, timestamps)
        pose_indices.append(pose_index)

        img = frame.to_ndarray(format='rgb24')
        Image.fromarray(img).save(f"{output_folder}/frame_{frame_index}.png")
        frame_index += 1

    output_dataframe = pd.DataFrame(columns=['x', 'y', 'yaw', 'speed'])
    output_dataframe = pd.concat([output_dataframe, trimmed.iloc[pose_indices, 1:]]).reset_index(drop=True)
    output_dataframe.to_csv(f'{output_folder}/trajectory_data.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process trajectory data and video frames.')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the input data folder containing rollout_data.csv and recording.mp4')
    parser.add_argument('--context_window', type=int, default=0, help='Context window of previous frames to include')

    args = parser.parse_args()

    base_path = '/Users/tom/Development/carbot_ai_model'
    
    num_trajectories = 0

    for name in os.listdir(f'{base_path}/data'):
        if name.startswith(args.data_folder):
            print(f'Processing dataset: {name}')
            for index, trajectory in enumerate(os.listdir(f'{base_path}/data/{name}/rollouts')):
                data_folder = f'{base_path}/data/{name}/rollouts/{trajectory}'
                output_folder = f'{base_path}/processed_trajectories/{args.data_folder}'
                process_trajectory(data_folder, output_folder, num_trajectory=num_trajectories, context_window=args.context_window)
                num_trajectories += 1