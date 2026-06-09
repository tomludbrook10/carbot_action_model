import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
from src.model_context import ModelContext

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

def normalise_heading(yaw):
    while yaw > np.pi:
        yaw -= 2 * np.pi
    while yaw < -np.pi:
        yaw += 2 * np.pi
    return yaw

class TrajectorySample:
    def __init__(self, data_folder, model_context, transform=None, frame_skips=1, waypoint_skips=1):
        self.data_folder = data_folder
        self.transform = transform
        self.trajectory_data = pd.read_csv(f'{data_folder}/trajectory_data.csv')
        self.trajectory_data.drop(columns='speed', inplace=True)
        self.num_waypoints = model_context.waypoint_num
        self.data_type = model_context.data_type
        self.frame_skips = frame_skips
        self.waypoint_skips = waypoint_skips
        self.num_prev_observations = model_context.num_prev_observations
        self.model_context = model_context

    def orient_to_reference(self, current_waypoint, waypoints):
        ref_x = current_waypoint['x']
        ref_y = current_waypoint['y']
        ref_yaw = current_waypoint['yaw']

        h_sin = np.sin(-ref_yaw)
        h_cos = np.cos(-ref_yaw)

        ref_yaw = normalise_heading(ref_yaw)

        for i in range(len(waypoints)):
            dx = waypoints.at[i, 'x'] - ref_x
            dy = waypoints.at[i, 'y'] - ref_y

            # Rotate by -ref_yaw
            rotated_x = dx * h_cos - dy * h_sin
            rotated_y = dx * h_sin + dy * h_cos

            waypoints.at[i, 'x'] = rotated_x
            waypoints.at[i, 'y'] = rotated_y

            # orient yaw to heading =0. 
            waypoints.at[i, 'yaw'] -= ref_yaw
            waypoints.at[i, 'yaw'] = normalise_heading(waypoints.at[i, 'yaw'])

    def __getitem__(self, idx):
        if (idx < 0) or (idx >= len(self.trajectory_data)):
            raise IndexError("Index out of bounds")
        
        num_observations = []

        for obs_idx in range(idx - (self.num_prev_observations * self.frame_skips), idx + 1, self.frame_skips):
            obs_idx = max(0, obs_idx) ## clamp to 0, padding with first frame if needed
            o_t = Image.open(f'{self.data_folder}/frame_{obs_idx}.png').convert('RGB')
            if self.transform:
                o_t = self.transform(o_t).type(self.data_type).to(device)
            num_observations.append(o_t)

        num_observations.reverse()

        ## to match the tensorRT input 
        o_t = torch.cat(num_observations, dim=0)
        o_t = o_t.view(self.model_context.num_prev_observations + 1, 
                       self.model_context.channels, 
                       self.model_context.height, 
                       self.model_context.width)
    
        waypoints = None

        current_waypoint = self.trajectory_data.iloc[idx]
        end_index = idx + (self.num_waypoints * self.waypoint_skips) + 1

        indexes = list(range(idx + self.waypoint_skips, end_index, self.waypoint_skips))

        for waypoint_index, index in enumerate(indexes):
            if index >= len(self.trajectory_data):
                indexes[waypoint_index] = len(self.trajectory_data) - 1

        waypoints = self.trajectory_data.iloc[indexes]
        waypoints = waypoints.reset_index(drop=True)
        self.orient_to_reference(current_waypoint, waypoints)

        waypoints = waypoints[['x', 'y']]
        return o_t, torch.tensor(waypoints.values, dtype=self.data_type).to(device)

    def __len__(self):
        return len(self.trajectory_data) - (self.num_waypoints * self.waypoint_skips)

class WaypointDataset(Dataset):
    def __init__(self, data_folder, frame_skips, waypoint_skips, model_context, transform=None):
        self.w_num = model_context.waypoint_num
        self.transform = transform
        self.trajectories = []
        self.len = 0
        self.data_type = model_context.data_type
        self.mean = model_context.mean
        self.std = model_context.std
        self.frame_skips = frame_skips
        self.waypoint_skips = waypoint_skips
        self.model_context = model_context

        base_path = '/Users/tom/Development/carbot_ai_model/processed_trajectories'
        self.data_folder = f'{base_path}/{data_folder}'

        self.load_trajectories()

    def load_trajectories(self):
        for trajectory in os.listdir(self.data_folder):
            trajectory_path = f'{self.data_folder}/{trajectory}'
            sample = TrajectorySample(trajectory_path, 
                                    model_context=self.model_context, 
                                    transform=self.transform, 
                                    frame_skips=self.frame_skips, 
                                    waypoint_skips=self.waypoint_skips)
            self.trajectories.append(sample)
            self.len += len(sample)

    def map_index_to_trajectory(self, idx):
        cumulative_length = 0
        for traj_index, trajectory in enumerate(self.trajectories):
            traj_length = len(trajectory)
            if idx < cumulative_length + traj_length:
                sample_index = idx - cumulative_length
                return traj_index, sample_index
            cumulative_length += traj_length
        raise IndexError("Index out of bounds")
    
    def __getitem__(self, idx):
        traj_index, sample_index = self.map_index_to_trajectory(idx)
        return self.trajectories[traj_index][sample_index]
    
    def __len__(self):
        return self.len
    
    def denormalize_image(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
    
    def view_sample(self, idx, num_prev_observations=0):
        o_t, waypoints = self.__getitem__(idx)

        if num_prev_observations > self.model_context.num_prev_observations:
            num_prev_observations = self.model_context.num_prev_observations

        o_t = o_t.to('cpu')[0, :, :, :]
        o_t = self.denormalize_image(o_t)
        o_t = o_t.permute(1, 2, 0).numpy()

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(o_t)
        ax[0].set_title(f'Observation Image, Sample Index: {idx}, Previous Observations: {num_prev_observations}')
        ax[0].axis('off')
        ax[1].set_title('Waypoints in Robot-Centric Frame')

        xy = waypoints[:, :2]
        x = xy[:, 0].cpu().numpy()
        y = xy[:, 1].cpu().numpy()

        y = -y

        max_y = max(abs(y)) + 0.5
        max_x = max(abs(x)) + 2.0

        ax[1].plot(y, x, marker='o')
        ax[1].set_xlim(-max_y, max_y)
        ax[1].set_ylim(0, max_x)
        ax[1].set_aspect('equal', adjustable='box')
        plt.show()

def plot_waypoints(waypoints):
    xy = waypoints[:, :2]
    x = xy[:, 0].cpu().numpy()
    y = xy[:, 1].cpu().numpy()

    # Axis scale
    max_value = max(max(abs(x)), max(abs(y))) + 0.5

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, marker='o')

    # Label points 1–8
    for i in range(xy.shape[0]):
        plt.text(x[i], y[i], str(i + 1), fontsize=12, ha='right', va='bottom')

    # Formatting
    plt.xlim(-max_value, max_value)
    plt.ylim(-max_value, max_value)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Waypoints in Robot-Centric Frame')
    plt.grid()
    plt.show()