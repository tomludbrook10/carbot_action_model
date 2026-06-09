import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import display
from ipywidgets import interactive

class Position():
    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw

class Observation():
    def __init__(self, image, ref_position: Position, waypoints):
        self.image = image
        self.ref_position = ref_position
        self.waypoint_in_odom_frame = waypoints

class ReplayModelRollout:
    def __init__(self, 
                 data_dir, 
                 model_context):
        self.data_dir = data_dir
        self.model_context = model_context
        self.observations = []
        self.load_data()

    def compare_predictions(self, model, model_context, index):
        index = index + model_context.num_prev_observations
        obs = self.observations[index]

        obs_images = []
        obs_images.append(obs.image)

        for i in range(index - model_context.num_prev_observations, index + 1):
            obs_images.append(self.observations[i].image)

        input_tensor = torch.stack(obs_images, dim=0).unsqueeze(0).to(model_context.device).to(model_context.data_type)

        print(input_tensor.shape)


    def load_data(self):
        data = pd.read_csv(f'{self.data_dir}/debug_data.csv')
        for index, row in data.iterrows():
            waypoints = []

            ref_position = Position(row['ref_x'], row['ref_y'], row['ref_yaw'])
            for i in range(self.model_context.waypoint_num):
                wp_x = row[f'wp_{i}_x']
                wp_y = row[f'wp_{i}_y']
                waypoints.append(Position(wp_x, wp_y, 0.0))

            # This is done because how we normalise in TensorRT inference engine
            # specifically using the deepstream nvpreprocess plugin with the pixel normalisation
            mean_vals = list(self.model_context.mean)
            for i in range(len(mean_vals)):
                mean_vals[i] = mean_vals[i] * self.model_context.std[i]
            

            array = np.fromfile(f'{self.data_dir}/input_tensor_{index}.bin', dtype=self.model_context.data_type_np)
            tensor = array.reshape(1, self.model_context.channels, self.model_context.height, self.model_context.width)
            mean_np_array = np.array(mean_vals).reshape(1, self.model_context.channels, 1, 1)
            std_np_array = np.array(self.model_context.std).reshape(1, self.model_context.channels, 1, 1)
            x = (tensor[0] * std_np_array) + mean_np_array
            x = x.squeeze(0)
            x = x.clip(0, 1)

            observation = Observation(image=x, ref_position=ref_position, waypoints=waypoints)
            self.observations.append(observation)

    def see_observations_with_slider(self):
        max_y = 0.5
        max_x = 1.0
        def show_frame(index):
            obs = self.observations[index]
            img = obs.image.transpose(1, 2, 0)
            waypoints = obs.waypoint_in_odom_frame

            wp_xs = [wp.x for wp in waypoints]
            wp_ys = [-wp.y for wp in waypoints]

            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            # Left: image
            axs[0].imshow(img)
            axs[0].set_title(f'Observation {index}')
            axs[0].axis('off')

            # Right: waypoints
            axs[1].plot(wp_ys, wp_xs, 'ro-')
            axs[1].set_title('Waypoints in Odom Frame')
            axs[1].set_xlabel('Y (m)')
            axs[1].set_ylabel('X (m)')
            axs[1].set_xlim([-max_y, max_y])
            axs[1].set_ylim([0, max_x])
            plt.show()

        interactive_plot = interactive(show_frame, index=(0, len(self.observations) - 1))
        output = interactive_plot.children[-1]
        output.layout.height = '600px'
        display(interactive_plot)


    def see_observation(self, index):
        obs = self.observations[index]
        img = obs.image.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)

        waypoints = obs.waypoint_in_odom_frame

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        wp_xs = [wp.x for wp in waypoints]
        wp_ys = [-wp.y for wp in waypoints]
        axs[1].plot(wp_ys, wp_xs, 'ro-', label='Waypoints')
        axs[1].set_title('Waypoints in Odom Frame')

        max_y = max(abs(max(wp_ys)), abs(min(wp_ys)))
        axs[1].set_xlim([-max_y, max_y])

        axs[1].set_xlabel('Y (m)')
        axs[1].set_ylabel('X (m)')
        axs[1].legend()

        axs[0].imshow(img)
        axs[0].set_title('Observation Image')
        plt.show()

    def plot_all_waypoints(self):
        fig, ax = plt.subplots(figsize=(8, 8))

        max_y = float('-inf')

        for idx, obs in enumerate(self.observations):
            if (idx < 6 or idx > 25):
                continue    
            waypoints = list(obs.waypoint_in_odom_frame)
            transformed_waypoints = [self.base_to_world(obs.ref_position, wp.x, wp.y) for wp in waypoints]
            wp_xs = [wp[0] for wp in transformed_waypoints]
            wp_ys = [-wp[1] for wp in transformed_waypoints]
            ax.plot(wp_ys, wp_xs, 'ro-')
            max_y = max(max_y, max(abs(min(wp_ys)), abs(max(wp_ys))))

        max_y += 0.5    
        ax.set_xlim([-max_y, max_y])
        ax.set_xlabel('Y (m)')
        ax.set_ylabel('X (m)')        

        ax.set_title('All Waypoints in Odom Frame')
        plt.show()

    def base_to_world(self, position: Position, base_x, base_y):
        import math
        cos_yaw = math.cos(position.yaw)
        sin_yaw = math.sin(position.yaw)

        world_x = position.x + (base_x * cos_yaw - base_y * sin_yaw)
        world_y = position.y + (base_x * sin_yaw + base_y * cos_yaw)

        return (world_x, world_y)