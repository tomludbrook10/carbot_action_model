import torch
import numpy as np

class ModelContext():
    def __init__(self, 
                 channels = 3,
                 width = 256,
                 height = 256,
                 mean = [0.485, 0.456, 0.406],
                 std=[0.225, 0.225, 0.225],
                 levels = [64, 128, 256],
                 hidden_dim = 512,
                 waypoint_num = 8,
                 data_type = torch.float32,
                 data_type_np = np.float32
                 ):
          self.mean = mean
          self.std = std
          self.waypoint_num = waypoint_num
          self.data_type = data_type
          self.data_type_np = data_type_np
          self.channels = channels
          self.width = width
          self.height = height
          self.device = torch.device('mps' if torch.mps.is_available() else 'cpu')
          self.levels = levels
          self.hidden_dim = hidden_dim


class ModelContextV2():
    def __init__(self, 
                 channels = 3,
                 width = 256,
                 height = 256,
                 mean = [0.485, 0.456, 0.406],
                 std=[0.225, 0.225, 0.225],
                 levels = [64, 128, 256],
                 hidden_dim = 512,
                 waypoint_num = 8,
                 data_type = torch.float32,
                 data_type_np = np.float32,
                 num_prev_observations = 4,
                 embedding_dim = 512,
                 attention_blocks = 2
                ):
          self.mean = mean
          self.std = std
          self.waypoint_num = waypoint_num
          self.data_type = data_type
          self.data_type_np = data_type_np
          self.channels = channels
          self.width = width
          self.height = height
          self.device = torch.device('mps' if torch.mps.is_available() else 'cpu')
          self.levels = levels
          self.hidden_dim = hidden_dim
          self.num_prev_observations = num_prev_observations
          self.embedding_dim = embedding_dim
          self.attention_blocks = attention_blocks

          print('Using device:', self.device)
