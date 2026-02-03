import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from itertools import permutations
from itertools import product

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
from tqdm import tqdm

import lanelet2
from lanelet2.projection import UtmProjector

from utils import compute_angles_lengths_2D
from utils import transform_point_to_local_coordinate
from utils import get_index_of_A_in_B
from utils import wrap_angle
from utils import compute_vehicle_endpoints


class INTERACTIONDataset(Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 num_historical_steps: int = 20,
                 num_future_steps: int = 30,
                 margin: float = 50) -> None:
        self.root = root

        if split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test'
        else:
            raise ValueError(split + ' is not valid')
        
        self._raw_file_names = [f for f in os.listdir(self.raw_dir) if f.endswith('.csv')]
        self._processed_file_names = []
        
        for raw_file_name in tqdm(self._raw_file_names, desc='Scanning raw files'):
            raw_path = os.path.join(self.raw_dir, raw_file_name)
            scenario_name = os.path.splitext(raw_file_name)[0]
            df = pd.read_csv(raw_path)
            for case_id in df['case_id'].unique():
                self._processed_file_names.append(scenario_name + '_' + str(int(case_id)) + '.pt')
        
        self._processed_paths = [os.path.join(self.processed_dir, name) for name in self._processed_file_names]
        
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = num_historical_steps + num_future_steps
        self.margin = margin

        # Use UtmProjector with origin (0, 0) since coordinates are already in x,y system
        self.projector = lanelet2.io.Origin(0, 0)
        self.traffic_rules = lanelet2.traffic_rules.create(
            lanelet2.traffic_rules.Locations.Germany,
            lanelet2.traffic_rules.Participants.Vehicle
        )

        self._agent_type = ['agent', 'others']
        self._polyline_side = ['left', 'center', 'right']
        
        super(INTERACTIONDataset, self).__init__(root=root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'processed_data')
    
    @property
    def map_dir(self) -> str:
        return os.path.join(self.root, 'maps')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def process(self) -> None:
        for raw_file_name in tqdm(self._raw_file_names, desc='Processing files'):
            raw_path = os.path.join(self.raw_dir, raw_file_name)
            scenario_name = os.path.splitext(raw_file_name)[0]
            
            # Load map
            map_path = os.path.join(self.map_dir, scenario_name + '.osm')
            # Load without coordinate transformation since already in x,y
            projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0, 0))
            map_api = lanelet2.io.load(map_path, projector)
            routing_graph = lanelet2.routing.RoutingGraph(map_api, self.traffic_rules)
            
            # Load trajectory data
            df = pd.read_csv(raw_path)
            
            for case_id, case_df in tqdm(df.groupby('case_id'), desc=f'Processing {scenario_name}', leave=False):
                data = dict()
                data['scenario_name'] = scenario_name
                data['case_id'] = int(case_id)
                data.update(self.get_features(case_df, map_api, routing_graph))
                torch.save(data, os.path.join(self.processed_dir, scenario_name + '_' + str(int(case_id)) + '.pt'))

    def get_features(self,
                     df: pd.DataFrame,
                     map_api,
                     routing_graph) -> Dict:
        data = {
            'agent': {},
            'lane': {},
            'centerline': {},
            ('centerline', 'lane'): {},
            ('lane', 'lane'): {}
        }
        
        # AGENT
        # Filter out actors that are unseen during the historical time steps
        frame_ids = list(np.sort(df['frame_id'].unique()))
        historical_frames = frame_ids[:self.num_historical_steps]
        historical_df = df[df['frame_id'].isin(historical_frames)]
        agent_ids = list(historical_df['track_id'].unique())
        num_agents = len(agent_ids)
        df = df[df['track_id'].isin(agent_ids)]
        
        # Find the agent index (the one with agent_type == 'agent')
        agent_df = df[df['agent_type'] == 'agent']
        if len(agent_df) > 0:
            agent_track_id = agent_df['track_id'].values[0]
            agent_index = agent_ids.index(agent_track_id)
        else:
            # If no 'agent' type found, use first track
            agent_index = 0

        # Initialization
        visible_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        length_mask = torch.zeros(num_agents, self.num_historical_steps, dtype=torch.bool)
        agent_position = torch.zeros(num_agents, self.num_steps, 2, dtype=torch.float)
        agent_heading = torch.zeros(num_agents, self.num_steps, dtype=torch.float)  # Extended to full steps for endpoint calculation
        agent_length = torch.zeros(num_agents, self.num_historical_steps, dtype=torch.float)  # velocity magnitude
        vehicle_length = torch.zeros(num_agents, dtype=torch.float)
        vehicle_width = torch.zeros(num_agents, dtype=torch.float)
        velocity_theta = torch.zeros(num_agents, self.num_historical_steps, dtype=torch.float)
        
        # Vehicle endpoint related data
        front_midpoint = torch.zeros(num_agents, self.num_steps, 2, dtype=torch.float)
        rear_midpoint = torch.zeros(num_agents, self.num_steps, 2, dtype=torch.float)
        vlength = torch.zeros(num_agents, dtype=torch.float)  # length/2
        
        for track_id, track_df in df.groupby('track_id'):
            agent_idx = agent_ids.index(track_id)
            agent_steps = [frame_ids.index(frame_id) for frame_id in track_df['frame_id']]
            
            visible_mask[agent_idx, agent_steps] = True
            
            length_mask[agent_idx, 0] = False
            length_mask[agent_idx, 1:] = ~(visible_mask[agent_idx, 1:self.num_historical_steps] & 
                                           visible_mask[agent_idx, :self.num_historical_steps-1])
            
            agent_position[agent_idx, agent_steps] = torch.from_numpy(
                np.stack([track_df['x'].values, track_df['y'].values], axis=-1)
            ).float()
            
            # Get heading from psi_rad for all steps
            if 'psi_rad' in track_df.columns:
                psi_rad_values = track_df['psi_rad'].values
                for i, step in enumerate(agent_steps):
                    agent_heading[agent_idx, step] = psi_rad_values[i]
            
            # Compute velocity magnitude (agent_length) from vx, vy
            if 'vx' in track_df.columns and 'vy' in track_df.columns:
                vx = track_df['vx'].values
                vy = track_df['vy'].values
                velocity_magnitude = np.sqrt(vx**2 + vy**2)
                velocity_direction = np.arctan2(vy, vx)
                
                for i, step in enumerate(agent_steps):
                    if step < self.num_historical_steps:
                        agent_length[agent_idx, step] = velocity_magnitude[i]
                        # velocity_theta: angle between velocity direction and heading
                        if 'psi_rad' in track_df.columns:
                            velocity_theta[agent_idx, step] = wrap_angle(
                                torch.tensor(velocity_direction[i] - psi_rad_values[i])
                            )
            
            # Get vehicle dimensions
            if 'length' in track_df.columns:
                vehicle_length[agent_idx] = track_df['length'].values[0]
                vlength[agent_idx] = track_df['length'].values[0] / 2.0
            if 'width' in track_df.columns:
                vehicle_width[agent_idx] = track_df['width'].values[0]
            
            # Handle missing heading - compute from motion if needed
            agent_length[agent_idx, length_mask[agent_idx]] = 0
            velocity_theta[agent_idx, length_mask[agent_idx]] = 0

        # Compute front and rear midpoints for all agents and all steps
        for agent_idx in range(num_agents):
            half_len = vlength[agent_idx].item()
            for step in range(self.num_steps):
                if visible_mask[agent_idx, step]:
                    pos = agent_position[agent_idx, step]
                    heading = agent_heading[agent_idx, step]
                    cos_h = torch.cos(heading)
                    sin_h = torch.sin(heading)
                    
                    # Front midpoint
                    front_midpoint[agent_idx, step, 0] = pos[0] + half_len * cos_h
                    front_midpoint[agent_idx, step, 1] = pos[1] + half_len * sin_h
                    
                    # Rear midpoint
                    rear_midpoint[agent_idx, step, 0] = pos[0] - half_len * cos_h
                    rear_midpoint[agent_idx, step, 1] = pos[1] - half_len * sin_h

        data['agent']['num_nodes'] = num_agents
        data['agent']['agent_index'] = agent_index
        data['agent']['visible_mask'] = visible_mask
        data['agent']['position'] = agent_position
        data['agent']['heading'] = agent_heading[:, :self.num_historical_steps]  # Historical steps for compatibility
        data['agent']['heading_full'] = agent_heading  # Full heading for endpoint calculations
        data['agent']['length'] = agent_length  # velocity magnitude
        data['agent']['vehicle_length'] = vehicle_length
        data['agent']['vehicle_width'] = vehicle_width
        data['agent']['velocity_theta'] = velocity_theta
        
        # Vehicle endpoint data
        data['agent']['front_midpoint'] = front_midpoint
        data['agent']['rear_midpoint'] = rear_midpoint
        data['agent']['vlength'] = vlength  # length/2 for each agent
        
        # MAP
        # Get lane information from Lanelet2 map
        lane_ids = []
        for lane in map_api.laneletLayer:
            lane_ids.append(lane.id)
        
        num_lanes = len(lane_ids)
        
        if num_lanes == 0:
            # Handle empty map case
            data['lane']['num_nodes'] = 0
            data['lane']['position'] = torch.zeros(0, 2, dtype=torch.float)
            data['lane']['length'] = torch.zeros(0, dtype=torch.float)
            data['lane']['heading'] = torch.zeros(0, dtype=torch.float)
            data['lane']['is_intersection'] = torch.zeros(0, dtype=torch.uint8)
            data['lane']['turn_direction'] = torch.zeros(0, dtype=torch.uint8)
            data['lane']['traffic_control'] = torch.zeros(0, dtype=torch.uint8)
            data['centerline']['num_nodes'] = 0
            data['centerline']['position'] = torch.zeros(0, 2, dtype=torch.float)
            data['centerline']['heading'] = torch.zeros(0, dtype=torch.float)
            data['centerline']['length'] = torch.zeros(0, dtype=torch.float)
            data['centerline', 'lane']['centerline_to_lane_edge_index'] = torch.tensor([[], []], dtype=torch.long)
            data['lane', 'lane']['left_neighbor_edge_index'] = torch.tensor([[], []], dtype=torch.long)
            data['lane', 'lane']['right_neighbor_edge_index'] = torch.tensor([[], []], dtype=torch.long)
            data['lane', 'lane']['predecessor_edge_index'] = torch.tensor([[], []], dtype=torch.long)
            data['lane', 'lane']['successor_edge_index'] = torch.tensor([[], []], dtype=torch.long)
            return data
        
        lane_position = torch.zeros(num_lanes, 2, dtype=torch.float)
        lane_heading = torch.zeros(num_lanes, dtype=torch.float)
        lane_length = torch.zeros(num_lanes, dtype=torch.float)
        lane_is_intersection = torch.zeros(num_lanes, dtype=torch.uint8)
        lane_turn_direction = torch.zeros(num_lanes, dtype=torch.uint8)
        lane_traffic_control = torch.zeros(num_lanes, dtype=torch.uint8)
        
        num_centerlines = torch.zeros(num_lanes, dtype=torch.long)
        centerline_position: List[Optional[torch.Tensor]] = [None] * num_lanes
        centerline_heading: List[Optional[torch.Tensor]] = [None] * num_lanes
        centerline_length: List[Optional[torch.Tensor]] = [None] * num_lanes
        
        lane_left_neighbor_edge_index = []
        lane_right_neighbor_edge_index = []
        lane_predecessor_edge_index = []
        lane_successor_edge_index = []
        
        for lane in map_api.laneletLayer:
            lane_idx = lane_ids.index(lane.id)
            
            # Get centerline points (coordinates are already in x,y)
            points = [np.array([pt.x, pt.y]) for pt in lane.centerline]
            centerlines = torch.from_numpy(np.array(points)).float()
            
            if centerlines.size(0) < 2:
                # Handle single point case
                num_centerlines[lane_idx] = 0
                centerline_position[lane_idx] = torch.zeros(0, 2, dtype=torch.float)
                centerline_heading[lane_idx] = torch.zeros(0, dtype=torch.float)
                centerline_length[lane_idx] = torch.zeros(0, dtype=torch.float)
                lane_position[lane_idx] = centerlines[0, :2] if centerlines.size(0) > 0 else torch.zeros(2)
                lane_heading[lane_idx] = 0
                lane_length[lane_idx] = 0
                continue
            
            num_centerlines[lane_idx] = centerlines.size(0) - 1
            centerline_position[lane_idx] = (centerlines[1:] + centerlines[:-1]) / 2
            centerline_vectors = centerlines[1:] - centerlines[:-1]
            centerline_length[lane_idx], centerline_heading[lane_idx] = compute_angles_lengths_2D(centerline_vectors)
            
            lane_length[lane_idx] = centerline_length[lane_idx].sum()
            center_index = int(num_centerlines[lane_idx] / 2)
            lane_position[lane_idx] = centerlines[center_index]
            lane_heading[lane_idx] = torch.atan2(
                centerlines[center_index + 1, 1] - centerlines[center_index, 1],
                centerlines[center_index + 1, 0] - centerlines[center_index, 0]
            )
            
            # Check for intersection (simplified - based on lane attributes if available)
            lane_is_intersection[lane_idx] = 0
            lane_turn_direction[lane_idx] = 0
            lane_traffic_control[lane_idx] = 0
            
            # Get topology from routing graph
            lane_left_neighbor_lane = routing_graph.left(lane)
            lane_left_neighbor_id = [lane_left_neighbor_lane.id] if lane_left_neighbor_lane else []
            lane_left_neighbor_idx = get_index_of_A_in_B(lane_left_neighbor_id, lane_ids)
            if len(lane_left_neighbor_idx) != 0:
                edge_index = torch.stack([
                    torch.tensor(lane_left_neighbor_idx, dtype=torch.long),
                    torch.full((len(lane_left_neighbor_idx),), lane_idx, dtype=torch.long)
                ], dim=0)
                lane_left_neighbor_edge_index.append(edge_index)
            
            lane_right_neighbor_lane = routing_graph.right(lane)
            lane_right_neighbor_id = [lane_right_neighbor_lane.id] if lane_right_neighbor_lane else []
            lane_right_neighbor_idx = get_index_of_A_in_B(lane_right_neighbor_id, lane_ids)
            if len(lane_right_neighbor_idx) != 0:
                edge_index = torch.stack([
                    torch.tensor(lane_right_neighbor_idx, dtype=torch.long),
                    torch.full((len(lane_right_neighbor_idx),), lane_idx, dtype=torch.long)
                ], dim=0)
                lane_right_neighbor_edge_index.append(edge_index)
            
            lane_predecessor_lanes = routing_graph.previous(lane)
            lane_predecessor_ids = [ll.id for ll in lane_predecessor_lanes] if lane_predecessor_lanes else []
            lane_predecessor_idx = get_index_of_A_in_B(lane_predecessor_ids, lane_ids)
            if len(lane_predecessor_idx) != 0:
                edge_index = torch.stack([
                    torch.tensor(lane_predecessor_idx, dtype=torch.long),
                    torch.full((len(lane_predecessor_idx),), lane_idx, dtype=torch.long)
                ], dim=0)
                lane_predecessor_edge_index.append(edge_index)
            
            lane_successor_lanes = routing_graph.following(lane)
            lane_successor_ids = [ll.id for ll in lane_successor_lanes] if lane_successor_lanes else []
            lane_successor_idx = get_index_of_A_in_B(lane_successor_ids, lane_ids)
            if len(lane_successor_idx) != 0:
                edge_index = torch.stack([
                    torch.tensor(lane_successor_idx, dtype=torch.long),
                    torch.full((len(lane_successor_idx),), lane_idx, dtype=torch.long)
                ], dim=0)
                lane_successor_edge_index.append(edge_index)
        
        data['lane']['num_nodes'] = num_lanes
        data['lane']['position'] = lane_position
        data['lane']['length'] = lane_length
        data['lane']['heading'] = lane_heading
        data['lane']['is_intersection'] = lane_is_intersection
        data['lane']['turn_direction'] = lane_turn_direction
        data['lane']['traffic_control'] = lane_traffic_control
        
        # Filter out None values for centerlines
        valid_centerline_position = [cp for cp in centerline_position if cp is not None and cp.size(0) > 0]
        valid_centerline_heading = [ch for ch in centerline_heading if ch is not None and ch.size(0) > 0]
        valid_centerline_length = [cl for cl in centerline_length if cl is not None and cl.size(0) > 0]
        
        if len(valid_centerline_position) > 0:
            data['centerline']['num_nodes'] = num_centerlines.sum().item()
            data['centerline']['position'] = torch.cat(valid_centerline_position, dim=0)
            data['centerline']['heading'] = torch.cat(valid_centerline_heading, dim=0)
            data['centerline']['length'] = torch.cat(valid_centerline_length, dim=0)
        else:
            data['centerline']['num_nodes'] = 0
            data['centerline']['position'] = torch.zeros(0, 2, dtype=torch.float)
            data['centerline']['heading'] = torch.zeros(0, dtype=torch.float)
            data['centerline']['length'] = torch.zeros(0, dtype=torch.float)
        
        if data['centerline']['num_nodes'] > 0:
            centerline_to_lane_edge_index = torch.stack([
                torch.arange(num_centerlines.sum(), dtype=torch.long),
                torch.arange(num_lanes, dtype=torch.long).repeat_interleave(num_centerlines)
            ], dim=0)
        else:
            centerline_to_lane_edge_index = torch.tensor([[], []], dtype=torch.long)
        data['centerline', 'lane']['centerline_to_lane_edge_index'] = centerline_to_lane_edge_index
        
        if len(lane_left_neighbor_edge_index) != 0:
            lane_left_neighbor_edge_index = torch.cat(lane_left_neighbor_edge_index, dim=1)
        else:
            lane_left_neighbor_edge_index = torch.tensor([[], []], dtype=torch.long)
        
        if len(lane_right_neighbor_edge_index) != 0:
            lane_right_neighbor_edge_index = torch.cat(lane_right_neighbor_edge_index, dim=1)
        else:
            lane_right_neighbor_edge_index = torch.tensor([[], []], dtype=torch.long)
        
        if len(lane_predecessor_edge_index) != 0:
            lane_predecessor_edge_index = torch.cat(lane_predecessor_edge_index, dim=1)
        else:
            lane_predecessor_edge_index = torch.tensor([[], []], dtype=torch.long)
        
        if len(lane_successor_edge_index) != 0:
            lane_successor_edge_index = torch.cat(lane_successor_edge_index, dim=1)
        else:
            lane_successor_edge_index = torch.tensor([[], []], dtype=torch.long)
        
        # Use adjacent edge index combining left and right neighbors for compatibility
        if lane_left_neighbor_edge_index.size(1) > 0 or lane_right_neighbor_edge_index.size(1) > 0:
            adjacent_edge_index = torch.cat([lane_left_neighbor_edge_index, lane_right_neighbor_edge_index], dim=1)
        else:
            adjacent_edge_index = torch.tensor([[], []], dtype=torch.long)
        
        data['lane', 'lane']['adjacent_edge_index'] = adjacent_edge_index
        data['lane', 'lane']['predecessor_edge_index'] = lane_predecessor_edge_index
        data['lane', 'lane']['successor_edge_index'] = lane_successor_edge_index
        
        return data

    def len(self) -> int:
        return len(self._processed_file_names)

    def get(self, idx: int) -> HeteroData:
        return HeteroData(torch.load(self.processed_paths[idx]))
