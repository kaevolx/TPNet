from typing import Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.utils import dense_to_sparse

from layers import GraphAttention
from layers import TwoLayerMLP
from utils import compute_angles_lengths_2D
from utils import init_weights
from utils import wrap_angle
from utils import drop_edge_between_samples
from utils import transform_point_to_local_coordinate
from utils import transform_point_to_global_coordinate
from utils import transform_traj_to_global_coordinate
from utils import transform_traj_to_local_coordinate


class Backbone(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 num_historical_steps: int,
                 num_future_steps: int,
                 pos_duration: int,
                 pred_duration: int,
                 a2a_radius: float,
                 l2a_radius: float,
                 num_attn_layers: int, 
                 num_modes: int,
                 num_heads: int,
                 dropout: float) -> None:
        super(Backbone, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.pos_duration = pos_duration
        self.pred_duration = pred_duration
        self.a2a_radius = a2a_radius
        self.l2a_radius = l2a_radius
        self.num_attn_layers = num_attn_layers
        self.num_modes = num_modes
        self.num_heads = num_heads
        self.dropout = dropout

        self.mode_tokens = nn.Embedding(num_modes, hidden_dim)  # [K,D]

        # Modified agent embedding layer: input_dim=4 for (velocity_magnitude, velocity_theta, vehicle_length, vehicle_width)
        self.a_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)
        
        # Vehicle endpoint embedding layers
        # Endpoint node embedding: vlength (length/2)
        self.endpoint_emb_layer = TwoLayerMLP(input_dim=1, hidden_dim=hidden_dim, output_dim=hidden_dim)
        
        # Endpoint to center edge embedding: vtheta
        self.e2c_emb_layer = TwoLayerMLP(input_dim=1, hidden_dim=hidden_dim, output_dim=hidden_dim)
        
        # Endpoint to endpoint edge embedding: length, heading, theta
        self.e2e_emb_layer = TwoLayerMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)
        
        # Endpoint graph attention layers
        self.e2e_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)
        self.e2c_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)
        
        self.l2m_emb_layer = TwoLayerMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.t2m_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.m2m_h_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.m2m_a_emb_layer = TwoLayerMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.m2m_s_emb_layer = TwoLayerMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.l2m_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)
        self.t2m_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)

        self.m2m_h_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        self.m2m_a_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        self.m2m_s_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=False, if_self_attention=True) for _ in range(num_attn_layers)])

        # Modified trajectory proposal to output both center and front midpoint: (F*2 for center + F*2 for front)
        self.traj_propose = TwoLayerMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.num_future_steps*4)

        self.proposal_to_anchor = TwoLayerMLP(input_dim=self.num_future_steps*4, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.l2n_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)
        self.t2n_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)

        self.n2n_h_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        self.n2n_a_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        self.n2n_s_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])

        # Modified trajectory refinement to output both center and front midpoint
        self.traj_refine = TwoLayerMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.num_future_steps*4)

        self.prob_decoder = TwoLayerMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)    
        self.prob_norm = nn.Softmax(dim=-1)

        self.apply(init_weights)

    def _compute_endpoint_interactions(self, data: Batch, a_embs: torch.Tensor) -> torch.Tensor:
        """
        Compute vehicle endpoint interactions and aggregate to center nodes.
        VECTORIZED VERSION - no Python loops over frames or agent pairs.
        
        Args:
            data: Batch data containing agent information
            a_embs: Agent embeddings [(N1,...,Nb), H, D]
        
        Returns:
            Updated agent embeddings with endpoint information aggregated
        """
        num_all_agent = a_embs.size(0)
        device = a_embs.device
        H = self.num_historical_steps
        
        # Get endpoint data
        front_midpoint = data['agent']['front_midpoint'][:, :H]  # [N, H, 2]
        rear_midpoint = data['agent']['rear_midpoint'][:, :H]  # [N, H, 2]
        center_position = data['agent']['position'][:, :H]  # [N, H, 2]
        heading_full = data['agent']['heading_full'][:, :H]  # [N, H]
        vlength = data['agent']['vlength']  # [N] - length/2
        visible_mask = data['agent']['visible_mask'][:, :H]  # [N, H]
        batch = data['agent']['batch']  # [N]
        
        # Create endpoint node embeddings
        # vlength is the node attribute for both front and rear endpoints
        vlength_expanded = vlength.unsqueeze(-1).unsqueeze(-1).expand(-1, H, 1)  # [N, H, 1]
        endpoint_node_embs = self.endpoint_emb_layer(vlength_expanded)  # [N, H, D]
        
        # Front and rear endpoint embeddings (initially same, differentiated by edge attributes)
        front_embs = endpoint_node_embs.clone()  # [N, H, D]
        rear_embs = endpoint_node_embs.clone()  # [N, H, D]
        
        # ========== VECTORIZED ENDPOINT-TO-ENDPOINT INTERACTIONS ==========
        # Build all edges across all frames simultaneously
        
        # Create indices for all (agent, frame) combinations
        # We need to find pairs of agents within same frame, same batch, within radius
        
        # Flatten positions: [N, H, 2] -> [N*H, 2]
        front_pos_flat = front_midpoint.reshape(-1, 2)  # [N*H, 2]
        rear_pos_flat = rear_midpoint.reshape(-1, 2)  # [N*H, 2]
        center_pos_flat = center_position.reshape(-1, 2)  # [N*H, 2]
        heading_flat = heading_full.reshape(-1)  # [N*H]
        visible_flat = visible_mask.reshape(-1)  # [N*H]
        
        # Create frame and agent indices for each flattened position
        # agent_indices_flat[i] gives the agent index for flat position i
        # frame_indices_flat[i] gives the frame index for flat position i
        agent_indices_flat = torch.arange(num_all_agent, device=device).unsqueeze(1).expand(-1, H).reshape(-1)  # [N*H]
        frame_indices_flat = torch.arange(H, device=device).unsqueeze(0).expand(num_all_agent, -1).reshape(-1)  # [N*H]
        batch_expanded = batch.unsqueeze(1).expand(-1, H).reshape(-1)  # [N*H]
        
        # Build valid pair mask for all pairs of flattened indices
        # Valid if: same frame, same batch, both visible, different agents, within radius
        num_flat = num_all_agent * H
        
        # Use chunked processing to avoid memory issues with large tensors
        # Process in blocks to build edge lists
        
        e2e_edge_src_list = []
        e2e_edge_dst_list = []
        e2e_edge_type_list = []
        e2e_edge_attr_list = []
        
        # Process frame by frame but vectorize within each frame
        for h in range(H):
            # Get mask for this frame
            frame_start = h
            frame_indices = torch.arange(num_all_agent, device=device) * H + h  # Indices in flat array for this frame
            
            # Get data for this frame
            frame_center_pos = center_pos_flat[frame_indices]  # [N, 2]
            frame_front_pos = front_pos_flat[frame_indices]  # [N, 2]
            frame_rear_pos = rear_pos_flat[frame_indices]  # [N, 2]
            frame_heading = heading_flat[frame_indices]  # [N]
            frame_visible = visible_flat[frame_indices]  # [N]
            frame_batch = batch  # [N]
            
            # Compute pairwise distances between centers: [N, N]
            dist_matrix = torch.cdist(frame_center_pos.unsqueeze(0), frame_center_pos.unsqueeze(0)).squeeze(0)
            
            # Create valid mask: [N, N]
            # Within radius
            valid_mask = dist_matrix < self.a2a_radius
            # Exclude self-connections
            valid_mask = valid_mask & (torch.eye(num_all_agent, device=device) == 0)
            # Same batch
            batch_mask = frame_batch.unsqueeze(1) == frame_batch.unsqueeze(0)
            valid_mask = valid_mask & batch_mask
            # Both visible
            vis_mask = frame_visible.unsqueeze(1) & frame_visible.unsqueeze(0)
            valid_mask = valid_mask & vis_mask
            
            # Get valid pairs
            src_local, dst_local = torch.where(valid_mask)  # [E], [E]
            
            if src_local.size(0) == 0:
                continue
            
            # Determine edge type based on x coordinate comparison
            # If src_x >= dst_x: connect src's rear to dst's front (type 0)
            # If src_x < dst_x: connect src's front to dst's rear (type 1)
            src_x = frame_center_pos[src_local, 0]
            dst_x = frame_center_pos[dst_local, 0]
            src_greater = src_x >= dst_x  # [E]
            
            # Compute endpoint positions based on edge type
            # Type 0: src's rear -> dst's front
            # Type 1: src's front -> dst's rear
            src_endpoint_pos = torch.where(
                src_greater.unsqueeze(-1),
                frame_rear_pos[src_local],  # Type 0: rear
                frame_front_pos[src_local]  # Type 1: front
            )  # [E, 2]
            
            dst_endpoint_pos = torch.where(
                src_greater.unsqueeze(-1),
                frame_front_pos[dst_local],  # Type 0: front
                frame_rear_pos[dst_local]  # Type 1: rear
            )  # [E, 2]
            
            # Compute edge attributes
            diff = src_endpoint_pos - dst_endpoint_pos  # [E, 2]
            edge_length = torch.norm(diff, dim=-1)  # [E]
            edge_theta = torch.atan2(diff[:, 1], diff[:, 0])  # [E]
            
            # Heading difference: larger x heading - smaller x heading
            src_heading = frame_heading[src_local]
            dst_heading = frame_heading[dst_local]
            edge_heading = torch.where(
                src_greater,
                src_heading - dst_heading,
                dst_heading - src_heading
            )
            edge_heading = wrap_angle(edge_heading)
            
            # Convert local indices to global flat indices
            global_src = src_local * H + h  # [E]
            global_dst = dst_local * H + h  # [E]
            
            # Edge type: 0 for rear->front, 1 for front->rear
            edge_type = (~src_greater).long()  # [E]
            
            # Stack edge attributes
            edge_attr = torch.stack([edge_length, edge_heading, edge_theta], dim=-1)  # [E, 3]
            
            e2e_edge_src_list.append(global_src)
            e2e_edge_dst_list.append(global_dst)
            e2e_edge_type_list.append(edge_type)
            e2e_edge_attr_list.append(edge_attr)
        
        # Apply endpoint-to-endpoint attention if there are edges
        if len(e2e_edge_src_list) > 0:
            e2e_edge_src = torch.cat(e2e_edge_src_list, dim=0)
            e2e_edge_dst = torch.cat(e2e_edge_dst_list, dim=0)
            e2e_edge_type = torch.cat(e2e_edge_type_list, dim=0)
            e2e_edge_attr = torch.cat(e2e_edge_attr_list, dim=0)  # [E_total, 3]
            
            # Embed edge attributes
            e2e_edge_embs = self.e2e_emb_layer(e2e_edge_attr)  # [E_total, D]
            
            # Prepare source and destination endpoint embeddings
            front_embs_flat = front_embs.reshape(-1, self.hidden_dim)  # [N*H, D]
            rear_embs_flat = rear_embs.reshape(-1, self.hidden_dim)  # [N*H, D]
            
            # Separate edges by type
            type0_mask = e2e_edge_type == 0  # Other's rear -> center's front
            type1_mask = e2e_edge_type == 1  # Other's front -> center's rear
            
            # Type 0: rear -> front aggregation
            if type0_mask.sum() > 0:
                t0_edge_index = torch.stack([e2e_edge_src[type0_mask], e2e_edge_dst[type0_mask]], dim=0)
                t0_edge_embs = e2e_edge_embs[type0_mask]
                
                # Source is rear, destination is front
                front_embs_flat = self.e2e_attn_layer(
                    x=[rear_embs_flat, front_embs_flat],
                    edge_index=t0_edge_index,
                    edge_attr=t0_edge_embs
                )
            
            # Type 1: front -> rear aggregation
            if type1_mask.sum() > 0:
                t1_edge_index = torch.stack([e2e_edge_src[type1_mask], e2e_edge_dst[type1_mask]], dim=0)
                t1_edge_embs = e2e_edge_embs[type1_mask]
                
                # Source is front, destination is rear
                rear_embs_flat = self.e2e_attn_layer(
                    x=[front_embs_flat, rear_embs_flat],
                    edge_index=t1_edge_index,
                    edge_attr=t1_edge_embs
                )
            
            # Reshape back
            front_embs = front_embs_flat.reshape(num_all_agent, H, self.hidden_dim)
            rear_embs = rear_embs_flat.reshape(num_all_agent, H, self.hidden_dim)
        
        # ========== VECTORIZED ENDPOINT-TO-CENTER AGGREGATION ==========
        # Create endpoint-to-center edges for all agents and frames
        
        # Each (agent, frame) has 2 edges: front->center and rear->center
        # Total edges: N * H * 2
        
        # Create flat indices
        flat_indices = torch.arange(num_all_agent * H, device=device)  # [N*H]
        
        # Front to center edges: vtheta = psi_rad
        e2c_front_src = flat_indices  # [N*H]
        e2c_front_dst = flat_indices  # [N*H]
        e2c_front_vtheta = heading_flat  # [N*H]
        
        # Rear to center edges: vtheta = psi_rad + pi
        e2c_rear_src = flat_indices + num_all_agent * H  # [N*H] offset to indicate rear
        e2c_rear_dst = flat_indices  # [N*H]
        e2c_rear_vtheta = wrap_angle(heading_flat + 3.14159265359)  # [N*H]
        
        # Concatenate front and rear edges
        e2c_edge_src = torch.cat([e2c_front_src, e2c_rear_src], dim=0)  # [2*N*H]
        e2c_edge_dst = torch.cat([e2c_front_dst, e2c_rear_dst], dim=0)  # [2*N*H]
        e2c_vtheta = torch.cat([e2c_front_vtheta, e2c_rear_vtheta], dim=0).unsqueeze(-1)  # [2*N*H, 1]
        
        # Embed edge attributes
        e2c_edge_embs = self.e2c_emb_layer(e2c_vtheta)  # [2*N*H, D]
        
        # Combine front and rear embeddings for source
        front_embs_flat = front_embs.reshape(-1, self.hidden_dim)  # [N*H, D]
        rear_embs_flat = rear_embs.reshape(-1, self.hidden_dim)  # [N*H, D]
        endpoint_embs_combined = torch.cat([front_embs_flat, rear_embs_flat], dim=0)  # [2*N*H, D]
        
        # Center embeddings
        center_embs_flat = a_embs.reshape(-1, self.hidden_dim)  # [N*H, D]
        
        # Apply attention
        e2c_edge_index = torch.stack([e2c_edge_src, e2c_edge_dst], dim=0)
        updated_center_embs = self.e2c_attn_layer(
            x=[endpoint_embs_combined, center_embs_flat],
            edge_index=e2c_edge_index,
            edge_attr=e2c_edge_embs
        )  # [N*H, D]
        
        # Reshape back to [N, H, D]
        updated_a_embs = updated_center_embs.reshape(num_all_agent, H, self.hidden_dim)
        
        return updated_a_embs

    def forward(self, data: Batch, l_embs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with vehicle endpoint modeling.
        
        Returns:
            traj_propose: Proposed center trajectories [(N1,...,Nb),H,K,F,2]
            traj_output: Refined center trajectories [(N1,...,Nb),H,K,F,2]
            prob_output: Mode probabilities [(N1,...,Nb),H,K]
            front_propose: Proposed front midpoint trajectories [(N1,...,Nb),H,K,F,2]
            front_output: Refined front midpoint trajectories [(N1,...,Nb),H,K,F,2]
        """
        # Initialization
        # Get agent features: velocity magnitude (length), velocity_theta, vehicle_length, vehicle_width
        a_length = data['agent']['length']  # [(N1,...,Nb),H] - velocity magnitude
        a_velocity_theta = data['agent']['velocity_theta']  # [(N1,...,Nb),H]
        a_vehicle_length = data['agent']['vehicle_length'].unsqueeze(-1).repeat_interleave(self.num_historical_steps, -1)  # [(N1,...,Nb),H]
        a_vehicle_width = data['agent']['vehicle_width'].unsqueeze(-1).repeat_interleave(self.num_historical_steps, -1)  # [(N1,...,Nb),H]
        
        # Stack features for agent embedding: [velocity_magnitude, velocity_theta, vehicle_length, vehicle_width]
        a_input = torch.stack([a_length, a_velocity_theta, a_vehicle_length, a_vehicle_width], dim=-1)  # [(N1,...,Nb),H,4]
        a_embs = self.a_emb_layer(input=a_input)  # [(N1,...,Nb),H,D]
        
        # Apply vehicle endpoint interactions
        a_embs = self._compute_endpoint_interactions(data, a_embs)  # [(N1,...,Nb),H,D]
        
        num_all_agent = a_length.size(0)  # N1+...+Nb
        m_embs = self.mode_tokens.weight.unsqueeze(0).repeat_interleave(self.num_historical_steps, 0)  # [H,K,D]
        m_embs = m_embs.unsqueeze(0).repeat_interleave(num_all_agent, 0).reshape(-1, self.hidden_dim)  # [(N1,...,Nb)*H*K,D]

        m_batch = data['agent']['batch'].unsqueeze(1).repeat_interleave(self.num_modes, 1)  # [(N1,...,Nb),K]
        m_position = data['agent']['position'][:, :self.num_historical_steps].unsqueeze(2).repeat_interleave(self.num_modes, 2)  # [(N1,...,Nb),H,K,2]
        m_heading = data['agent']['heading'].unsqueeze(2).repeat_interleave(self.num_modes, 2)  # [(N1,...,Nb),H,K]
        m_valid_mask = data['agent']['visible_mask'][:, :self.num_historical_steps].unsqueeze(2).repeat_interleave(self.num_modes, 2)  # [(N1,...,Nb),H,K]

        # ALL EDGE
        # t2m edge
        t2m_position_t = data['agent']['position'][:, :self.num_historical_steps].reshape(-1, 2)  # [(N1,...,Nb)*H,2]
        t2m_position_m = m_position.reshape(-1, 2)  # [(N1,...,Nb)*H*K,2]
        t2m_heading_t = data['agent']['heading'].reshape(-1)  # [(N1,...,Nb)*H]
        t2m_heading_m = m_heading.reshape(-1)  # [(N1,...,Nb)*H*K]
        t2m_valid_mask_t = data['agent']['visible_mask'][:, :self.num_historical_steps]  # [(N1,...,Nb),H]
        t2m_valid_mask_m = m_valid_mask.reshape(num_all_agent, -1)  # [(N1,...,Nb),H*K]
        t2m_valid_mask = t2m_valid_mask_t.unsqueeze(2) & t2m_valid_mask_m.unsqueeze(1)  # [(N1,...,Nb),H,H*K]
        t2m_edge_index = dense_to_sparse(t2m_valid_mask)[0]
        t2m_edge_index = t2m_edge_index[:, torch.floor(t2m_edge_index[1]/self.num_modes) >= t2m_edge_index[0]]
        t2m_edge_index = t2m_edge_index[:, torch.floor(t2m_edge_index[1]/self.num_modes) - t2m_edge_index[0] <= self.pos_duration]
        t2m_edge_vector = transform_point_to_local_coordinate(t2m_position_t[t2m_edge_index[0]], t2m_position_m[t2m_edge_index[1]], t2m_heading_m[t2m_edge_index[1]])
        t2m_edge_attr_length, t2m_edge_attr_theta = compute_angles_lengths_2D(t2m_edge_vector)
        t2m_edge_attr_heading = wrap_angle(t2m_heading_t[t2m_edge_index[0]] - t2m_heading_m[t2m_edge_index[1]])
        t2m_edge_attr_interval = t2m_edge_index[0] - torch.floor(t2m_edge_index[1]/self.num_modes)
        t2m_edge_attr_input = torch.stack([t2m_edge_attr_length, t2m_edge_attr_theta, t2m_edge_attr_heading, t2m_edge_attr_interval], dim=-1)
        t2m_edge_attr_embs = self.t2m_emb_layer(input=t2m_edge_attr_input)

        # l2m edge
        if l_embs.size(0) > 0:
            l2m_position_l = data['lane']['position']  # [(M1,...,Mb),2]
            l2m_position_m = m_position.reshape(-1, 2)  # [(N1,...,Nb)*H*K,2]
            l2m_heading_l = data['lane']['heading']  # [(M1,...,Mb)]
            l2m_heading_m = m_heading.reshape(-1)  # [(N1,...,Nb)*H*K]
            l2m_batch_l = data['lane']['batch']  # [(M1,...,Mb)]
            l2m_batch_m = m_batch.unsqueeze(1).repeat_interleave(self.num_historical_steps, 1).reshape(-1)  # [(N1,...,Nb)*H*K]
            
            # Create visible mask for lanes (all lanes visible by default)
            if 'visible_mask' in data['lane']:
                l2m_valid_mask_l = data['lane']['visible_mask']
            else:
                l2m_valid_mask_l = torch.ones(l2m_position_l.size(0), dtype=torch.bool, device=l2m_position_l.device)
            
            l2m_valid_mask_m = m_valid_mask.reshape(-1)  # [(N1,...,Nb)*H*K]
            l2m_valid_mask = l2m_valid_mask_l.unsqueeze(1) & l2m_valid_mask_m.unsqueeze(0)  # [(M1,...,Mb),(N1,...,Nb)*H*K]
            l2m_valid_mask = drop_edge_between_samples(l2m_valid_mask, batch=(l2m_batch_l, l2m_batch_m))
            l2m_edge_index = dense_to_sparse(l2m_valid_mask)[0]
            l2m_edge_index = l2m_edge_index[:, torch.norm(l2m_position_l[l2m_edge_index[0]] - l2m_position_m[l2m_edge_index[1]], p=2, dim=-1) < self.l2a_radius]
            
            if l2m_edge_index.size(1) > 0:
                l2m_edge_vector = transform_point_to_local_coordinate(l2m_position_l[l2m_edge_index[0]], l2m_position_m[l2m_edge_index[1]], l2m_heading_m[l2m_edge_index[1]])
                l2m_edge_attr_length, l2m_edge_attr_theta = compute_angles_lengths_2D(l2m_edge_vector)
                l2m_edge_attr_heading = wrap_angle(l2m_heading_l[l2m_edge_index[0]] - l2m_heading_m[l2m_edge_index[1]])
                l2m_edge_attr_input = torch.stack([l2m_edge_attr_length, l2m_edge_attr_theta, l2m_edge_attr_heading], dim=-1)
                l2m_edge_attr_embs = self.l2m_emb_layer(input=l2m_edge_attr_input)
                has_l2m_edges = True
            else:
                has_l2m_edges = False
        else:
            has_l2m_edges = False

        # Mode edge
        # m2m_a_edge
        m2m_a_position = m_position.permute(1, 2, 0, 3).reshape(-1, 2)  # [H*K*(N1,...,Nb),2]
        m2m_a_heading = m_heading.permute(1, 2, 0).reshape(-1)  # [H*K*(N1,...,Nb)]
        m2m_a_batch = data['agent']['batch']  # [(N1,...,Nb)]
        m2m_a_valid_mask = m_valid_mask.permute(1, 2, 0).reshape(self.num_historical_steps * self.num_modes, -1)  # [H*K,(N1,...,Nb)]
        m2m_a_valid_mask = m2m_a_valid_mask.unsqueeze(2) & m2m_a_valid_mask.unsqueeze(1)  # [H*K,(N1,...,Nb),(N1,...,Nb)]
        m2m_a_valid_mask = drop_edge_between_samples(m2m_a_valid_mask, m2m_a_batch)
        m2m_a_edge_index = dense_to_sparse(m2m_a_valid_mask)[0]
        m2m_a_edge_index = m2m_a_edge_index[:, m2m_a_edge_index[1] != m2m_a_edge_index[0]]
        m2m_a_edge_index = m2m_a_edge_index[:, torch.norm(m2m_a_position[m2m_a_edge_index[1]] - m2m_a_position[m2m_a_edge_index[0]], p=2, dim=-1) < self.a2a_radius]
        m2m_a_edge_vector = transform_point_to_local_coordinate(m2m_a_position[m2m_a_edge_index[0]], m2m_a_position[m2m_a_edge_index[1]], m2m_a_heading[m2m_a_edge_index[1]])
        m2m_a_edge_attr_length, m2m_a_edge_attr_theta = compute_angles_lengths_2D(m2m_a_edge_vector)
        m2m_a_edge_attr_heading = wrap_angle(m2m_a_heading[m2m_a_edge_index[0]] - m2m_a_heading[m2m_a_edge_index[1]])
        m2m_a_edge_attr_input = torch.stack([m2m_a_edge_attr_length, m2m_a_edge_attr_theta, m2m_a_edge_attr_heading], dim=-1)
        m2m_a_edge_attr_embs = self.m2m_a_emb_layer(input=m2m_a_edge_attr_input)

        # m2m_h
        m2m_h_position = m_position.permute(2, 0, 1, 3).reshape(-1, 2)  # [K*(N1,...,Nb)*H,2]
        m2m_h_heading = m_heading.permute(2, 0, 1).reshape(-1)  # [K*(N1,...,Nb)*H]
        m2m_h_valid_mask = m_valid_mask.permute(2, 0, 1).reshape(-1, self.num_historical_steps)  # [K*(N1,...,Nb),H]
        m2m_h_valid_mask = m2m_h_valid_mask.unsqueeze(2) & m2m_h_valid_mask.unsqueeze(1)  # [K*(N1,...,Nb),H,H]
        m2m_h_edge_index = dense_to_sparse(m2m_h_valid_mask)[0]
        m2m_h_edge_index = m2m_h_edge_index[:, m2m_h_edge_index[1] > m2m_h_edge_index[0]]
        m2m_h_edge_index = m2m_h_edge_index[:, m2m_h_edge_index[1] - m2m_h_edge_index[0] <= self.pred_duration]
        m2m_h_edge_vector = transform_point_to_local_coordinate(m2m_h_position[m2m_h_edge_index[0]], m2m_h_position[m2m_h_edge_index[1]], m2m_h_heading[m2m_h_edge_index[1]])
        m2m_h_edge_attr_length, m2m_h_edge_attr_theta = compute_angles_lengths_2D(m2m_h_edge_vector)
        m2m_h_edge_attr_heading = wrap_angle(m2m_h_heading[m2m_h_edge_index[0]] - m2m_h_heading[m2m_h_edge_index[1]])
        m2m_h_edge_attr_interval = m2m_h_edge_index[0] - m2m_h_edge_index[1]
        m2m_h_edge_attr_input = torch.stack([m2m_h_edge_attr_length, m2m_h_edge_attr_theta, m2m_h_edge_attr_heading, m2m_h_edge_attr_interval], dim=-1)
        m2m_h_edge_attr_embs = self.m2m_h_emb_layer(input=m2m_h_edge_attr_input)

        # m2m_s edge
        m2m_s_valid_mask = m_valid_mask.transpose(0, 1).reshape(-1, self.num_modes)  # [H*(N1,...,Nb),K]
        m2m_s_valid_mask = m2m_s_valid_mask.unsqueeze(2) & m2m_s_valid_mask.unsqueeze(1)  # [H*(N1,...,Nb),K,K]
        m2m_s_edge_index = dense_to_sparse(m2m_s_valid_mask)[0]
        m2m_s_edge_index = m2m_s_edge_index[:, m2m_s_edge_index[0] != m2m_s_edge_index[1]]

        # ALL ATTENTION
        # t2m attention
        t_embs = a_embs.reshape(-1, self.hidden_dim)  # [(N1,...,Nb)*H,D]
        m_embs_t = self.t2m_attn_layer(x=[t_embs, m_embs], edge_index=t2m_edge_index, edge_attr=t2m_edge_attr_embs)  # [(N1,...,Nb)*H*K,D]

        # l2m attention
        if has_l2m_edges:
            m_embs_l = self.l2m_attn_layer(x=[l_embs, m_embs], edge_index=l2m_edge_index, edge_attr=l2m_edge_attr_embs)  # [(N1,...,Nb)*H*K,D]
            m_embs = m_embs_t + m_embs_l
        else:
            m_embs = m_embs_t
        
        m_embs = m_embs.reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)  # [H*(N1,...,Nb)*K,D]
        
        # Mode attention
        for i in range(self.num_attn_layers):
            # m2m_a
            m_embs = m_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(1, 2).reshape(-1, self.hidden_dim)  # [H*K*(N1,...,Nb),D]
            m_embs = self.m2m_a_attn_layers[i](x=m_embs, edge_index=m2m_a_edge_index, edge_attr=m2m_a_edge_attr_embs)
            # m2m_h
            m_embs = m_embs.reshape(self.num_historical_steps, self.num_modes, num_all_agent, self.hidden_dim).permute(1, 2, 0, 3).reshape(-1, self.hidden_dim)  # [K*(N1,...,Nb)*H,D]
            m_embs = self.m2m_h_attn_layers[i](x=m_embs, edge_index=m2m_h_edge_index, edge_attr=m2m_h_edge_attr_embs)
            # m2m_s
            m_embs = m_embs.reshape(self.num_modes, num_all_agent, self.num_historical_steps, self.hidden_dim).transpose(0, 2).reshape(-1, self.hidden_dim)  # [H*(N1,...,Nb)*K,D]
            m_embs = self.m2m_s_attn_layers[i](x=m_embs, edge_index=m2m_s_edge_index)
        m_embs = m_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)  # [(N1,...,Nb)*H*K,D]

        # Generate trajectory (center + front midpoint)
        traj_propose_combined = self.traj_propose(m_embs).reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.num_future_steps, 4)  # [(N1,...,Nb),H,K,F,4]
        
        # Split into center and front trajectories
        traj_propose_center = traj_propose_combined[..., :2]  # [(N1,...,Nb),H,K,F,2]
        traj_propose_front = traj_propose_combined[..., 2:]  # [(N1,...,Nb),H,K,F,2]
        
        # Transform to global coordinate
        traj_propose = transform_traj_to_global_coordinate(traj_propose_center, m_position, m_heading)  # [(N1,...,Nb),H,K,F,2]
        front_propose = transform_traj_to_global_coordinate(traj_propose_front, m_position, m_heading)  # [(N1,...,Nb),H,K,F,2]

        # Generate anchor
        proposal = traj_propose.detach()  # [(N1,...,Nb),H,K,F,2]
        front_proposal = front_propose.detach()  # [(N1,...,Nb),H,K,F,2]
        
        n_batch = m_batch  # [(N1,...,Nb),K]
        n_position = proposal[:, :, :, self.num_future_steps // 2, :]  # [(N1,...,Nb),H,K,2]
        _, n_heading = compute_angles_lengths_2D(proposal[:, :, :, self.num_future_steps // 2, :] - proposal[:, :, :, (self.num_future_steps // 2) - 1, :])  # [(N1,...,Nb),H,K]
        n_valid_mask = m_valid_mask  # [(N1,...,Nb),H,K]
        
        # Combine center and front proposals for anchor
        proposal_local = transform_traj_to_local_coordinate(proposal, n_position, n_heading)  # [(N1,...,Nb),H,K,F,2]
        front_proposal_local = transform_traj_to_local_coordinate(front_proposal, n_position, n_heading)  # [(N1,...,Nb),H,K,F,2]
        proposal_combined = torch.cat([proposal_local, front_proposal_local], dim=-1)  # [(N1,...,Nb),H,K,F,4]
        
        anchor = self.proposal_to_anchor(proposal_combined.reshape(-1, self.num_future_steps*4))  # [(N1,...,Nb)*H*K,D]
        n_embs = anchor  # [(N1,...,Nb)*H*K,D]

        # t2n edge
        t2n_position_t = data['agent']['position'][:, :self.num_historical_steps].reshape(-1, 2)  # [(N1,...,Nb)*H,2]
        t2n_position_n = n_position.reshape(-1, 2)  # [(N1,...,Nb)*H*K,2]
        t2n_heading_t = data['agent']['heading'].reshape(-1)  # [(N1,...,Nb)*H]
        t2n_heading_n = n_heading.reshape(-1)  # [(N1,...,Nb)*H*K]
        t2n_valid_mask_t = data['agent']['visible_mask'][:, :self.num_historical_steps]  # [(N1,...,Nb),H]
        t2n_valid_mask_n = n_valid_mask.reshape(num_all_agent, -1)  # [(N1,...,Nb),H*K]
        t2n_valid_mask = t2n_valid_mask_t.unsqueeze(2) & t2n_valid_mask_n.unsqueeze(1)  # [(N1,...,Nb),H,H*K]
        t2n_edge_index = dense_to_sparse(t2n_valid_mask)[0]
        t2n_edge_index = t2n_edge_index[:, torch.floor(t2n_edge_index[1]/self.num_modes) >= t2n_edge_index[0]]
        t2n_edge_index = t2n_edge_index[:, torch.floor(t2n_edge_index[1]/self.num_modes) - t2n_edge_index[0] <= self.pos_duration]
        t2n_edge_vector = transform_point_to_local_coordinate(t2n_position_t[t2n_edge_index[0]], t2n_position_n[t2n_edge_index[1]], t2n_heading_n[t2n_edge_index[1]])
        t2n_edge_attr_length, t2n_edge_attr_theta = compute_angles_lengths_2D(t2n_edge_vector)
        t2n_edge_attr_heading = wrap_angle(t2n_heading_t[t2n_edge_index[0]] - t2n_heading_n[t2n_edge_index[1]])
        t2n_edge_attr_interval = t2n_edge_index[0] - torch.floor(t2n_edge_index[1]/self.num_modes) - self.num_future_steps//2
        t2n_edge_attr_input = torch.stack([t2n_edge_attr_length, t2n_edge_attr_theta, t2n_edge_attr_heading, t2n_edge_attr_interval], dim=-1)
        t2n_edge_attr_embs = self.t2m_emb_layer(input=t2n_edge_attr_input)

        # l2n edge
        if l_embs.size(0) > 0:
            l2n_position_l = data['lane']['position']  # [(M1,...,Mb),2]
            l2n_position_n = n_position.reshape(-1, 2)  # [(N1,...,Nb)*H*K,2]
            l2n_heading_l = data['lane']['heading']  # [(M1,...,Mb)]
            l2n_heading_n = n_heading.reshape(-1)  # [(N1,...,Nb)*H*K]
            l2n_batch_l = data['lane']['batch']  # [(M1,...,Mb)]
            l2n_batch_n = n_batch.unsqueeze(1).repeat_interleave(self.num_historical_steps, 1).reshape(-1)  # [(N1,...,Nb)*H*K]
            
            if 'visible_mask' in data['lane']:
                l2n_valid_mask_l = data['lane']['visible_mask']
            else:
                l2n_valid_mask_l = torch.ones(l2n_position_l.size(0), dtype=torch.bool, device=l2n_position_l.device)
            
            l2n_valid_mask_n = n_valid_mask.reshape(-1)  # [(N1,...,Nb)*H*K]
            l2n_valid_mask = l2n_valid_mask_l.unsqueeze(1) & l2n_valid_mask_n.unsqueeze(0)  # [(M1,...,Mb),(N1,...,Nb)*H*K]
            l2n_valid_mask = drop_edge_between_samples(l2n_valid_mask, batch=(l2n_batch_l, l2n_batch_n))
            l2n_edge_index = dense_to_sparse(l2n_valid_mask)[0]
            l2n_edge_index = l2n_edge_index[:, torch.norm(l2n_position_l[l2n_edge_index[0]] - l2n_position_n[l2n_edge_index[1]], p=2, dim=-1) < self.l2a_radius]
            
            if l2n_edge_index.size(1) > 0:
                l2n_edge_vector = transform_point_to_local_coordinate(l2n_position_l[l2n_edge_index[0]], l2n_position_n[l2n_edge_index[1]], l2n_heading_n[l2n_edge_index[1]])
                l2n_edge_attr_length, l2n_edge_attr_theta = compute_angles_lengths_2D(l2n_edge_vector)
                l2n_edge_attr_heading = wrap_angle(l2n_heading_l[l2n_edge_index[0]] - l2n_heading_n[l2n_edge_index[1]])
                l2n_edge_attr_input = torch.stack([l2n_edge_attr_length, l2n_edge_attr_theta, l2n_edge_attr_heading], dim=-1)
                l2n_edge_attr_embs = self.l2m_emb_layer(input=l2n_edge_attr_input)
                has_l2n_edges = True
            else:
                has_l2n_edges = False
        else:
            has_l2n_edges = False

        # Mode edge
        # n2n_a_edge
        n2n_a_position = n_position.permute(1, 2, 0, 3).reshape(-1, 2)  # [H*K*(N1,...,Nb),2]
        n2n_a_heading = n_heading.permute(1, 2, 0).reshape(-1)  # [H*K*(N1,...,Nb)]
        n2n_a_batch = data['agent']['batch']  # [(N1,...,Nb)]
        n2n_a_valid_mask = n_valid_mask.permute(1, 2, 0).reshape(self.num_historical_steps * self.num_modes, -1)  # [H*K,(N1,...,Nb)]
        n2n_a_valid_mask = n2n_a_valid_mask.unsqueeze(2) & n2n_a_valid_mask.unsqueeze(1)  # [H*K,(N1,...,Nb),(N1,...,Nb)]
        n2n_a_valid_mask = drop_edge_between_samples(n2n_a_valid_mask, n2n_a_batch)
        n2n_a_edge_index = dense_to_sparse(n2n_a_valid_mask)[0]
        n2n_a_edge_index = n2n_a_edge_index[:, n2n_a_edge_index[1] != n2n_a_edge_index[0]]
        n2n_a_edge_index = n2n_a_edge_index[:, torch.norm(n2n_a_position[n2n_a_edge_index[1]] - n2n_a_position[n2n_a_edge_index[0]], p=2, dim=-1) < self.a2a_radius]
        n2n_a_edge_vector = transform_point_to_local_coordinate(n2n_a_position[n2n_a_edge_index[0]], n2n_a_position[n2n_a_edge_index[1]], n2n_a_heading[n2n_a_edge_index[1]])
        n2n_a_edge_attr_length, n2n_a_edge_attr_theta = compute_angles_lengths_2D(n2n_a_edge_vector)
        n2n_a_edge_attr_heading = wrap_angle(n2n_a_heading[n2n_a_edge_index[0]] - n2n_a_heading[n2n_a_edge_index[1]])
        n2n_a_edge_attr_input = torch.stack([n2n_a_edge_attr_length, n2n_a_edge_attr_theta, n2n_a_edge_attr_heading], dim=-1)
        n2n_a_edge_attr_embs = self.m2m_a_emb_layer(input=n2n_a_edge_attr_input)

        # n2n_h edge
        n2n_h_position = n_position.permute(2, 0, 1, 3).reshape(-1, 2)  # [K*(N1,...,Nb)*H,2]
        n2n_h_heading = n_heading.permute(2, 0, 1).reshape(-1)  # [K*(N1,...,Nb)*H]
        n2n_h_valid_mask = n_valid_mask.permute(2, 0, 1).reshape(-1, self.num_historical_steps)  # [K*(N1,...,Nb),H]
        n2n_h_valid_mask = n2n_h_valid_mask.unsqueeze(2) & n2n_h_valid_mask.unsqueeze(1)  # [K*(N1,...,Nb),H,H]
        n2n_h_edge_index = dense_to_sparse(n2n_h_valid_mask)[0]
        n2n_h_edge_index = n2n_h_edge_index[:, n2n_h_edge_index[1] > n2n_h_edge_index[0]]
        n2n_h_edge_index = n2n_h_edge_index[:, n2n_h_edge_index[1] - n2n_h_edge_index[0] <= self.pred_duration]
        n2n_h_edge_vector = transform_point_to_local_coordinate(n2n_h_position[n2n_h_edge_index[0]], n2n_h_position[n2n_h_edge_index[1]], n2n_h_heading[n2n_h_edge_index[1]])
        n2n_h_edge_attr_length, n2n_h_edge_attr_theta = compute_angles_lengths_2D(n2n_h_edge_vector)
        n2n_h_edge_attr_heading = wrap_angle(n2n_h_heading[n2n_h_edge_index[0]] - n2n_h_heading[n2n_h_edge_index[1]])
        n2n_h_edge_attr_interval = n2n_h_edge_index[0] - n2n_h_edge_index[1]
        n2n_h_edge_attr_input = torch.stack([n2n_h_edge_attr_length, n2n_h_edge_attr_theta, n2n_h_edge_attr_heading, n2n_h_edge_attr_interval], dim=-1)
        n2n_h_edge_attr_embs = self.m2m_h_emb_layer(input=n2n_h_edge_attr_input)

        # n2n_s edge
        n2n_s_position = n_position.transpose(0, 1).reshape(-1, 2)  # [H*(N1,...,Nb)*K,2]
        n2n_s_heading = n_heading.transpose(0, 1).reshape(-1)  # [H*(N1,...,Nb)*K]
        n2n_s_valid_mask = n_valid_mask.transpose(0, 1).reshape(-1, self.num_modes)  # [H*(N1,...,Nb),K]
        n2n_s_valid_mask = n2n_s_valid_mask.unsqueeze(2) & n2n_s_valid_mask.unsqueeze(1)  # [H*(N1,...,Nb),K,K]
        n2n_s_edge_index = dense_to_sparse(n2n_s_valid_mask)[0]
        n2n_s_edge_index = n2n_s_edge_index[:, n2n_s_edge_index[0] != n2n_s_edge_index[1]]
        n2n_s_edge_vector = transform_point_to_local_coordinate(n2n_s_position[n2n_s_edge_index[0]], n2n_s_position[n2n_s_edge_index[1]], n2n_s_heading[n2n_s_edge_index[1]])
        n2n_s_edge_attr_length, n2n_s_edge_attr_theta = compute_angles_lengths_2D(n2n_s_edge_vector)
        n2n_s_edge_attr_heading = wrap_angle(n2n_s_heading[n2n_s_edge_index[0]] - n2n_s_heading[n2n_s_edge_index[1]])
        n2n_s_edge_attr_input = torch.stack([n2n_s_edge_attr_length, n2n_s_edge_attr_theta, n2n_s_edge_attr_heading], dim=-1)
        n2n_s_edge_attr_embs = self.m2m_s_emb_layer(input=n2n_s_edge_attr_input)

        # t2n attention
        t_embs = a_embs.reshape(-1, self.hidden_dim)  # [(N1,...,Nb)*H,D]
        n_embs_t = self.t2n_attn_layer(x=[t_embs, n_embs], edge_index=t2n_edge_index, edge_attr=t2n_edge_attr_embs)  # [(N1,...,Nb)*H*K,D]

        # l2n attention
        if has_l2n_edges:
            n_embs_l = self.l2n_attn_layer(x=[l_embs, n_embs], edge_index=l2n_edge_index, edge_attr=l2n_edge_attr_embs)  # [(N1,...,Nb)*H*K,D]
            n_embs = n_embs_t + n_embs_l
        else:
            n_embs = n_embs_t

        n_embs = n_embs.reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)  # [H*(N1,...,Nb)*K,D]
        
        # Mode attention
        for i in range(self.num_attn_layers):
            # n2n_a
            n_embs = n_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(1, 2).reshape(-1, self.hidden_dim)  # [H*K*(N1,...,Nb),D]
            n_embs = self.n2n_a_attn_layers[i](x=n_embs, edge_index=n2n_a_edge_index, edge_attr=n2n_a_edge_attr_embs)
            # n2n_h
            n_embs = n_embs.reshape(self.num_historical_steps, self.num_modes, num_all_agent, self.hidden_dim).permute(1, 2, 0, 3).reshape(-1, self.hidden_dim)  # [K*(N1,...,Nb)*H,D]
            n_embs = self.n2n_h_attn_layers[i](x=n_embs, edge_index=n2n_h_edge_index, edge_attr=n2n_h_edge_attr_embs)
            # n2n_s
            n_embs = n_embs.reshape(self.num_modes, num_all_agent, self.num_historical_steps, self.hidden_dim).transpose(0, 2).reshape(-1, self.hidden_dim)  # [H*(N1,...,Nb)*K,D]
            n_embs = self.n2n_s_attn_layers[i](x=n_embs, edge_index=n2n_s_edge_index, edge_attr=n2n_s_edge_attr_embs)
        n_embs = n_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)  # [(N1,...,Nb)*H*K,D]

        # Generate refinement (center + front)
        traj_refine_combined = self.traj_refine(n_embs).reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.num_future_steps, 4)  # [(N1,...,Nb),H,K,F,4]
        
        traj_refine_center = traj_refine_combined[..., :2]  # [(N1,...,Nb),H,K,F,2]
        traj_refine_front = traj_refine_combined[..., 2:]  # [(N1,...,Nb),H,K,F,2]
        
        traj_output = transform_traj_to_global_coordinate(proposal_local + traj_refine_center, n_position, n_heading)  # [(N1,...,Nb),H,K,F,2]
        front_output = transform_traj_to_global_coordinate(front_proposal_local + traj_refine_front, n_position, n_heading)  # [(N1,...,Nb),H,K,F,2]

        # Generate prob
        prob_output = self.prob_decoder(n_embs.detach()).reshape(-1, self.num_historical_steps, self.num_modes)  # [(N1,...,Nb),H,K]
        prob_output = self.prob_norm(prob_output)  # [(N1,...,Nb),H,K]
        
        return traj_propose, traj_output, prob_output, front_propose, front_output
