import torch
import math

from typing import Any, List, Optional, Tuple, Union


def wrap_angle(angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)


def get_index_of_A_in_B(list_A: Optional[List[Any]], list_B: Optional[List[Any]]) -> List[int]: 
    if not list_A or not list_B:
        return []

    set_B = set(list_B)
    indices = [list_B.index(i) for i in list_A if i in set_B]

    return indices

    
def generate_clockwise_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros_like(angle).unsqueeze(-1).repeat_interleave(2, -1).unsqueeze(-1).repeat_interleave(2, -1)
    matrix[..., 0, 0] = torch.cos(angle)
    matrix[..., 0, 1] = torch.sin(angle)
    matrix[..., 1, 0] = -torch.sin(angle)
    matrix[..., 1, 1] = torch.cos(angle)
    return matrix


def generate_counterclockwise_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros_like(angle).unsqueeze(-1).repeat_interleave(2, -1).unsqueeze(-1).repeat_interleave(2, -1)
    matrix[..., 0, 0] = torch.cos(angle)
    matrix[..., 0, 1] = -torch.sin(angle)
    matrix[..., 1, 0] = torch.sin(angle)
    matrix[..., 1, 1] = torch.cos(angle)
    return matrix

    
def compute_angles_lengths_3D(vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = torch.norm(vectors, dim=-1)
    theta = torch.atan2(vectors[..., 1], vectors[..., 0])
    r_xy = torch.norm(vectors[..., :2], dim=-1)
    phi = torch.atan2(vectors[..., 2], r_xy)
    return length, theta, phi


def compute_angles_lengths_2D(vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    length = torch.norm(vectors, dim=-1)
    theta = torch.atan2(vectors[..., 1], vectors[..., 0])
    return length, theta


def drop_edge_between_samples(valid_mask: torch.Tensor, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        batch_matrix = batch.unsqueeze(-1) == batch.unsqueeze(-2)
    else:
        batch_src, batch_dst = batch
        batch_matrix = batch_src.unsqueeze(-1) == batch_dst.unsqueeze(-2)
    valid_mask = valid_mask * batch_matrix.unsqueeze(0)
    return valid_mask


def transform_traj_to_local_coordinate(traj: torch.Tensor, position: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    traj = traj - position.unsqueeze(-2)
    rotation_matrix = generate_clockwise_rotation_matrix(heading)
    traj = torch.matmul(rotation_matrix.unsqueeze(-3), traj.unsqueeze(-1)).squeeze(-1)
    return traj


def transform_traj_to_global_coordinate(traj: torch.Tensor, position: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    rotation_matrix = generate_counterclockwise_rotation_matrix(heading)
    traj = torch.matmul(rotation_matrix.unsqueeze(-3), traj.unsqueeze(-1)).squeeze(-1)
    traj = traj + position.unsqueeze(-2)
    return traj


def transform_point_to_local_coordinate(point: torch.Tensor, position: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    point = point - position
    rotation_matrix = generate_clockwise_rotation_matrix(heading)
    point = torch.matmul(rotation_matrix, point.unsqueeze(-1)).squeeze(-1)
    return point


def transform_point_to_global_coordinate(point: torch.Tensor, position: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    rotation_matrix = generate_counterclockwise_rotation_matrix(heading)
    point = torch.matmul(rotation_matrix, point.unsqueeze(-1)).squeeze(-1)
    point = point + position
    return point


def generate_target(position: torch.Tensor, mask: torch.Tensor, num_historical_steps: int, num_future_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
    target_traj = [position[:, i+1:i+1+num_future_steps] for i in range(num_historical_steps)]
    target_traj = torch.stack(target_traj, dim=1)
    target_mask = [mask[:, i+1:i+1+num_future_steps] for i in range(num_historical_steps)]
    target_mask = torch.stack(target_mask, dim=1)
    return target_traj, target_mask


def generate_reachable_matrix(edge_index: torch.Tensor, num_hops: int, max_nodes: int) -> list:
    values = torch.ones(edge_index.size(1), device=edge_index.device)
    sparse_mat = torch.sparse_coo_tensor(edge_index, values, torch.Size([max_nodes, max_nodes]))

    reach_matrices = []
    current_matrix = sparse_mat.clone()
    for _ in range(num_hops):
        current_matrix = current_matrix.coalesce()
        current_matrix = torch.sparse_coo_tensor(current_matrix.indices(), torch.ones_like(current_matrix.values()), current_matrix.size())

        edge_index_now = current_matrix.coalesce().indices()
        reach_matrices.append(edge_index_now)

        next_matrix = torch.sparse.mm(current_matrix, sparse_mat)

        current_matrix = next_matrix
    return reach_matrices


def generate_predict_mask(visible_mask: torch.Tensor, num_visible_steps: int) -> torch.Tensor:
    
    window = torch.ones((1, num_visible_steps), dtype=torch.float32, device=visible_mask.device)

    conv_result = torch.nn.functional.conv2d(visible_mask.float().unsqueeze(0).unsqueeze(0), window.unsqueeze(0).unsqueeze(0))
    conv_result = conv_result.squeeze(0).squeeze(0)
    
    predict_mask = conv_result == num_visible_steps
    predict_mask = torch.cat([torch.zeros((visible_mask.size(0), num_visible_steps-1), dtype=torch.bool, device=visible_mask.device), predict_mask], dim=1)
    
    return predict_mask


def compute_vehicle_endpoints(position: torch.Tensor, heading: torch.Tensor, vehicle_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute front and rear midpoints of vehicles.
    
    Args:
        position: Vehicle center position [..., 2]
        heading: Vehicle heading angle in radians [...]
        vehicle_length: Vehicle length [...]
    
    Returns:
        front_midpoint: Front midpoint coordinates [..., 2]
        rear_midpoint: Rear midpoint coordinates [..., 2]
    """
    half_length = vehicle_length / 2.0
    
    # Direction vectors
    cos_heading = torch.cos(heading)
    sin_heading = torch.sin(heading)
    
    # Front midpoint: center + (length/2) * direction
    front_midpoint = torch.zeros_like(position)
    front_midpoint[..., 0] = position[..., 0] + half_length * cos_heading
    front_midpoint[..., 1] = position[..., 1] + half_length * sin_heading
    
    # Rear midpoint: center - (length/2) * direction
    rear_midpoint = torch.zeros_like(position)
    rear_midpoint[..., 0] = position[..., 0] - half_length * cos_heading
    rear_midpoint[..., 1] = position[..., 1] - half_length * sin_heading
    
    return front_midpoint, rear_midpoint


def compute_endpoint_edge_attributes(
    src_position: torch.Tensor,
    dst_position: torch.Tensor,
    src_heading: torch.Tensor,
    dst_heading: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute edge attributes for endpoint-to-endpoint connections.
    
    Args:
        src_position: Source endpoint position [E, 2]
        dst_position: Destination endpoint position [E, 2]
        src_heading: Source vehicle heading [E]
        dst_heading: Destination vehicle heading [E]
    
    Returns:
        length: Distance between endpoints [E]
        heading_diff: Heading angle difference [E]
        theta: Bearing angle from dst to src [E]
    """
    # Distance between endpoints
    diff = src_position - dst_position
    length = torch.norm(diff, dim=-1)
    
    # Bearing angle (theta) - direction from dst to src
    theta = torch.atan2(diff[..., 1], diff[..., 0])
    
    # Heading difference (larger x heading - smaller x heading)
    # This will be computed based on x coordinates in the calling function
    heading_diff = src_heading - dst_heading
    heading_diff = wrap_angle(heading_diff)
    
    return length, heading_diff, theta
