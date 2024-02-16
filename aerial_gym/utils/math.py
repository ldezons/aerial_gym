# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor

@torch.jit.script
def compute_vee_map(skew_matrix):
    # type: (Tensor) -> Tensor

    # return vee map of skew matrix
    vee_map = torch.stack(
        [-skew_matrix[:, 1, 2], skew_matrix[:, 0, 2], -skew_matrix[:, 0, 1]], dim=1)
    return vee_map


@torch.jit.script
def convert_enu_to_ned(states_enu):
    # states_enu is a tensor of shape (num_envs, 13) 
    # with [position(3), quaternion(4), velocity(3), angular_velocity(3)] for each environment
    states_ned = torch.empty_like(states_enu)
        
    # Position conversion: [y, x, -z]
    states_ned[:, 0] = states_enu[:, 1]  # y_N becomes the first component (North)
    states_ned[:, 1] = states_enu[:, 0]  # x_E becomes the second component (East)
    states_ned[:, 2] = -states_enu[:, 2]  # -z_U becomes the third component (Down)
        
    # Quaternion conversion: No change in values but interpretation changes due to coordinate system
    # Assuming the quaternion is in ENU, and your system can interpret it correctly in NED
    states_ned[:, 3:7] = states_enu[:, 3:7]
        
    # Velocity conversion: Similar to position
    states_ned[:, 7] = states_enu[:, 8]  # v_N becomes the first velocity component
    states_ned[:, 8] = states_enu[:, 7]  # v_E becomes the second velocity component
    states_ned[:, 9] = -states_enu[:, 9]  # -v_U becomes the third velocity component (Down)
        
    # Angular velocity conversion: Similar to position and velocity
    states_ned[:, 10] = states_enu[:, 11]  # omega_N becomes the first angular velocity component
    states_ned[:, 11] = states_enu[:, 10]  # omega_E becomes the second angular velocity component
    states_ned[:, 12] = -states_enu[:, 12]  # -omega_U becomes the third angular velocity component (Down)
        
    return states_ned
    
@torch.jit.script
def convert_ned_to_enu(states_ned):
    # states_ned is a tensor of shape (num_envs, 13) 
    # with [position(3), quaternion(4), velocity(3), angular_velocity(3)] for each environment
    states_enu = torch.empty_like(states_ned)
        
    # Position conversion: [y, x, -z]
    states_enu[:, 0] = states_ned[:, 1]  # y_N becomes the second component (East)
    states_enu[:, 1] = states_ned[:, 0]  # x_E becomes the first component (North)
    states_enu[:, 2] = -states_ned[:, 2]  # -z_D becomes the third component (Up)
        
    # Quaternion conversion: No change in values but interpretation changes due to coordinate system
    states_enu[:, 3:7] = states_ned[:, 3:7]
        
    # Velocity conversion: Similar to position
    states_enu[:, 7] = states_ned[:, 8]  # v_N becomes the second velocity component
    states_enu[:, 8] = states_ned[:, 7]  # v_E becomes the first velocity component
    states_enu[:, 9] = -states_ned[:, 9]  # -v_D becomes the third velocity component (Up)
        
    # Angular velocity conversion: Similar to position and velocity
    states_enu[:, 10] = states_ned[:, 11]  # omega_N becomes the second angular velocity component
    states_enu[:, 11] = states_ned[:, 10]  # omega_E becomes the first angular velocity component
    states_enu[:, 12] = -states_ned[:, 12]  # -omega_D becomes the third angular velocity component (Up)
        
    return states_enu

@torch.jit.script
def quaternion_multiply(q1, q2):
    # Extract real and imaginary parts
    w1, v1 = q1[..., 0], q1[..., 1:]
    w2, v2 = q2[..., 0], q2[..., 1:]

    # Compute the product
    w = w1 * w2 - torch.sum(v1 * v2, dim=-1)
    v = (w1.unsqueeze(-1) * v2) + (w2.unsqueeze(-1) * v1) + torch.cross(v1, v2, dim=-1)

    return torch.cat((w.unsqueeze(-1), v), dim=-1)

@torch.jit.script
def quaternion_inverse(q):
    q_inv = q.clone()
    q_inv[..., 1:] = -q_inv[..., 1:]  # Invert the vector part
    return q_inv

@torch.jit.script
def quaternion_to_dcm_z(q):
    # Extract components
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Compute the last column of the corresponding rotation matrix
    zz = 1 - 2 * (x**2 + y**2)
    xz = 2 * (x * z + w * y)
    yz = 2 * (y * z - w * x)

    return torch.stack((xz, yz, zz), dim=-1)

@torch.jit.script
def quaternion_from_two_vectors(v0, v1):
    # Normalize input vectors
    v0_n = torch.nn.functional.normalize(v0, dim=-1)
    v1_n = torch.nn.functional.normalize(v1, dim=-1)

    # Compute the cross product and dot product
    c = torch.cross(v0_n, v1_n, dim=-1)
    d = torch.sum(v0_n * v1_n, dim=-1)

    # Compute the real part and add a small epsilon to avoid division by zero in normalization
    w = 1 + d
    epsilon = 1e-10
    q = torch.cat((w.unsqueeze(-1), c), dim=-1)
    
    # Normalize the quaternion
    q_n = torch.nn.functional.normalize(q + epsilon, dim=-1)
    return q_n

@torch.jit.script
def quaternion_canonicalize(q):
    # Negate the quaternion if the scalar part is negative
    should_negate = q[..., 0] < 0
    q[should_negate] = -q[should_negate]
    return q
