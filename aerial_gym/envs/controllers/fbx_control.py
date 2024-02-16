import torch
import pytorch3d.transforms as p3d_transforms
from aerial_gym.utils.math import *


class FbxControl:
    def __init__(self):
        self.robot_mass = 0.0
        self.g = 0.0
        # acceleration to thrust sp and attitude sp paramaters
        self.thr_min = 4.0  # [N]
        self.thr_max = 16.0  # [N]
        self._idle_thrust = 1.0e-4
        self.thrust2newton = 20.0
        self.tilt_max_rad = torch.deg2rad(35.0)

        # attitude to rate sp parameters

        # rate control parameters
        self.rate_int = torch.Tensor
        self.rate_gain_K = 1.0
        self.rate_gain_P = 0.22
        self.rate_gain_I = 1.2
        self.rate_gain_ff = 0.0

    def __call__(self, state, command_actions):

        # convert to NED
        robot_state = convert_enu_to_ned(state)
        # extract states
        p = robot_state[:, :3]
        q = robot_state[:, 3:7]
        v = robot_state[:, 7:10]
        omega = robot_state[:, 10:13]

        # hover thrust tensor
        self.hover_thrust = torch.tensor(0.5).repeat(command_actions.size(0), 1)

        # sp commands
        acc_sp = command_actions[:, :3]
        yaw_sp = command_actions[:, 3]
        q_sp, T_sp = self.acc_to_T(acc_sp, yaw_sp)

        rate_sp = self.attitude_to_rate(q, q_sp, 1.0, yawspeed_setpoint=0.0)
        torque_sp = self.rate_control(rate_sp, omega, self.rate_int)

        return T_sp * quaternion_to_dcm_z(q), torque_sp

    @torch.jit.script.method
    def acc_to_T(self, acc_sp: torch.Tensor, yaw_sp: torch.Tensor):
        # desired z body axis
        g = torch.full((acc_sp.size(0), 1), self.g)
        body_z = torch.cat((-acc_sp[:, 0:2], g), dim=1)
        body_z = torch.nn.functional.normalize(body_z, p=2, dim=1)
        # limit tilt
        world_dir = torch.tensor([0, 0, 1])
        world_unit = world_dir.repeat(acc_sp.size(0), dim=1)
        body_z = self.limit_tilt(body_z, world_unit, self.tilt_max_rad)

        # Scale thrust assuming hover thrust produces standard gravity
        collective_thrust = acc_sp[:, 2] * (self.hover_thrust / g) - self.hover_thrust

        # Project thrust to planned body attitude
        collective_thrust = collective_thrust / body_z[:, 2]
        collective_thrust = torch.min(collective_thrust, torch.tensor(-self.thr_min))
        thr_sp = body_z * collective_thrust

        # Saturate maximal vertical thrust
        thr_sp[:, 2] = torch.max(thr_sp[:, 2], torch.tensor(-self.thr_max))

        # Get allowed horizontal thrust after prioritizing vertical control
        thr_max_squared = torch.tensor(self.thr_max**2).repeat(acc_sp.size(0), 1)
        thr_z_squared = thr_sp[:, 2] * thr_sp[:, 2]
        thr_max_xy = torch.sqrt(thr_max_squared - thr_z_squared)
        
        # Saturate thrust in horizontal direction
        thr_sp_xy = thr_sp[:, :2]
        thr_sp_xy_norm = torch.norm(thr_sp_xy, dim=1)
        cond = thr_sp_xy_norm > thr_max_xy
        thr_sp_xy[cond] = (thr_sp_xy[cond] / thr_sp_xy_norm[cond]) * thr_max_xy[cond]
        thr_sp[:, :2] = thr_sp_xy
        thr_sp = thr_sp * torch.tensor(self.thrust2newton)

        q_sp = self.bodyz_to_attitude(-thr_sp, yaw_sp)
        T_sp = torch.norm(thr_sp, dim=1)
        return q_sp, T_sp

    @torch.jit.script.method
    def limit_tilt(self, body_unit, world_unit, max_angle):

        # Determine tilt
        dot_product_unit = torch.sum(body_unit * world_unit, dim=1, keepdim=True)
        angle = torch.acos(dot_product_unit)

        # Limit tilt
        angle = torch.min(angle, torch.tensor(max_angle))

        rejection = body_unit - (dot_product_unit * world_unit)

        # Corner case for exactly parallel vectors
        rejection_norm_squared = torch.sum(rejection**2, dim=1, keepdim=True)
        close_to_zero = rejection_norm_squared < torch.finfo(torch.float32).eps

        # Handling the corner case by setting a unit vector in the x direction
        rejection[close_to_zero, 0] = 1.0
        # rejection[close_to_zero, 1:] = 0.0

        # Normalize the rejection vector
        rejection_unit = rejection / torch.norm(rejection, dim=1, keepdim=True)

        # Update body_unit
        body_unit = torch.cos(angle) * world_unit + torch.sin(angle) * rejection_unit

        return body_unit

    @torch.jit.script.method
    def bodyz_to_attitude(self, body_z, yaw_sp):
        # Ensure body_z and yaw_sp are torch tensors
        num_envs = body_z.shape[0]

        # Check for near-zero vector and set to safe value
        norm_squared = torch.sum(body_z**2, dim=1, keepdim=True)
        near_zero = norm_squared < torch.finfo(body_z.dtype).eps
        body_z[near_zero, 2] = 1.0

        # Normalize body_z
        body_z = torch.nn.functional.normalize(body_z, dim=1)

        # Vector of desired yaw direction in XY plane, rotated by PI/2
        y_C = torch.stack(
            [
                -torch.sin(yaw_sp).squeeze(),
                torch.cos(yaw_sp).squeeze(),
                torch.zeros(num_envs, device=body_z.device),
            ],
            dim=1,
        )

        # Desired body_x axis, orthogonal to body_z (cross product)
        body_x = torch.cross(y_C, body_z, dim=1)

        # Keep nose to front while inverted upside down
        inverted = body_z[:, 2] < 0.0
        body_x[inverted] = -body_x[inverted]

        # If body_z(2) is near zero, adjust body_x
        near_zero_z = torch.abs(body_z[:, 2:3]) < 1e-6
        body_x[near_zero_z.squeeze(), :] = torch.tensor(
            [0.0, 0.0, 1.0], device=body_z.device
        )

        # Normalize body_x
        body_x = torch.nn.functional.normalize(body_x, dim=1)

        # Desired body_y axis (cross product)
        body_y = torch.cross(body_z, body_x, dim=1)

        # Construct rotation matrix R_sp from body_x, body_y, body_z
        R_sp = torch.stack([body_x, body_y, body_z], dim=-1)  # Shape: (num_envs, 3, 3)

        # Calculate quaternion from R_sp
        # Preallocate quaternion tensor
        q = torch.zeros((num_envs, 4), device=body_z.device)

        # Compute quaternion components
        q[:, 0] = 0.5 * torch.sqrt(
            1 + R_sp[:, 0, 0] + R_sp[:, 1, 1] + R_sp[:, 2, 2]
        ).unsqueeze(
            -1
        )  # w
        q[:, 1] = (R_sp[:, 2, 1] - R_sp[:, 1, 2]) / (4 * q[:, 0])  # x
        q[:, 2] = (R_sp[:, 0, 2] - R_sp[:, 2, 0]) / (4 * q[:, 0])  # y
        q[:, 3] = (R_sp[:, 1, 0] - R_sp[:, 0, 1]) / (4 * q[:, 0])  # z

        # Normalize quaternion to ensure it represents a valid rotation
        q = torch.nn.functional.normalize(q, dim=1)

        return q

    @torch.jit.script.method
    def attitude_to_rate(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        yaw_w: float,
        yawspeed_setpoint: float,
        proportional_gain: torch.Tensor,
        rate_limit: torch.Tensor,
    ) -> torch.Tensor:
        # Calculate reduced desired attitude neglecting vehicle's yaw to prioritize roll and pitch
        e_z = quaternion_to_dcm_z(q)
        e_z_d = quaternion_to_dcm_z(qd)
        qd_red = quaternion_from_two_vectors(e_z, e_z_d)

        # Check for conditions leading to singularity or numerical instability
        condition1 = torch.abs(qd_red[:, 1]) > 1 - 1e-5
        condition2 = torch.abs(qd_red[:, 2]) > 1 - 1e-5
        qd_red = torch.where(condition1 | condition2, qd, qd_red)

        # Transform rotation from current to desired thrust vector into a world frame reduced desired attitude
        qd_red = quaternion_multiply(qd_red, q)

        # Mix full and reduced desired attitude
        q_mix = quaternion_multiply(quaternion_inverse(qd_red), qd)
        q_mix = quaternion_canonicalize(q_mix)
        q_mix[:, 0] = torch.clamp(
            q_mix[:, 0], -1, 1
        )  # Clamp to avoid numerical issues with acos
        q_mix[:, 3] = torch.clamp(q_mix[:, 3], -1, 1)

        # Update qd based on the mixing and yaw weighting
        cos_component = torch.cos(yaw_w * torch.acos(q_mix[:, 0]))
        sin_component = torch.sin(yaw_w * torch.asin(q_mix[:, 3]))
        qd_update = torch.stack(
            [
                cos_component,
                torch.zeros_like(cos_component),
                torch.zeros_like(cos_component),
                sin_component,
            ],
            dim=-1,
        )
        qd = quaternion_multiply(qd_red, qd_update)

        # Quaternion attitude control law, qe is rotation from q to qd
        qe = quaternion_multiply(quaternion_inverse(q), qd)

        # Using sin(alpha/2) scaled rotation axis as attitude error
        eq = (
            2 * qe[:, 1:]
        )  # Assuming qe is in the form [w, x, y, z] and eq is the vector part [x, y, z]

        # Calculate angular rates setpoint
        rate_setpoint = eq * proportional_gain.unsqueeze(
            0
        )  # Element-wise multiplication

        # Adjust for yawspeed setpoint
        if torch.isfinite(yawspeed_setpoint):
            rate_setpoint += (
                quaternion_to_dcm_z(quaternion_inverse(q)) * yawspeed_setpoint
            )

        # Limit rates
        rate_setpoint = torch.clamp(
            rate_setpoint, -rate_limit.unsqueeze(0), rate_limit.unsqueeze(0)
        )

        return rate_setpoint

    @torch.jit.script.method
    def rate_control(
        self,
        rate_setpoint: torch.Tensor,
        rate: torch.Tensor,
        rate_integral: torch.Tensor,
    ) -> torch.Tensor:
        return (
            torch.tensor(self.rate_gain_P) * (rate_setpoint - rate)
            - rate_integral
            + self.rate_gain_ff * rate_setpoint
        )
