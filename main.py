from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
from collections import deque
from datetime import datetime

import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import mujoco
import numpy as np

try:
	from stable_baselines3 import PPO
	from stable_baselines3.common.callbacks import BaseCallback
	from stable_baselines3.common.monitor import Monitor
	from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
except Exception:  # pragma: no cover
	PPO = None
	BaseCallback = object  # type: ignore[assignment]
	Monitor = None
	DummyVecEnv = None
	SubprocVecEnv = None
	VecVideoRecorder = None

try:
	import mujoco.viewer as mj_viewer
except Exception:  # pragma: no cover
	mj_viewer = None


GOAL_ANGLES_ARMS: dict[str, float] = {
	"r_sho_pitch": 1.57,
	"l_sho_pitch": -1.57,
	"r_sho_roll": -0.9,
	"l_sho_roll": 0.9,
	"r_el": 0.8,
	"l_el": -0.8,
}

GOAL_ANGLES_ARMSLEGS: dict[str, float] = {
	"r_sho_pitch": 1.57,
	"l_sho_pitch": -1.57,
	"r_sho_roll": -0.9,
	"l_sho_roll": 0.9,
	"r_el": 0.8,
	"l_el": -0.8,
	"r_hip_pitch": 0.3,
	"l_hip_pitch": -0.3,
	"r_knee": -0.6,
	"l_knee": 0.6,
}


@dataclass
class GainConfig:
	kp_scale_stiff: float = 1.0
	kd_scale_stiff: float = 1.0
	kp_scale_compliant: float = 0.45
	kd_scale_compliant: float = 1.9
	kp_scale_min: float = 0.2
	kp_scale_max: float = 1.5
	kd_scale_min: float = 0.5
	kd_scale_max: float = 3.0


@dataclass
class ImpactConfig:
	rolling_window: int = 10
	sigma_multiplier: float = 4.0
	jump_ratio: float = 1.25
	refractory_steps: int = 5


@dataclass
class GoalGateConfig:
	mean_err_thr: float = 0.12
	max_err_thr: float = 0.20
	mean_qvel_thr: float = 0.8
	consecutive_steps: int = 1


@dataclass
class RestoreConfig:
	no_spike_steps: int = 20
	low_load_steps: int = 10
	baseline_multiplier: float = 1.2
	ramp_steps: int = 50


@dataclass
class PushConfig:
	force_xyz: tuple[float, float, float] = (25.0, 0.0, 0.0)
	duration_sec: float = 0.08


@dataclass
class TimeToImpactConfig:
	threshold: float = 0.1  # Time to impact threshold in seconds (increase for earlier detection)
	eps: float = 1e-6  # Epsilon for avoiding division by zero


@dataclass
class RewardConfig:
	w_head: float = 1.0
	w_torque: float = 0.5
	w_speed: float = 0.1
	w_align: float = 0.3
	max_head_vel: float = 5.0
	max_torque: float = 500.0
	max_fall_speed: float = 10.0


@dataclass
class RLTrainingConfig:
	total_timesteps: int = 200_000
	max_episode_steps: int = 250
	n_envs: int = 1
	checkpoint_interval_episodes: int = 100
	video_interval_episodes: int = 100
	n_steps: int = 2048
	batch_size: int = 256
	learning_rate: float = 3e-4
	gamma: float = 0.99
	gae_lambda: float = 0.95


@dataclass
class AlignConfig:
	angle_threshold_rad: float = float(np.pi / 4.0)


class Op3FallControlEnv(gym.Env):
	metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

	def __init__(
		self,
		model_xml: str | Path,
		goal_angles: dict[str, float],
		render_mode: str = "human",
		control_timestep: float = 0.02,
		camera_distance: float = 1.5,
		camera_azimuth: float = 160.0,
		camera_elevation: float = -20.0,
	) -> None:
		super().__init__()
		self.render_mode = render_mode
		self.model = mujoco.MjModel.from_xml_path(str(model_xml))
		self.data = mujoco.MjData(self.model)
		self.model.opt.timestep = control_timestep

		self._viewer = None
		self._renderer = None
		# Ensure offscreen framebuffer can satisfy requested render size.
		self.model.vis.global_.offwidth = int(max(self.model.vis.global_.offwidth, 1280))
		self.model.vis.global_.offheight = int(max(self.model.vis.global_.offheight, 720))
		self._camera = mujoco.MjvCamera()
		mujoco.mjv_defaultFreeCamera(self.model, self._camera)
		self._camera.distance = camera_distance
		self._camera.azimuth = camera_azimuth
		self._camera.elevation = camera_elevation
		self._camera.lookat[:] = np.array([0.0, 0.0, 0.2], dtype=np.float64)

		self.goal_angles = goal_angles
		self.control_joint_names = list(goal_angles.keys())

		self.joint_ids = {
			n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
			for n in self.control_joint_names
		}
		self.actuator_ids = {
			n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_act")
			for n in self.control_joint_names
		}
		self.dof_ids = {n: int(self.model.jnt_dofadr[jid]) for n, jid in self.joint_ids.items()}
		self.qpos_ids = {n: int(self.model.jnt_qposadr[jid]) for n, jid in self.joint_ids.items()}

		self.control_actuator_ids = np.array(list(self.actuator_ids.values()), dtype=np.int32)
		self.control_dof_ids = np.array(list(self.dof_ids.values()), dtype=np.int32)
		self.control_qpos_ids = np.array(list(self.qpos_ids.values()), dtype=np.int32)

		# Gain baselines (position actuator + dof damping).
		self.base_act_gainprm = self.model.actuator_gainprm.copy()
		self.base_act_biasprm = self.model.actuator_biasprm.copy()
		self.base_dof_damping = self.model.dof_damping.copy()

		self.current_kp_scale = 1.0
		self.current_kd_scale = 1.0
		self.is_in_compliant_mode = False

		self.head_body_id = self._resolve_head_body_id()
		self.step_count = 0

		# Initialize IMU sensor tracking
		self.imu_sensor_ids = self._find_imu_sensors()
		self.imu_data_ranges = self._get_imu_data_ranges()
		print(f"Found {len(self.imu_sensor_ids)} IMU sensors: {list(self.imu_sensor_ids.keys())}")

		# NEW: Initialize body IDs for time-to-impact calculation
		self.body_ids_for_impact = self._get_body_ids_for_impact()
		print(f"Found {len(self.body_ids_for_impact)} bodies for impact detection: {list(self.body_ids_for_impact.keys())}")

		# Minimal observation/action spaces for gym compatibility.
		self.observation_space = spaces.Box(
			low=-np.inf,
			high=np.inf,
			shape=(self.model.nq + self.model.nv,),
			dtype=np.float64,
		)
		self.action_space = spaces.Box(
			low=-np.inf,
			high=np.inf,
			shape=(len(self.control_joint_names),),
			dtype=np.float64,
		)

		self._target_ctrl = np.zeros(len(self.control_joint_names), dtype=np.float64)

	def _resolve_head_body_id(self) -> int:
		for name in ("head_tilt_link", "head_pan_link"):
			body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
			if body_id >= 0:
				return body_id
		raise ValueError("Could not resolve head body for push force.")

	def _get_body_ids_for_impact(self) -> dict[str, int]:
		"""Get body IDs for all body parts needed for time-to-impact calculation."""
		body_names = {
			# Hands (if sites exist, otherwise use elbow links)
			"left_hand": "l_el_link",
			"right_hand": "r_el_link",
			# Elbows
			"left_elbow": "l_el_link",
			"right_elbow": "r_el_link",
			# Knees
			"left_knee": "l_knee_link",
			"right_knee": "r_knee_link",
			# Torso (COM)
			"torso": "body_link",
			# Head
			"head": "head_tilt_link",
			# Feet
			"left_foot": "l_ank_roll_link",
			"right_foot": "r_ank_roll_link",
		}
		
		body_ids = {}
		for friendly_name, xml_name in body_names.items():
			body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, xml_name)
			if body_id >= 0:
				body_ids[friendly_name] = body_id
			else:
				print(f"Warning: Could not find body {xml_name} for {friendly_name}")
		
		return body_ids

	def compute_time_to_impact(self) -> tuple[float, str | None]:
		"""
		Compute minimum time to impact across all tracked body parts.
		
		Returns:
			tuple[float, str | None]: (minimum time to impact, name of body part with min time)
		"""
		min_t_impact = float('inf')
		min_body_name = None
		
		for body_name, body_id in self.body_ids_for_impact.items():
			if body_id >= 0:
				# Get z-position (height above ground)
				z_pos = float(self.data.xpos[body_id, 2])
				
				# Get vertical velocity from cvel (composite velocity)
				# cvel contains [linear_vel_x, linear_vel_y, linear_vel_z, angular_vel_x, angular_vel_y, angular_vel_z]
				z_vel = float(self.data.cvel[body_id, 2])  # Linear Z velocity
				
				# Only consider if falling downward (z_vel < 0)
				if z_vel < -1e-6:  # Small threshold to avoid near-zero velocities
					# Time to impact = height / downward speed
					t_impact = z_pos / abs(z_vel)
					
					if t_impact < min_t_impact:
						min_t_impact = t_impact
						min_body_name = body_name
		
		# If no body parts are falling, return infinity
		if min_t_impact == float('inf'):
			return float('inf'), None
		
		return min_t_impact, min_body_name

	def _find_imu_sensors(self) -> dict[str, int]:
		"""Find all IMU-related sensors in the model."""
		imu_sensors = {}
		for i in range(self.model.nsensor):
			sensor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
			if sensor_name:
				# Common IMU sensor types in MuJoCo
				sensor_type = self.model.sensor_type[i]
				if sensor_type in (
					mujoco.mjtSensor.mjSENS_ACCELEROMETER,
					mujoco.mjtSensor.mjSENS_GYRO,
					mujoco.mjtSensor.mjSENS_MAGNETOMETER,
					mujoco.mjtSensor.mjSENS_FRAMEQUAT,
					mujoco.mjtSensor.mjSENS_FRAMELINVEL,
					mujoco.mjtSensor.mjSENS_FRAMEANGVEL,
				):
					imu_sensors[sensor_name] = i
		return imu_sensors

	def _get_imu_data_ranges(self) -> dict[str, tuple[int, int]]:
		"""Get the data address ranges for each IMU sensor."""
		data_ranges = {}
		for name, sensor_id in self.imu_sensor_ids.items():
			adr = self.model.sensor_adr[sensor_id]
			dim = self.model.sensor_dim[sensor_id]
			data_ranges[name] = (int(adr), int(adr + dim))
		return data_ranges

	def _read_imu_data(self) -> dict[str, Any]:
		"""Read all IMU sensor data with detailed component labels."""
		imu_data = {}
		for sensor_name, sensor_id in self.imu_sensor_ids.items():
			sensor_type = self.model.sensor_type[sensor_id]
			adr_start, adr_end = self.imu_data_ranges[sensor_name]
			raw_data = self.data.sensordata[adr_start:adr_end]
			
			# Label components based on sensor type
			if sensor_type == mujoco.mjtSensor.mjSENS_ACCELEROMETER:
				imu_data[f"{sensor_name}_x"] = float(raw_data[0])
				imu_data[f"{sensor_name}_y"] = float(raw_data[1])
				imu_data[f"{sensor_name}_z"] = float(raw_data[2])
			elif sensor_type == mujoco.mjtSensor.mjSENS_GYRO:
				imu_data[f"{sensor_name}_x"] = float(raw_data[0])
				imu_data[f"{sensor_name}_y"] = float(raw_data[1])
				imu_data[f"{sensor_name}_z"] = float(raw_data[2])
			elif sensor_type == mujoco.mjtSensor.mjSENS_MAGNETOMETER:
				imu_data[f"{sensor_name}_x"] = float(raw_data[0])
				imu_data[f"{sensor_name}_y"] = float(raw_data[1])
				imu_data[f"{sensor_name}_z"] = float(raw_data[2])
			elif sensor_type == mujoco.mjtSensor.mjSENS_FRAMEQUAT:
				imu_data[f"{sensor_name}_w"] = float(raw_data[0])
				imu_data[f"{sensor_name}_x"] = float(raw_data[1])
				imu_data[f"{sensor_name}_y"] = float(raw_data[2])
				imu_data[f"{sensor_name}_z"] = float(raw_data[3])
			elif sensor_type == mujoco.mjtSensor.mjSENS_FRAMELINVEL:
				imu_data[f"{sensor_name}_x"] = float(raw_data[0])
				imu_data[f"{sensor_name}_y"] = float(raw_data[1])
				imu_data[f"{sensor_name}_z"] = float(raw_data[2])
			elif sensor_type == mujoco.mjtSensor.mjSENS_FRAMEANGVEL:
				imu_data[f"{sensor_name}_x"] = float(raw_data[0])
				imu_data[f"{sensor_name}_y"] = float(raw_data[1])
				imu_data[f"{sensor_name}_z"] = float(raw_data[2])
			else:
				# Generic fallback for other sensor types
				for i, val in enumerate(raw_data):
					imu_data[f"{sensor_name}_data_{i}"] = float(val)
				
		return imu_data

	def _get_obs(self) -> np.ndarray:
		return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

	def _sync_viewer_camera(self) -> None:
		if self._viewer is None:
			return
		self._viewer.cam.distance = float(self._camera.distance)
		self._viewer.cam.azimuth = float(self._camera.azimuth)
		self._viewer.cam.elevation = float(self._camera.elevation)
		self._viewer.cam.lookat[:] = self._camera.lookat

	def set_goal_angles(self, goal_angles: dict[str, float]) -> None:
		self.goal_angles = goal_angles
		self._target_ctrl = np.array([goal_angles[n] for n in self.control_joint_names], dtype=np.float64)

	def apply_gain_scales(self, kp_scale: float, kd_scale: float, gain_cfg: GainConfig) -> None:
		kp_scale = float(np.clip(kp_scale, gain_cfg.kp_scale_min, gain_cfg.kp_scale_max))
		kd_scale = float(np.clip(kd_scale, gain_cfg.kd_scale_min, gain_cfg.kd_scale_max))

		# Position actuator stiffness-like parameter.
		for aid in self.control_actuator_ids:
			self.model.actuator_gainprm[aid] = self.base_act_gainprm[aid]
			self.model.actuator_biasprm[aid] = self.base_act_biasprm[aid]
			self.model.actuator_gainprm[aid, 0] = self.base_act_gainprm[aid, 0] * kp_scale
			self.model.actuator_biasprm[aid, 1] = self.base_act_biasprm[aid, 1] * kp_scale
			self.model.actuator_biasprm[aid, 2] = self.base_act_biasprm[aid, 2] * kp_scale

		# Joint damping for controlled DOFs.
		for did in self.control_dof_ids:
			self.model.dof_damping[did] = self.base_dof_damping[did] * kd_scale

		self.current_kp_scale = kp_scale
		self.current_kd_scale = kd_scale
		self.is_in_compliant_mode = (
			abs(kp_scale - gain_cfg.kp_scale_compliant) < 1e-6
			and abs(kd_scale - gain_cfg.kd_scale_compliant) < 1e-6
		)

	def get_goal_errors(self) -> np.ndarray:
		current = self.data.qpos[self.control_qpos_ids]
		target = np.array([self.goal_angles[n] for n in self.control_joint_names], dtype=np.float64)
		return np.abs(current - target)

	def get_control_qvel(self) -> np.ndarray:
		return np.abs(self.data.qvel[self.control_dof_ids])

	def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
		super().reset(seed=seed)
		del options
		mujoco.mj_resetData(self.model, self.data)
		self.step_count = 0

		# Start controls from current joint positions for controlled actuators.
		self._target_ctrl = self.data.qpos[self.control_qpos_ids].copy()
		self.data.ctrl[self.control_actuator_ids] = self._target_ctrl
		mujoco.mj_forward(self.model, self.data)

		return self._get_obs(), {}

	def step(self, action: np.ndarray):
		if action.shape != self.action_space.shape:
			raise ValueError(f"Action shape mismatch: got {action.shape}, expected {self.action_space.shape}")

		# Optional additive action; experiment runner uses zeros.
		target = self._target_ctrl + action
		self.data.ctrl[self.control_actuator_ids] = target

		mujoco.mj_step(self.model, self.data)
		self.step_count += 1

		info = {
			"qfrc_actuator_l1": float(np.sum(np.abs(self.data.qfrc_actuator[self.control_dof_ids]))),
			"qfrc_constraint_l1": float(np.sum(np.abs(self.data.qfrc_constraint))),
		}
		info["total_load"] = (
			info["qfrc_actuator_l1"] + info["qfrc_constraint_l1"]
		)
		
		# Add IMU readings with detailed component information
		imu_readings = self._read_imu_data()
		info.update(imu_readings)
		
		# NEW: Add time-to-impact information
		t_impact, critical_body = self.compute_time_to_impact()
		info["time_to_impact"] = t_impact
		info["critical_body"] = critical_body if critical_body else "none"

		terminated = False
		truncated = False
		reward = 0.0
		return self._get_obs(), reward, terminated, truncated, info

	def render(self):
		if self.render_mode == "human":
			if self._viewer is None:
				if mj_viewer is None:
					raise RuntimeError("mujoco.viewer is unavailable for human rendering.")
				self._viewer = mj_viewer.launch_passive(self.model, self.data)
			self._sync_viewer_camera()
			self._viewer.sync()
			return None

		if self.render_mode == "rgb_array":
			if self._renderer is None:
				self._renderer = mujoco.Renderer(self.model, height=720, width=1280)
			self._renderer.update_scene(self.data, self._camera)
			return self._renderer.render()

		return None

	def close(self):
		if self._viewer is not None:
			self._viewer.close()
			self._viewer = None
		if self._renderer is not None:
			self._renderer.close()
			self._renderer = None


def compute_envelope(curves: list[np.ndarray]) -> dict[str, np.ndarray]:
	stacked = np.stack(curves, axis=0)
	return {
		"median": np.percentile(stacked, 50, axis=0),
		"p10": np.percentile(stacked, 10, axis=0),
		"p90": np.percentile(stacked, 90, axis=0),
	}


def _normalize_2d(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
	n = float(np.linalg.norm(v))
	if n < eps:
		return np.zeros(2, dtype=np.float64)
	return v / n


def _normalize_3d(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
	n = float(np.linalg.norm(v))
	if n < eps:
		return np.zeros(3, dtype=np.float64)
	return v / n


def _estimate_fall_direction_vector(
	quat_wxyz: np.ndarray,
	linvel_xy: np.ndarray,
	alpha_tilt: float = 0.85,
) -> np.ndarray:
	"""Estimate fall direction on ground plane from torso tilt + horizontal velocity."""
	w, x, y, z = quat_wxyz

	# Body +Z axis expressed in world frame from quaternion.
	up_x = 2.0 * (x * z + w * y)
	up_y = 2.0 * (y * z - w * x)

	# Ground-plane fall direction points opposite to projected up vector.
	v_tilt = _normalize_2d(np.array([-up_x, -up_y], dtype=np.float64))
	v_vel = _normalize_2d(linvel_xy.astype(np.float64, copy=False))

	if np.allclose(v_tilt, 0.0) and np.allclose(v_vel, 0.0):
		return np.zeros(2, dtype=np.float64)
	if np.allclose(v_tilt, 0.0):
		return v_vel
	if np.allclose(v_vel, 0.0):
		return v_tilt

	v = alpha_tilt * v_tilt + (1.0 - alpha_tilt) * v_vel
	return _normalize_2d(v)


def _wrap_to_pi(angle: float) -> float:
	return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _direction_to_quat(direction: np.ndarray) -> np.ndarray:
	"""Convert a 3D direction vector to a quaternion [w, x, y, z] that rotates
	the +X axis (cylinder default after euler='0 90 0') to point in that direction.
	"""
	direction = _normalize_3d(direction)
	if float(np.linalg.norm(direction)) < 1e-9:
		return np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
	
	# Default cylinder orientation after euler="0 90 0" is +X axis
	default_axis = np.array([1.0, 0.0, 0.0])
	
	# Rotation axis: cross product
	axis = np.cross(default_axis, direction)
	axis_norm = float(np.linalg.norm(axis))
	
	# If parallel or anti-parallel
	if axis_norm < 1e-9:
		if float(np.dot(default_axis, direction)) > 0:
			# Already aligned
			return np.array([1.0, 0.0, 0.0, 0.0])
		else:
			# 180 degree rotation - use Y axis
			return np.array([0.0, 0.0, 1.0, 0.0])
	
	axis = axis / axis_norm
	angle = np.arccos(np.clip(np.dot(default_axis, direction), -1.0, 1.0))
	
	# Quaternion from axis-angle
	half_angle = angle * 0.5
	w = np.cos(half_angle)
	xyz = axis * np.sin(half_angle)
	
	return np.array([w, xyz[0], xyz[1], xyz[2]])


def _estimate_fall_direction_vector_3d(
	quat_wxyz: np.ndarray,
	linvel_xyz: np.ndarray,
) -> np.ndarray:
	del quat_wxyz
	return _normalize_3d(linvel_xyz.astype(np.float64, copy=False))


def _resolve_body_id(model: mujoco.MjModel, candidates: list[str]) -> int | None:
	for name in candidates:
		body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
		if body_id >= 0:
			return body_id
	return None


def _resolve_site_id(model: mujoco.MjModel, candidates: list[str]) -> int | None:
	for name in candidates:
		site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
		if site_id >= 0:
			return site_id
	return None


def _try_get_torso_quat_and_linvel(info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
	q = np.array(
		[
			float(info.get("torso_quat_w", 1.0)),
			float(info.get("torso_quat_x", 0.0)),
			float(info.get("torso_quat_y", 0.0)),
			float(info.get("torso_quat_z", 0.0)),
		],
		dtype=np.float64,
	)
	qn = float(np.linalg.norm(q))
	if qn > 1e-9:
		q = q / qn

	v = np.array(
		[
			float(info.get("torso_linvel_x", 0.0)),
			float(info.get("torso_linvel_y", 0.0)),
			float(info.get("torso_linvel_z", 0.0)),
		],
		dtype=np.float64,
	)
	return q, v


def _get_arm_vector_and_critical(
	env: Op3FallControlEnv,
	side: str,
	left_ids: dict[str, int | None],
	right_ids: dict[str, int | None],
) -> tuple[np.ndarray, str]:
	ids = left_ids if side == "left" else right_ids
	shoulder_id = ids.get("shoulder")
	if shoulder_id is None:
		return np.zeros(3, dtype=np.float64), "none"

	candidates: list[tuple[str, np.ndarray]] = []
	hand_site_id = ids.get("hand_site")
	if hand_site_id is not None:
		candidates.append(("hand", env.data.site_xpos[hand_site_id].astype(np.float64)))
	hand_body_id = ids.get("hand_body")
	if hand_body_id is not None:
		candidates.append(("hand_body", env.data.xpos[hand_body_id].astype(np.float64)))
	elbow_id = ids.get("elbow")
	if elbow_id is not None:
		candidates.append(("elbow", env.data.xpos[elbow_id].astype(np.float64)))

	if not candidates:
		return np.zeros(3, dtype=np.float64), "none"

	critical_label, critical_pos = min(candidates, key=lambda item: float(item[1][2]))
	shoulder_pos = env.data.xpos[shoulder_id].astype(np.float64)
	arm_vec = _normalize_3d(critical_pos - shoulder_pos)
	return arm_vec, critical_label


def _alignment_score_from_angles(
	arm_angle: float,
	fall_angle: float,
	threshold_rad: float,
) -> tuple[float, float]:
	diff = abs(_wrap_to_pi(arm_angle - fall_angle))
	score = (1.0 - diff / max(threshold_rad, 1e-9)) if diff <= threshold_rad else 0.0
	return float(np.clip(score, 0.0, 1.0)), float(diff)


def _draw_vectors_programmatic(
	env: Op3FallControlEnv,
	fall_dir: np.ndarray,
	left_arm_dir: np.ndarray,
	right_arm_dir: np.ndarray,
	arrow_ids: dict[str, int],
	arrow_positions: dict[str, np.ndarray],
	torso_angvel_x: float,
	torso_angvel_y: float,
) -> None:
	"""Update arrow orientations while preserving fixed positions.
	fall_speed is calculated from angular velocity: sqrt(wx^2 + wy^2)
	"""
	fall_speed = float(np.sqrt(torso_angvel_x**2 + torso_angvel_y**2))
	
	# Update fall direction arrow (red) - always update
	if float(np.linalg.norm(fall_dir)) > 1e-9:
		fall_quat = _direction_to_quat(fall_dir)
		shaft_addr = arrow_ids["fall"]
		env.data.qpos[shaft_addr:shaft_addr+3] = arrow_positions["fall"]
		env.data.qpos[shaft_addr+3:shaft_addr+7] = fall_quat
	else:
		# Keep arrow position fixed
		shaft_addr = arrow_ids["fall"]
		env.data.qpos[shaft_addr:shaft_addr+3] = arrow_positions["fall"]
	
	# Update left arm arrow (green)
	if float(np.linalg.norm(left_arm_dir)) > 1e-9:
		left_quat = _direction_to_quat(left_arm_dir)
		shaft_addr = arrow_ids["left"]
		env.data.qpos[shaft_addr:shaft_addr+3] = arrow_positions["left"]
		env.data.qpos[shaft_addr+3:shaft_addr+7] = left_quat
	
	# Update right arm arrow (blue)
	if float(np.linalg.norm(right_arm_dir)) > 1e-9:
		right_quat = _direction_to_quat(right_arm_dir)
		shaft_addr = arrow_ids["right"]
		env.data.qpos[shaft_addr:shaft_addr+3] = arrow_positions["right"]
		env.data.qpos[shaft_addr+3:shaft_addr+7] = right_quat
	
	mujoco.mj_forward(env.model, env.data)


def run_experiments() -> None:
	root = Path(__file__).resolve().parent
	scene_xml = root / "robotis_op3" / "scene.xml"
	out_dir = root / "outputs"
	imu_out_dir = out_dir / "IMU" / "experiment_freefall"
	imu_out_dir.mkdir(parents=True, exist_ok=True)
	out_dir.mkdir(parents=True, exist_ok=True)

	num_runs = 4
	max_steps = 125
	base_seed = 1234

	gain_cfg = GainConfig()
	impact_cfg = ImpactConfig()
	gate_cfg = GoalGateConfig()
	restore_cfg = RestoreConfig()
	push_cfg = PushConfig()
	tti_cfg = TimeToImpactConfig()  # Time-to-impact config

	env = Op3FallControlEnv(
		model_xml=scene_xml,
		goal_angles=GOAL_ANGLES_ARMS,
		render_mode="human",
		control_timestep=0.02,
		camera_distance=1.5,
		camera_azimuth=160.0,
		camera_elevation=-20.0,
	)

	left_ids = {
		"shoulder": _resolve_body_id(env.model, ["l_sho_roll_link", "l_sho_pitch_link"]),
		"elbow": _resolve_body_id(env.model, ["l_el_link"]),
		"hand_site": _resolve_site_id(env.model, ["l_hand"]),
		"hand_body": _resolve_body_id(env.model, ["l_hand_link", "l_grip_link"]),
	}
	right_ids = {
		"shoulder": _resolve_body_id(env.model, ["r_sho_roll_link", "r_sho_pitch_link"]),
		"elbow": _resolve_body_id(env.model, ["r_el_link"]),
		"hand_site": _resolve_site_id(env.model, ["r_hand"]),
		"hand_body": _resolve_body_id(env.model, ["r_hand_link", "r_grip_link"]),
	}

	# Logs by cohort.
	stiff_total: list[np.ndarray] = []
	compliant_total: list[np.ndarray] = []
	stiff_m1: list[np.ndarray] = []
	compliant_m1: list[np.ndarray] = []
	stiff_m2: list[np.ndarray] = []
	compliant_m2: list[np.ndarray] = []
	
	# Store time-to-impact data
	tti_history: dict[int, list[float]] = {}  # run_idx -> list of tti values
	critical_body_history: dict[int, list[str]] = {}  # run_idx -> list of critical bodies
	
	# Store final limb Z-positions for each run
	final_limb_positions: dict[str, list[float]] = {
		"left_hand": [], "right_hand": [],
		"left_elbow": [], "right_elbow": [],
		"left_knee": [], "right_knee": [],
		"torso": [], "head": [],
		"left_foot": [], "right_foot": []
	}

	# Get arrow shaft body IDs and qpos addresses for visualization
	arrow_ids = {}
	arrow_positions = {}
	for name in ["fall", "left", "right"]:
		shaft_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"arrow_{name}_shaft")
		if shaft_body_id >= 0:
			shaft_jnt_id = env.model.body_jntadr[shaft_body_id]
			qpos_addr = env.model.jnt_qposadr[shaft_jnt_id]
			arrow_ids[name] = qpos_addr
			# Store initial position from qpos
			arrow_positions[name] = env.data.qpos[qpos_addr:qpos_addr+3].copy()
		else:
			print(f"Warning: Arrow '{name}' not found in scene")
	arrows_available = all(name in arrow_ids for name in ("fall", "left", "right"))

	try:
		for run_idx in range(num_runs):
			compliant_run = run_idx >= num_runs // 2
			env.reset(seed=base_seed + run_idx)
			env.set_goal_angles(GOAL_ANGLES_ARMS)
			env.apply_gain_scales(gain_cfg.kp_scale_stiff, gain_cfg.kd_scale_stiff, gain_cfg)

			push_steps = int(np.ceil(push_cfg.duration_sec / env.model.opt.timestep))

			m1_list: list[float] = []
			m2_list: list[float] = []
			total_list: list[float] = []
			spike_flags: list[bool] = []
			
			# NEW: Store time-to-impact data for this run
			tti_list: list[float] = []
			critical_body_list: list[str] = []
			
			# Fall dynamics tracking
			fall_speed_list: list[float] = []
			fall_angle_list: list[float] = []
			fall_vec_filt = np.array([0.0, 0.0], dtype=np.float64)
			fall_angle_locked: float | None = None
			fall_lock_active = False

			goal_count = 0
			goal_reached_step: int | None = None
			switched_to_compliance = False
			compliance_step: int | None = None

			rolling = deque(maxlen=impact_cfg.rolling_window)
			refractory = 0
			last_total = 1e-6
			last_spike_step: int | None = None
			low_load_count = 0
			restore_started = False
			restore_step = 0
			restore_start_step: int | None = None
			impact_detected = False

			for step in range(max_steps):
				# Initial push at head.
				env.data.xfrc_applied[:, :] = 0.0
				if step < push_steps:
					env.data.xfrc_applied[env.head_body_id, :3] = np.array(push_cfg.force_xyz)

				obs, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float64))
				del obs

				torso_q, torso_linvel = _try_get_torso_quat_and_linvel(info)
				fall_dir_3d = _estimate_fall_direction_vector_3d(torso_q, torso_linvel)
				left_arm_dir, _ = _get_arm_vector_and_critical(env, "left", left_ids, right_ids)
				right_arm_dir, _ = _get_arm_vector_and_critical(env, "right", left_ids, right_ids)
				torso_angvel_x = float(info.get("torso_angvel_x", 0.0))
				torso_angvel_y = float(info.get("torso_angvel_y", 0.0))
				if arrows_available:
					_draw_vectors_programmatic(
						env,
						fall_dir_3d,
						left_arm_dir,
						right_arm_dir,
						arrow_ids,
						arrow_positions,
						torso_angvel_x,
						torso_angvel_y,
					)
				env.render()

				m1 = info["qfrc_actuator_l1"]
				m2 = info["qfrc_constraint_l1"]
				total = info["total_load"]
				t_impact = info["time_to_impact"]
				critical_body = info["critical_body"]

				m1_list.append(m1)
				m2_list.append(m2)
				total_list.append(total)
				tti_list.append(t_impact)
				critical_body_list.append(critical_body)

				# Keep original fall-speed definition from angular velocity.
				wx = float(info.get("torso_angvel_x", 0.0))
				wy = float(info.get("torso_angvel_y", 0.0))
				fall_speed = float(np.sqrt(wx**2 + wy**2))

				# NEW: Trigger compliance based on time-to-impact
				if compliant_run and (not switched_to_compliance) and t_impact <= tti_cfg.threshold:
					env.apply_gain_scales(gain_cfg.kp_scale_compliant, gain_cfg.kd_scale_compliant, gain_cfg)
					switched_to_compliance = True
					impact_detected = True
					compliance_step = step
					print(f"Run {run_idx} - Time-to-impact triggered at step {step}: t_impact={t_impact:.3f}s <= {tti_cfg.threshold}s (critical body: {critical_body})")

				# Moderate goal gate.
				err = env.get_goal_errors()
				qv = env.get_control_qvel()
				reached_now = (
					float(np.mean(err)) <= gate_cfg.mean_err_thr
					and float(np.max(err)) <= gate_cfg.max_err_thr
					and float(np.mean(qv)) <= gate_cfg.mean_qvel_thr
				)
				goal_count = goal_count + 1 if reached_now else 0
				if goal_reached_step is None and goal_count >= gate_cfg.consecutive_steps:
					goal_reached_step = step

				# Impact spike detection on total load (backup method)
				is_spike = False
				if len(rolling) >= impact_cfg.rolling_window:
					mu = float(np.mean(rolling))
					sigma = float(np.std(rolling) + 1e-9)
					jump = float(total / max(last_total, 1e-9))
					if refractory == 0 and total > (mu + impact_cfg.sigma_multiplier * sigma) and jump > impact_cfg.jump_ratio:
						is_spike = True
						refractory = impact_cfg.refractory_steps
						last_spike_step = step
						# Spike-based compliance trigger (backup)
						if compliant_run and not switched_to_compliance:
							env.apply_gain_scales(gain_cfg.kp_scale_compliant, gain_cfg.kd_scale_compliant, gain_cfg)
							switched_to_compliance = True
							compliance_step = step
							impact_detected = True
							print(f"Run {run_idx} - Force spike triggered at step {step}")
       
				rolling.append(total)
				spike_flags.append(is_spike)
				refractory = max(0, refractory - 1)

				# Robust fall-angle estimate (tilt + velocity, filtered, and locked on impact).
				if not fall_lock_active:
					qw = float(info.get("torso_quat_w", 1.0))
					qx = float(info.get("torso_quat_x", 0.0))
					qy = float(info.get("torso_quat_y", 0.0))
					qz = float(info.get("torso_quat_z", 0.0))
					q = np.array([qw, qx, qy, qz], dtype=np.float64)
					q_norm = float(np.linalg.norm(q))
					if q_norm > 1e-9:
						q = q / q_norm

					vx = float(info.get("torso_linvel_x", 0.0))
					vy = float(info.get("torso_linvel_y", 0.0))
					v_inst = _estimate_fall_direction_vector(q, np.array([vx, vy], dtype=np.float64), alpha_tilt=0.85)

					beta = 0.12
					fall_vec_filt = (1.0 - beta) * fall_vec_filt + beta * v_inst
					fall_vec_filt = _normalize_2d(fall_vec_filt)

					if is_spike and float(np.linalg.norm(fall_vec_filt)) > 1e-6:
						fall_angle_locked = float(np.arctan2(fall_vec_filt[1], fall_vec_filt[0]))
						fall_lock_active = True

				if fall_lock_active and fall_angle_locked is not None:
					fall_angle = fall_angle_locked
				else:
					if float(np.linalg.norm(fall_vec_filt)) > 1e-6:
						fall_angle = float(np.arctan2(fall_vec_filt[1], fall_vec_filt[0]))
					else:
						fall_angle = 0.0

				fall_speed_list.append(fall_speed)
				fall_angle_list.append(fall_angle)

				# Restoration condition after loads subside (for compliant runs only)
				if switched_to_compliance and impact_detected and not restore_started:
					baseline = float(np.mean(total_list[: impact_cfg.rolling_window])) if len(total_list) >= impact_cfg.rolling_window else float(np.mean(total_list))
					no_recent_spike = (
						last_spike_step is None or (step - last_spike_step) >= restore_cfg.no_spike_steps
					)
					if total < restore_cfg.baseline_multiplier * baseline:
						low_load_count += 1
					else:
						low_load_count = 0

					if no_recent_spike and low_load_count >= restore_cfg.low_load_steps:
						restore_started = True
						restore_start_step = step
						restore_step = 0

				if restore_started:
					alpha = min(1.0, restore_step / max(1, restore_cfg.ramp_steps))
					kp_scale = gain_cfg.kp_scale_compliant + alpha * (gain_cfg.kp_scale_stiff - gain_cfg.kp_scale_compliant)
					kd_scale = gain_cfg.kd_scale_compliant + alpha * (gain_cfg.kd_scale_stiff - gain_cfg.kd_scale_compliant)
					env.apply_gain_scales(kp_scale, kd_scale, gain_cfg)
					restore_step += 1

				last_total = total

			# Store time-to-impact history
			tti_history[run_idx] = tti_list
			critical_body_history[run_idx] = critical_body_list

			# Record final limb positions
			for limb_name, body_id in env.body_ids_for_impact.items():
				if body_id >= 0:
					z_position = float(env.data.xpos[body_id, 2])
					final_limb_positions[limb_name].append(z_position)
					print(f"Run {run_idx} - {limb_name} final Z-position: {z_position:.4f}")
				else:
					final_limb_positions[limb_name].append(float('nan'))

			# Plot time-to-impact data
			t = np.arange(max_steps)
			fig, ax = plt.subplots(2, 1, figsize=(12, 8))
			
			# Time-to-impact
			ax[0].plot(t, tti_list, label="time_to_impact", color="green", linewidth=2)
			ax[0].axhline(y=tti_cfg.threshold, color="red", linestyle="--", alpha=0.7, label=f"threshold ({tti_cfg.threshold}s)")
			if compliance_step is not None:
				ax[0].axvline(compliance_step, color="purple", linestyle=":", alpha=0.6, label="compliance ON")
			ax[0].set_ylabel("Time to Impact (s)")
			ax[0].set_title(f"Run {run_idx}: Time-to-Impact")
			ax[0].grid(True, alpha=0.3)
			ax[0].legend(loc="upper right")
			ax[0].set_ylim(bottom=0)
			
			# Critical body (as categorical)
			# Convert to numeric for plotting
			unique_bodies = list(set(critical_body_list))
			body_to_num = {body: i for i, body in enumerate(unique_bodies)}
			critical_numeric = [body_to_num.get(b, -1) for b in critical_body_list]
			
			ax[1].scatter(t, critical_numeric, s=10, alpha=0.5)
			ax[1].set_yticks(range(len(unique_bodies)))
			ax[1].set_yticklabels(unique_bodies)
			ax[1].set_ylabel("Critical Body")
			ax[1].set_xlabel("Timestep")
			ax[1].set_title("Critical Body (closest to impact)")
			ax[1].grid(True, alpha=0.3)
			
			fig.tight_layout()
			fig.savefig(imu_out_dir / f"run_{run_idx:02d}_time_to_impact.png", dpi=150)
			plt.close(fig)

			# Original load plots
			fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
			ax[0].plot(t, m1_list, label="qfrc_actuator (L1)")
			ax[1].plot(t, m2_list, label="qfrc_constraint (L1)", color="tab:orange")
			ax[2].plot(t, total_list, label="total_load", color="tab:red")

			for i in range(3):
				ax[i].grid(True, alpha=0.3)
				# Add vertical lines for impact spikes
				impact_labeled = False
				for step, is_spike in enumerate(spike_flags):
					if is_spike:
						label = "force spike" if not impact_labeled else None
						ax[i].axvline(step, color="red", linestyle="-", alpha=0.3, linewidth=1, label=label)
						impact_labeled = True
				if goal_reached_step is not None:
					ax[i].axvline(goal_reached_step, color="k", linestyle="--", alpha=0.4, label="goal reached")
				if compliance_step is not None:
					ax[i].axvline(compliance_step, color="purple", linestyle=":", alpha=0.6, label="compliance ON")
				if restore_start_step is not None:
					ax[i].axvline(restore_start_step, color="brown", linestyle="-.", alpha=0.6, label="restore gains")
				ax[i].legend(loc="upper right")

			ax[-1].set_xlabel("Timestep")
			ax[0].set_title(f"Run {run_idx}: {'compliant' if compliant_run else 'stiff'}")
			fig.tight_layout()
			fig.savefig(out_dir / f"run_{run_idx:02d}_loads.png", dpi=150)
			plt.close(fig)

			# Fall dynamics plot
			fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
			axs[0].plot(t, fall_speed_list, label="fall_speed", color="tab:red", linewidth=2)
			axs[0].set_ylabel("Fall Speed (rad/s)")
			axs[0].grid(True, alpha=0.3)
			axs[0].legend(loc="upper right")
			if compliance_step is not None:
				axs[0].axvline(compliance_step, color="purple", linestyle=":", alpha=0.6)

			axs[1].plot(t, fall_angle_list, label="fall_angle", color="tab:blue", linewidth=2)
			axs[1].set_ylabel("Fall Angle (radians)")
			axs[1].set_xlabel("Timestep")
			axs[1].grid(True, alpha=0.3)
			axs[1].legend(loc="upper right")
			if compliance_step is not None:
				axs[1].axvline(compliance_step, color="purple", linestyle=":", alpha=0.6)

			fig.suptitle(f"Run {run_idx}: Fall Dynamics")
			fig.tight_layout()
			fig.savefig(imu_out_dir / f"run_{run_idx:02d}_fall_dynamics.png", dpi=150)
			plt.close(fig)

			m1_arr = np.array(m1_list)
			m2_arr = np.array(m2_list)
			total_arr = np.array(total_list)
			if compliant_run:
				compliant_m1.append(m1_arr)
				compliant_m2.append(m2_arr)
				compliant_total.append(total_arr)
			else:
				stiff_m1.append(m1_arr)
				stiff_m2.append(m2_arr)
				stiff_total.append(total_arr)

		# Summary statistics
		print("\n" + "="*60)
		print("EXPERIMENT SUMMARY")
		print("="*60)
		
		# Time-to-impact statistics
		print("\nTime-to-Impact Statistics:")
		for run_idx, tti_vals in tti_history.items():
			valid_tti = [t for t in tti_vals if t < float('inf')]
			if valid_tti:
				print(f"  Run {run_idx}: min={min(valid_tti):.3f}s, mean={np.mean(valid_tti):.3f}s, triggered={any(t <= tti_cfg.threshold for t in valid_tti)}")
		
		# Final limb positions
		print("\nFinal Limb Z-Positions (averages):")
		for limb_name, positions in final_limb_positions.items():
			if positions and not all(np.isnan(p) for p in positions):
				print(f"  {limb_name}: {np.nanmean(positions):.4f}m")

	finally:
		env.close()

class Op3FallControlArmsRL(gym.Env):
	"""RL wrapper around Op3FallControlEnv for forward-fall, arms-only PPO training."""

	metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

	def __init__(
		self,
		model_xml: str | Path,
		goal_angles: dict[str, float],
		reward_cfg: RewardConfig | None = None,
		max_episode_steps: int = 250,
		seed: int | None = None,
	) -> None:
		super().__init__()
		self.reward_cfg = reward_cfg or RewardConfig()
		self.max_episode_steps = max_episode_steps
		self._episode_step = 0
		self.align_cfg = AlignConfig()
		# Ensure Gym / SB3 see this env as rgb_array-only for recording.
		self.render_mode = "rgb_array"

		self.gain_cfg = GainConfig()
		self.push_cfg = PushConfig()
		self.tti_cfg = TimeToImpactConfig()
		self._switched_to_compliance = False
		self._push_steps = 0

		self.core_env = Op3FallControlEnv(
			model_xml=model_xml,
			goal_angles=goal_angles,
			render_mode="rgb_array",
			control_timestep=0.02,
			camera_distance=1.5,
			camera_azimuth=160.0,
			camera_elevation=-20.0,
		)

		if seed is not None:
			self.core_env.reset(seed=seed)

		self._left_ids = {
			"shoulder": _resolve_body_id(self.core_env.model, ["l_sho_roll_link", "l_sho_pitch_link"]),
			"elbow": _resolve_body_id(self.core_env.model, ["l_el_link"]),
			"hand_site": _resolve_site_id(self.core_env.model, ["l_hand"]),
			"hand_body": _resolve_body_id(self.core_env.model, ["l_hand_link", "l_grip_link"]),
		}
		self._right_ids = {
			"shoulder": _resolve_body_id(self.core_env.model, ["r_sho_roll_link", "r_sho_pitch_link"]),
			"elbow": _resolve_body_id(self.core_env.model, ["r_el_link"]),
			"hand_site": _resolve_site_id(self.core_env.model, ["r_hand"]),
			"hand_body": _resolve_body_id(self.core_env.model, ["r_hand_link", "r_grip_link"]),
		}

		# Arrow bodies (free joints in scene.xml) must be re-pinned every step.
		self._arrow_ids: dict[str, int] = {}
		self._arrow_positions: dict[str, np.ndarray] = {}
		for name in ["fall", "left", "right"]:
			shaft_body_id = mujoco.mj_name2id(self.core_env.model, mujoco.mjtObj.mjOBJ_BODY, f"arrow_{name}_shaft")
			if shaft_body_id >= 0:
				shaft_jnt_id = self.core_env.model.body_jntadr[shaft_body_id]
				qpos_addr = self.core_env.model.jnt_qposadr[shaft_jnt_id]
				self._arrow_ids[name] = qpos_addr
		self._arrows_available = all(name in self._arrow_ids for name in ("fall", "left", "right"))

		self.control_joint_names = list(goal_angles.keys())
		self._joint_indices = [self.core_env.qpos_ids[n] for n in self.control_joint_names]
		self._dof_indices = [self.core_env.dof_ids[n] for n in self.control_joint_names]

		# Action: normalized joint position deltas in [-1, 1] for each controlled joint.
		self.max_joint_delta = np.array([0.2] * len(self.control_joint_names), dtype=np.float64)
		self.action_space = spaces.Box(
			low=-1.0,
			high=1.0,
			shape=(len(self.control_joint_names),),
			dtype=np.float32,
		)

		# Observation components:
		# - joint angles (arms + hips/knees used in GOAL_ANGLES)
		# - joint velocities for those joints
		# - fall_speed (from torso angvel)
		# - time_to_impact
		# - fall_angle (estimated from torso quat + linvel_xy)
		# - is_in_compliant_mode flag
		self.obs_dim = 2 * len(self.control_joint_names) + 4
		self.observation_space = spaces.Box(
			low=-np.inf,
			high=np.inf,
			shape=(self.obs_dim,),
			dtype=np.float32,
		)

	def _update_scene_arrows(self, info: Dict[str, Any]) -> None:
		if not self._arrows_available:
			return

		torso_q, torso_linvel = _try_get_torso_quat_and_linvel(info)
		fall_dir_3d = _estimate_fall_direction_vector_3d(torso_q, torso_linvel)
		left_arm_dir, _ = _get_arm_vector_and_critical(self.core_env, "left", self._left_ids, self._right_ids)
		right_arm_dir, _ = _get_arm_vector_and_critical(self.core_env, "right", self._left_ids, self._right_ids)
		torso_angvel_x = float(info.get("torso_angvel_x", 0.0))
		torso_angvel_y = float(info.get("torso_angvel_y", 0.0))

		_draw_vectors_programmatic(
			self.core_env,
			fall_dir_3d,
			left_arm_dir,
			right_arm_dir,
			self._arrow_ids,
			self._arrow_positions,
			torso_angvel_x,
			torso_angvel_y,
		)

	def _build_obs(self, info: Dict[str, Any]) -> np.ndarray:
		qpos = self.core_env.data.qpos[self._joint_indices].astype(np.float64)
		qvel = self.core_env.data.qvel[self._dof_indices].astype(np.float64)

		# Normalize joint angles and velocities
		qpos_norm = qpos / np.pi
		qvel_norm = qvel / 10.0

		# Fall speed from torso angular velocity
		wx = float(info.get("torso_angvel_x", 0.0))
		wy = float(info.get("torso_angvel_y", 0.0))
		fall_speed = float(np.sqrt(wx**2 + wy**2))
		fall_speed_norm = fall_speed / max(self.reward_cfg.max_fall_speed, 1e-6)

		# Time to impact
		t_impact = float(info.get("time_to_impact", float("inf")))
		if np.isinf(t_impact):
			tti_norm = 1.0
		else:
			tti_norm = np.clip(t_impact / max(self.reward_cfg.max_head_vel, 1e-6), 0.0, 1.0)

		# Fall angle (reuse same estimator as in run_experiments)
		qw = float(info.get("torso_quat_w", 1.0))
		qx = float(info.get("torso_quat_x", 0.0))
		qy = float(info.get("torso_quat_y", 0.0))
		qz = float(info.get("torso_quat_z", 0.0))
		q = np.array([qw, qx, qy, qz], dtype=np.float64)
		q_norm = float(np.linalg.norm(q))
		if q_norm > 1e-9:
			q = q / q_norm

		vx = float(info.get("torso_linvel_x", 0.0))
		vy = float(info.get("torso_linvel_y", 0.0))
		v_inst = _estimate_fall_direction_vector(q, np.array([vx, vy], dtype=np.float64), alpha_tilt=0.85)
		if float(np.linalg.norm(v_inst)) > 1e-6:
			fall_angle = float(np.arctan2(v_inst[1], v_inst[0]))
		else:
			fall_angle = 0.0

		fall_angle_norm = fall_angle / np.pi

		compliant_flag = 1.0 if getattr(self.core_env, "is_in_compliant_mode", False) else 0.0

		return np.concatenate(
			[
				qpos_norm,
				qvel_norm,
				np.array([fall_speed_norm, tti_norm, fall_angle_norm, compliant_flag], dtype=np.float64),
			]
		).astype(np.float32)

	def _compute_reward(self, info: Dict[str, Any]) -> float:
		# Approximate head vertical velocity as impact velocity proxy
		head_body_id = self.core_env.head_body_id
		head_vel_z = float(self.core_env.data.cvel[head_body_id, 2])
		head_impact_vel = max(0.0, -head_vel_z)

		total_load = float(info.get("total_load", 0.0))

		wx = float(info.get("torso_angvel_x", 0.0))
		wy = float(info.get("torso_angvel_y", 0.0))
		fall_speed = float(np.sqrt(wx**2 + wy**2))

		# Normalize for reward scaling
		head_term = head_impact_vel / max(self.reward_cfg.max_head_vel, 1e-6)
		torque_term = total_load / max(self.reward_cfg.max_torque, 1e-6)
		speed_term = fall_speed / max(self.reward_cfg.max_fall_speed, 1e-6)

		# Base penalty terms
		reward = -(
			self.reward_cfg.w_head * head_term
			+ self.reward_cfg.w_torque * torque_term
			+ self.reward_cfg.w_speed * speed_term
		)

		# Arms alignment reward: pre-impact only.
		t_impact = float(info.get("time_to_impact", float("inf")))
		pre_impact = (not self._switched_to_compliance) and (t_impact > self.tti_cfg.threshold)
		if pre_impact:
			torso_q, torso_linvel = _try_get_torso_quat_and_linvel(info)
			raw_fall_dir = _estimate_fall_direction_vector_3d(torso_q, torso_linvel)

			if float(np.linalg.norm(raw_fall_dir)) > 1e-9:
				fall_dir = _normalize_3d(raw_fall_dir)

				left_arm_dir, _ = _get_arm_vector_and_critical(self.core_env, "left", self._left_ids, self._right_ids)
				right_arm_dir, _ = _get_arm_vector_and_critical(self.core_env, "right", self._left_ids, self._right_ids)

				# Vector azimuths in XY plane for alignment angle.
				if float(np.linalg.norm(fall_dir[:2])) > 1e-9:
					fall_angle = float(np.arctan2(fall_dir[1], fall_dir[0]))
				else:
					fall_angle = 0.0

				if float(np.linalg.norm(left_arm_dir[:2])) > 1e-9:
					left_arm_angle = float(np.arctan2(left_arm_dir[1], left_arm_dir[0]))
				else:
					left_arm_angle = 0.0

				if float(np.linalg.norm(right_arm_dir[:2])) > 1e-9:
					right_arm_angle = float(np.arctan2(right_arm_dir[1], right_arm_dir[0]))
				else:
					right_arm_angle = 0.0

				left_rew, _ = _alignment_score_from_angles(
					left_arm_angle,
					fall_angle,
					self.align_cfg.angle_threshold_rad,
				)
				right_rew, _ = _alignment_score_from_angles(
					right_arm_angle,
					fall_angle,
					self.align_cfg.angle_threshold_rad,
				)
				align_score = 0.5 * (left_rew + right_rew)
				reward += self.reward_cfg.w_align * align_score

		return float(reward)

	def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
		del options
		if seed is not None:
			super().reset(seed=seed)
		core_obs, core_info = self.core_env.reset(seed=seed)
		self.core_env.apply_gain_scales(
			self.gain_cfg.kp_scale_stiff,
			self.gain_cfg.kd_scale_stiff,
			self.gain_cfg,
		)
		self._switched_to_compliance = False
		self._push_steps = int(np.ceil(self.push_cfg.duration_sec / self.core_env.model.opt.timestep))
		del core_obs
		self._episode_step = 0
		if self._arrows_available:
			for name, qpos_addr in self._arrow_ids.items():
				self._arrow_positions[name] = self.core_env.data.qpos[qpos_addr:qpos_addr+3].copy()
			self._update_scene_arrows(core_info)
		obs = self._build_obs(core_info)
		return obs, {}

	def step(self, action: np.ndarray):
		if action.shape != self.action_space.shape:
			raise ValueError(f"Action shape mismatch: got {action.shape}, expected {self.action_space.shape}")

		action = np.asarray(action, dtype=np.float64)
		delta = np.clip(action, -1.0, 1.0) * self.max_joint_delta

		# Apply forward push at head for the initial phase of the episode.
		self.core_env.data.xfrc_applied[:, :] = 0.0
		if self._episode_step < self._push_steps:
			self.core_env.data.xfrc_applied[self.core_env.head_body_id, :3] = np.array(self.push_cfg.force_xyz)

		current_q = self.core_env.data.qpos[self._joint_indices].copy()
		target_q = current_q + delta
		self.core_env._target_ctrl = target_q
		self.core_env.data.ctrl[self.core_env.control_actuator_ids] = target_q

		core_obs, _, _, _, info = self.core_env.step(np.zeros_like(self.core_env.action_space.low, dtype=np.float64))
		del core_obs
		self._update_scene_arrows(info)

		# Time-to-impact based compliance trigger
		t_impact = float(info.get("time_to_impact", float("inf")))
		if (not self._switched_to_compliance) and (t_impact <= self.tti_cfg.threshold):
			self.core_env.apply_gain_scales(
				self.gain_cfg.kp_scale_compliant,
				self.gain_cfg.kd_scale_compliant,
				self.gain_cfg,
			)
			self._switched_to_compliance = True

		self._episode_step += 1
		obs = self._build_obs(info)
		reward = self._compute_reward(info)

		terminated = False
		truncated = self._episode_step >= self.max_episode_steps

		return obs, reward, terminated, truncated, info

	def render(self):
		return self.core_env.render()

	def close(self):
		self.core_env.close()


class EpisodeCounterCallback(BaseCallback):
	def __init__(
		self,
		checkpoint_dir: Path,
		video_dir: Path,
		env_fn: Callable[[], gym.Env],
		n_envs: int,
		checkpoint_interval_episodes: int,
		video_interval_episodes: int,
		verbose: int = 0,
	) -> None:
		super().__init__(verbose=verbose)
		self.checkpoint_dir = checkpoint_dir
		self.video_dir = video_dir
		self.env_fn = env_fn
		self.n_envs = n_envs
		self.checkpoint_interval_episodes = max(1, checkpoint_interval_episodes)
		self.video_interval_episodes = max(1, video_interval_episodes)
		self.episode_count = 0
		self.episode_rewards: List[float] = []
		self.episode_peak_loads: List[float] = []
		self._episode_rewards = np.zeros(self.n_envs, dtype=np.float64)
		self._episode_peak_loads = np.zeros(self.n_envs, dtype=np.float64)

	def _on_step(self) -> bool:
		if self.locals is None:
			return True
		dones = self.locals.get("dones")
		rewards = self.locals.get("rewards")
		infos = self.locals.get("infos")
		if dones is None or rewards is None:
			return True

		rewards_arr = np.asarray(rewards, dtype=np.float64)
		for i in range(self.n_envs):
			self._episode_rewards[i] += float(rewards_arr[i])
			if infos is not None and i < len(infos):
				info_i = infos[i] or {}
				load_i = float(info_i.get("total_load", 0.0))
				if load_i > self._episode_peak_loads[i]:
					self._episode_peak_loads[i] = load_i

		dones_arr = np.asarray(dones, dtype=bool)
		if np.any(dones_arr):
			for i, done in enumerate(dones_arr):
				if not done:
					continue
				self.episode_count += 1
				self.episode_rewards.append(float(self._episode_rewards[i]))
				self.episode_peak_loads.append(float(self._episode_peak_loads[i]))
				self._episode_rewards[i] = 0.0
				self._episode_peak_loads[i] = 0.0

				if self.episode_count % self.checkpoint_interval_episodes == 0:
					if self.model is not None:
						filename = self.checkpoint_dir / f"ppo_ep_{self.episode_count:06d}"
						self.model.save(str(filename))

				if self.episode_count % self.video_interval_episodes == 0 and VecVideoRecorder is not None:
					self._record_video()

		return True

	def _record_video(self) -> None:
		if self.model is None:
			return
		if DummyVecEnv is None:
			return

		eval_env = DummyVecEnv([self.env_fn])
		# VecVideoRecorder expects env.render_mode == "rgb_array".
		setattr(eval_env, "render_mode", "rgb_array")
		video_length = 250

		if VecVideoRecorder is not None:
			eval_env = VecVideoRecorder(
				eval_env,
				video_folder=str(self.video_dir),
				record_video_trigger=lambda _: True,
				video_length=video_length,
				name_prefix=f"ep_{self.episode_count:06d}",
			)

		obs = eval_env.reset()
		for _ in range(video_length):
			action, _ = self.model.predict(obs, deterministic=True)
			obs, _, dones, _ = eval_env.step(action)
			if np.any(dones):
				break
		eval_env.close()


def train_ppo_forward_fall_arms(
	training_cfg: RLTrainingConfig | None = None,
	reward_cfg: RewardConfig | None = None,
) -> None:
	if PPO is None or DummyVecEnv is None or Monitor is None:
		raise RuntimeError(
			"Stable-Baselines3 and its wrappers are required for training but could not be imported."
		)

	training_cfg = training_cfg or RLTrainingConfig()
	reward_cfg = reward_cfg or RewardConfig()

	root = Path(__file__).resolve().parent
	scene_xml = root / "robotis_op3" / "scene.xml"

	run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_dir = root / "runs" / run_ts
	checkpoint_dir = run_dir / "checkpoints"
	video_dir = run_dir / "videos"
	log_dir = run_dir / "logs"

	checkpoint_dir.mkdir(parents=True, exist_ok=True)
	video_dir.mkdir(parents=True, exist_ok=True)
	log_dir.mkdir(parents=True, exist_ok=True)

	def make_env(rank: int) -> Callable[[], gym.Env]:
		def _init() -> gym.Env:
			env = Op3FallControlArmsRL(
				model_xml=scene_xml,
				goal_angles=GOAL_ANGLES_ARMS,
				reward_cfg=reward_cfg,
				max_episode_steps=training_cfg.max_episode_steps,
				seed=rank,
			)
			return Monitor(env)

		return _init

	if training_cfg.n_envs > 1 and SubprocVecEnv is not None:
		env_fns: List[Callable[[], gym.Env]] = [make_env(i) for i in range(training_cfg.n_envs)]
		vec_env = SubprocVecEnv(env_fns)
	else:
		vec_env = DummyVecEnv([make_env(0)])

	actual_n_envs = getattr(vec_env, "num_envs", training_cfg.n_envs)

	model = PPO(
		"MlpPolicy",
		vec_env,
		n_steps=training_cfg.n_steps,
		batch_size=training_cfg.batch_size,
		learning_rate=training_cfg.learning_rate,
		gamma=training_cfg.gamma,
		gae_lambda=training_cfg.gae_lambda,
		verbose=1,
		tensorboard_log=str(log_dir),
		device="cpu",
	)

	callback = EpisodeCounterCallback(
		checkpoint_dir=checkpoint_dir,
		video_dir=video_dir,
		env_fn=make_env(0),
		n_envs=actual_n_envs,
		checkpoint_interval_episodes=training_cfg.checkpoint_interval_episodes,
		video_interval_episodes=training_cfg.video_interval_episodes,
	)

	model.learn(total_timesteps=training_cfg.total_timesteps, callback=callback)
	vec_env.close()

	# Plot training statistics.
	if callback.episode_rewards:
		episodes = np.arange(1, len(callback.episode_rewards) + 1)
		plt.figure(figsize=(8, 4))
		plt.plot(episodes, callback.episode_rewards)
		plt.xlabel("Episode")
		plt.ylabel("Total Reward")
		plt.title("Training Reward per Episode")
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.savefig(run_dir / "reward_per_episode.png", dpi=150)
		plt.close()

	if callback.episode_peak_loads:
		episodes = np.arange(1, len(callback.episode_peak_loads) + 1)
		plt.figure(figsize=(8, 4))
		plt.plot(episodes, callback.episode_peak_loads)
		plt.xlabel("Episode")
		plt.ylabel("Peak Total Load")
		plt.title("Peak Load per Episode")
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.savefig(run_dir / "peak_load_per_episode.png", dpi=150)
		plt.close()


if __name__ == "__main__":
	train_ppo_forward_fall_arms()