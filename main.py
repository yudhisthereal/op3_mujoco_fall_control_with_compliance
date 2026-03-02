from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections import deque

import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import mujoco
import numpy as np

try:
	import mujoco.viewer as mj_viewer
except Exception:  # pragma: no cover
	mj_viewer = None


GOAL_ANGLES: dict[str, float] = {
	"r_sho_pitch": 1.0,
	"l_sho_pitch": -1.0,
	"r_sho_roll": -0.9,
	"l_sho_roll": 0.9,
	"r_el": 0.4,
	"l_el": -0.4,
	"r_hip_pitch": 0.57,
	"l_hip_pitch": -0.57,
	"r_knee": -1.5,
	"l_knee": 1.5,
}


@dataclass
class GainConfig:
	kp_scale_stiff: float = 1.0
	kd_scale_stiff: float = 1.0
	kp_scale_compliant: float = 0.45
	kd_scale_compliant: float = 1.8
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

		self.head_body_id = self._resolve_head_body_id()
		self.step_count = 0

		# Initialize IMU sensor tracking
		self.imu_sensor_ids = self._find_imu_sensors()
		self.imu_data_ranges = self._get_imu_data_ranges()
		print(f"Found {len(self.imu_sensor_ids)} IMU sensors: {list(self.imu_sensor_ids.keys())}")

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
		# target = self._target_ctrl + action
		# self.data.ctrl[self.control_actuator_ids] = target

		mujoco.mj_step(self.model, self.data)
		self.step_count += 1

		info = {
			"qfrc_actuator_l1": float(np.sum(np.abs(self.data.qfrc_actuator[self.control_dof_ids]))),
			"qfrc_constraint_l1": float(np.sum(np.abs(self.data.qfrc_constraint))),
			"actuator_force_l1": float(np.sum(np.abs(self.data.actuator_force[self.control_actuator_ids]))),
		}
		info["total_load"] = (
			info["qfrc_actuator_l1"] + info["qfrc_constraint_l1"] + info["actuator_force_l1"]
		)
		
		# Add IMU readings with detailed component information
		imu_readings = self._read_imu_data()
		info.update(imu_readings)

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


def run_experiments() -> None:
	root = Path(__file__).resolve().parent
	scene_xml = root / "robotis_op3" / "scene.xml"
	out_dir = root / "outputs"
	imu_out_dir = out_dir / "IMU" / "experiment_freefall"
	imu_out_dir.mkdir(parents=True, exist_ok=True)
	out_dir.mkdir(parents=True, exist_ok=True)

	num_runs = 1
	max_steps = 50
	base_seed = 1234

	gain_cfg = GainConfig()
	impact_cfg = ImpactConfig()
	gate_cfg = GoalGateConfig()
	restore_cfg = RestoreConfig()
	push_cfg = PushConfig()

	env = Op3FallControlEnv(
		model_xml=scene_xml,
		goal_angles=GOAL_ANGLES,
		render_mode=None,
		control_timestep=0.02,
		camera_distance=1.5,
		camera_azimuth=160.0,
		camera_elevation=-20.0,
	)

	# Logs by cohort.
	stiff_total: list[np.ndarray] = []
	compliant_total: list[np.ndarray] = []
	stiff_m1: list[np.ndarray] = []
	compliant_m1: list[np.ndarray] = []
	stiff_m2: list[np.ndarray] = []
	compliant_m2: list[np.ndarray] = []
	stiff_m3: list[np.ndarray] = []
	compliant_m3: list[np.ndarray] = []

	try:
		for run_idx in range(num_runs):
			compliant_run = (run_idx % 2) == 1
			env.reset(seed=base_seed + run_idx)
			env.set_goal_angles(GOAL_ANGLES)
			env.apply_gain_scales(gain_cfg.kp_scale_stiff, gain_cfg.kd_scale_stiff, gain_cfg)

			push_steps = int(np.ceil(push_cfg.duration_sec / env.model.opt.timestep))

			m1_list: list[float] = []
			m2_list: list[float] = []
			m3_list: list[float] = []
			total_list: list[float] = []
			spike_flags: list[bool] = []
			
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

			for step in range(max_steps):
				# Initial push at head.
				env.data.xfrc_applied[:, :] = 0.0
				if step < push_steps:
					env.data.xfrc_applied[env.head_body_id, :3] = np.array(push_cfg.force_xyz)

				obs, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float64))
				del obs
				env.render()

				m1 = info["qfrc_actuator_l1"]
				m2 = info["qfrc_constraint_l1"]
				m3 = info["actuator_force_l1"]
				total = info["total_load"]

				m1_list.append(m1)
				m2_list.append(m2)
				m3_list.append(m3)
				total_list.append(total)

				# Keep original fall-speed definition from angular velocity.
				wx = float(info.get("torso_angvel_x", 0.0))
				wy = float(info.get("torso_angvel_y", 0.0))
				fall_speed = float(np.sqrt(wx**2 + wy**2))

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

				if compliant_run and (not switched_to_compliance) and (goal_reached_step is not None):
					env.apply_gain_scales(gain_cfg.kp_scale_compliant, gain_cfg.kd_scale_compliant, gain_cfg)
					switched_to_compliance = True
					compliance_step = step

				# Impact spike detection on total load.
				is_spike = False
				if len(rolling) >= impact_cfg.rolling_window:
					mu = float(np.mean(rolling))
					sigma = float(np.std(rolling) + 1e-9)
					jump = float(total / max(last_total, 1e-9))
					if refractory == 0 and total > (mu + impact_cfg.sigma_multiplier * sigma) and jump > impact_cfg.jump_ratio:
						is_spike = True
						refractory = impact_cfg.refractory_steps
						last_spike_step = step
						# If we get a spike before switching to compliance, we can trigger early compliance.
						if compliant_run and not switched_to_compliance:
							env.apply_gain_scales(gain_cfg.kp_scale_compliant, gain_cfg.kd_scale_compliant, gain_cfg)
							switched_to_compliance = True
							compliance_step = step
       
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

				# Restoration condition after loads subside (for compliant runs only).
				if switched_to_compliance and not restore_started:
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

			# Per-run plots.
			t = np.arange(max_steps)
			fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
			ax[0].plot(t, m1_list, label="qfrc_actuator (L1)")
			ax[1].plot(t, m2_list, label="qfrc_constraint (L1)", color="tab:orange")
			ax[2].plot(t, m3_list, label="actuator_force (L1)", color="tab:green")
			ax[3].plot(t, total_list, label="total_load", color="tab:red")

			for i in range(4):
				ax[i].grid(True, alpha=0.3)
				if goal_reached_step is not None:
					ax[i].axvline(goal_reached_step, color="k", linestyle="--", alpha=0.4, label="goal reached")
				if compliance_step is not None:
					ax[i].axvline(compliance_step, color="purple", linestyle=":", alpha=0.6, label="compliance ON")
				if restore_start_step is not None:
					ax[i].axvline(restore_start_step, color="brown", linestyle="-.", alpha=0.6, label="restore gains")
				ax[i].legend(loc="upper right")

			ax[-1].set_xlabel("Timestep")
			ax[0].set_title(f"Run {run_idx}: {'compliant-after-goal' if compliant_run else 'stiff'}")
			fig.tight_layout()
			fig.savefig(out_dir / f"run_{run_idx:02d}_loads.png", dpi=150)
			plt.close(fig)

			# Plot fall dynamics (fall speed and fall angle)
			t = np.arange(max_steps)
			fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

			# Fall speed
			axs[0].plot(t, fall_speed_list, label="fall_speed", color="tab:red", linewidth=2)
			axs[0].set_title(f"Run {run_idx}: Fall Speed = sqrt(wx² + wy²)")
			axs[0].set_ylabel("Fall Speed (rad/s)")
			axs[0].grid(True, alpha=0.3)
			axs[0].legend(loc="upper right")

			# Fall angle
			axs[1].plot(t, fall_angle_list, label="fall_angle", color="tab:blue", linewidth=2)
			axs[1].set_title("Fall Angle (robust): filtered tilt/velocity, locked at impact")
			axs[1].set_ylabel("Fall Angle (radians)")
			axs[1].set_xlabel("Timestep")
			axs[1].grid(True, alpha=0.3)
			axs[1].legend(loc="upper right")

			fig.tight_layout()
			fig.savefig(imu_out_dir / f"run_{run_idx:02d}_fall_dynamics.png", dpi=150)
			plt.close(fig)

			m1_arr = np.array(m1_list)
			m2_arr = np.array(m2_list)
			m3_arr = np.array(m3_list)
			total_arr = np.array(total_list)
			if compliant_run:
				compliant_m1.append(m1_arr)
				compliant_m2.append(m2_arr)
				compliant_m3.append(m3_arr)
				compliant_total.append(total_arr)
			else:
				stiff_m1.append(m1_arr)
				stiff_m2.append(m2_arr)
				stiff_m3.append(m3_arr)
				stiff_total.append(total_arr)

		# Envelope plots (stiff vs compliant) for all requested metrics + total.
		if num_runs > 1 and stiff_total and compliant_total:
			metric_pairs = [
				("qfrc_actuator", stiff_m1, compliant_m1),
				("qfrc_constraint", stiff_m2, compliant_m2),
				("actuator_force", stiff_m3, compliant_m3),
				("total_load", stiff_total, compliant_total),
			]

			fig, axs = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
			x = np.arange(max_steps)
			for row, (name, stiff_curves, comp_curves) in enumerate(metric_pairs):
				stiff_env = compute_envelope(stiff_curves)
				comp_env = compute_envelope(comp_curves)

				axs[row, 0].plot(x, stiff_env["median"], color="tab:blue", label="median")
				axs[row, 0].fill_between(x, stiff_env["p10"], stiff_env["p90"], color="tab:blue", alpha=0.2, label="p10-p90")
				axs[row, 0].set_title(f"Stiff: {name}")
				axs[row, 0].grid(True, alpha=0.3)
				axs[row, 0].legend(loc="upper right")

				axs[row, 1].plot(x, comp_env["median"], color="tab:purple", label="median")
				axs[row, 1].fill_between(x, comp_env["p10"], comp_env["p90"], color="tab:purple", alpha=0.2, label="p10-p90")
				axs[row, 1].set_title(f"Compliant-after-goal: {name}")
				axs[row, 1].grid(True, alpha=0.3)
				axs[row, 1].legend(loc="upper right")

				# Keep y-scale aligned per metric row for easier comparison.
				y_min = min(float(np.min(stiff_env["p10"])), float(np.min(comp_env["p10"])))
				y_max = max(float(np.max(stiff_env["p90"])), float(np.max(comp_env["p90"])))
				axs[row, 0].set_ylim(y_min, y_max)
				axs[row, 1].set_ylim(y_min, y_max)

			axs[-1, 0].set_xlabel("Timestep")
			axs[-1, 1].set_xlabel("Timestep")
			fig.tight_layout()
			fig.savefig(out_dir / "envelope_stiff_vs_compliant.png", dpi=150)
			plt.close(fig)
		else:
			print("Skipping envelope plot: need at least 2 runs with both stiff and compliant cohorts.")

		# Lightweight summary to terminal.
		stiff_peaks = [float(np.max(c)) for c in stiff_total]
		comp_peaks = [float(np.max(c)) for c in compliant_total]
		stiff_auc = [float(np.trapezoid(c)) for c in stiff_total]
		comp_auc = [float(np.trapezoid(c)) for c in compliant_total]

		stiff_peak_median = float(np.median(stiff_peaks)) if stiff_peaks else None
		comp_peak_median = float(np.median(comp_peaks)) if comp_peaks else None
		stiff_auc_median = float(np.median(stiff_auc)) if stiff_auc else None
		comp_auc_median = float(np.median(comp_auc)) if comp_auc else None

		peak_summary = (
			f"{stiff_peak_median:.6f} / {comp_peak_median:.6f}"
			if (stiff_peak_median is not None and comp_peak_median is not None)
			else "N/A (need both stiff and compliant runs)"
		)
		auc_summary = (
			f"{stiff_auc_median:.6f} / {comp_auc_median:.6f}"
			if (stiff_auc_median is not None and comp_auc_median is not None)
			else "N/A (need both stiff and compliant runs)"
		)

		print("Saved plots to:", out_dir)
		print("Median peak total_load (stiff/compliant):", peak_summary)
		print("Median AUC total_load (stiff/compliant):", auc_summary)

	finally:
		env.close()


if __name__ == "__main__":
	run_experiments()