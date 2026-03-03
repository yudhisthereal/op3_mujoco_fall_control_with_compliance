from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np

from main import GOAL_ANGLES_ARMS, GainConfig, Op3FallControlEnv, PushConfig


@dataclass
class AlignConfig:
    angle_threshold_rad: float = float(np.pi / 4.0)  # 45 deg
    alpha_tilt: float = 0.85


def _normalize_3d(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros(3, dtype=np.float64)
    return v / n


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


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])


def _wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _estimate_fall_direction_vector_3d(
    quat_wxyz: np.ndarray,
    linvel_xyz: np.ndarray,
) -> np.ndarray:
    """
    3D fall direction from torso linear velocity only.
    """
    del quat_wxyz  # Not needed in this estimator, kept for interface compatibility.
    return _normalize_3d(linvel_xyz.astype(np.float64, copy=False))


def _torso_forward_axis_3d(quat_wxyz: np.ndarray) -> np.ndarray:
    """Body +X axis expressed in world frame from quaternion (w, x, y, z)."""
    w, x, y, z = quat_wxyz
    fwd = np.array(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ],
        dtype=np.float64,
    )
    return _normalize_3d(fwd)


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
    shoulder_id = ids["shoulder"]
    if shoulder_id is None:
        return np.zeros(3, dtype=np.float64), "none"

    candidates: list[tuple[str, np.ndarray]] = []
    if ids["hand_site"] is not None:
        candidates.append(("hand", env.data.site_xpos[ids["hand_site"]].astype(np.float64)))
    if ids["hand_body"] is not None:
        candidates.append(("hand_body", env.data.xpos[ids["hand_body"]].astype(np.float64)))
    if ids["elbow"] is not None:
        candidates.append(("elbow", env.data.xpos[ids["elbow"]].astype(np.float64)))

    if not candidates:
        return np.zeros(3, dtype=np.float64), "none"

    critical_label, critical_pos = min(candidates, key=lambda item: float(item[1][2]))
    shoulder_pos = env.data.xpos[shoulder_id].astype(np.float64)
    arm_vec = _normalize_3d(critical_pos - shoulder_pos)
    return arm_vec, critical_label


def _alignment_score_from_angles(arm_angle: float, fall_angle: float, threshold_rad: float) -> tuple[float, float]:
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


def _draw_vectors_simple(
    env: Op3FallControlEnv,
    fall_dir: np.ndarray,
    left_arm_dir: np.ndarray,
    right_arm_dir: np.ndarray,
    left_shoulder_id: int | None,
    right_shoulder_id: int | None,
    vec_len: float = 0.22,  # This controls the visual length of the arrow
    cylinder_length_mult: float = 5.0,  # Additional multiplier for cylinder length
) -> None:
    """Draw vectors using cylinder geoms.

    - fall_dir starts at head position
    - left/right arm dirs start at each shoulder position
    """
    viewer = getattr(env, "_viewer", None)
    if viewer is None:
        return

    scn = viewer.user_scn
    scn.ngeom = 0

    rgba_fall = np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32)   # red
    rgba_left = np.array([0.2, 1.0, 0.2, 1.0], dtype=np.float32)   # green
    rgba_right = np.array([0.2, 0.5, 1.0, 1.0], dtype=np.float32)  # blue

    def _add_line(start: np.ndarray, direction: np.ndarray, rgba: np.ndarray, width: float = 0.1) -> None:
        if scn.ngeom >= len(scn.geoms):
            return
        d = _normalize_3d(direction)
        if float(np.linalg.norm(d)) < 1e-9:
            return
        
        # Calculate end point for positioning, but use full cylinder length for visual
        end = start + vec_len * d
        mid = (start + end) / 2.0

        g = scn.geoms[scn.ngeom]
        
        # Make cylinder longer by using full vec_len (remove the /2.0 or adjust multiplier)
        # Option 1: Use full vec_len
        cylinder_length = vec_len * cylinder_length_mult
        size = np.array([width, width, cylinder_length], dtype=np.float64)
        
        # Option 2: If you want even longer, you could do:
        # cylinder_length = vec_len * 2.0  # Twice as long
        # size = np.array([width, width, cylinder_length], dtype=np.float64)
        
        pos = mid.astype(np.float64)
        
        # Create rotation matrix to align cylinder with direction
        rot = np.eye(3, dtype=np.float64)
        if abs(d[2]) < 0.99999:  # Not aligned with Z-axis
            # Get rotation axis (cross product of Z and direction)
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            axis = np.cross(z_axis, d)
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm > 1e-9:
                axis = axis / axis_norm
                angle = np.arccos(np.clip(d[2], -1.0, 1.0))
                
                # Rodrigues rotation formula
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ], dtype=np.float64)
                rot = np.eye(3, dtype=np.float64) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        
        # Flatten rotation matrix to 9-element array as required by mjv_initGeom
        rot_flat = rot.flatten().astype(np.float64)
        
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            size,
            pos,
            rot_flat,
            rgba.astype(np.float32),
        )
        scn.ngeom += 1

    head_origin = env.data.xpos[env.head_body_id].astype(np.float64)
    _add_line(head_origin, fall_dir, rgba_fall)

    if left_shoulder_id is not None:
        left_origin = env.data.xpos[left_shoulder_id].astype(np.float64)
        _add_line(left_origin, left_arm_dir, rgba_left)

    if right_shoulder_id is not None:
        right_origin = env.data.xpos[right_shoulder_id].astype(np.float64)
        _add_line(right_origin, right_arm_dir, rgba_right)

    viewer.sync()

def run_test(steps: int, render_mode: str, csv_path: Path, video_path: Path | None = None) -> None:
    root = Path(__file__).resolve().parent
    scene_xml = root / "robotis_op3" / "scene.xml"

    align_cfg = AlignConfig()
    push_cfg = PushConfig()

    env = Op3FallControlEnv(
        model_xml=scene_xml,
        goal_angles=GOAL_ANGLES_ARMS,
        render_mode=render_mode,
        control_timestep=0.02,
        camera_distance=2.5,
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

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "fall_dir_x",
                "fall_dir_y",
                "fall_dir_z",
                "fall_speed_rad_s",
                "left_arm_dir_x",
                "left_arm_dir_y",
                "left_arm_dir_z",
                "right_arm_dir_x",
                "right_arm_dir_y",
                "right_arm_dir_z",
                "fall_angle_rad",
                "left_arm_angle_rad",
                "right_arm_angle_rad",
                "left_align_angle_rad",
                "right_align_angle_rad",
                "left_align_reward",
                "right_align_reward",
                "left_critical",
                "right_critical",
            ],
        )
        writer.writeheader()

        try:
            env.reset(seed=0)
            env.set_goal_angles(GOAL_ANGLES_ARMS)
            env.apply_gain_scales(1.0, 1.0, GainConfig())
            push_steps = int(np.ceil(push_cfg.duration_sec / env.model.opt.timestep))
            
            # Get arrow shaft body IDs and qpos addresses
            arrow_ids = {}
            arrow_positions = {}
            for name in ["fall", "left", "right"]:
                shaft_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"arrow_{name}_shaft")
                shaft_jnt_id = env.model.body_jntadr[shaft_body_id]
                qpos_addr = env.model.jnt_qposadr[shaft_jnt_id]
                arrow_ids[name] = qpos_addr
                # Store initial position from qpos
                arrow_positions[name] = env.data.qpos[qpos_addr:qpos_addr+3].copy()
                arrow_ids[name] = env.model.jnt_qposadr[shaft_jnt_id]

            zero_action = np.zeros(env.action_space.shape, dtype=np.float64)
            
            # Smoothing + locking for fall direction
            smoothed_fall_dir = np.zeros(3, dtype=np.float64)
            fall_dir_smoothing_alpha = 0.3
            fall_locked = False
            locked_fall_dir: np.ndarray | None = None
            fall_lock_threshold = 1.5  # rad/s
            
            # Video writer setup if recording is enabled
            video_writer = None
            if video_path is not None:
                video_path.parent.mkdir(parents=True, exist_ok=True)
                # Get frame from render to determine dimensions
                frame = env.render()
                if frame is not None and render_mode == "rgb_array":
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(video_path),
                        fourcc,
                        fps=30,  # 30 FPS (0.02s per frame * 50 frames = 1 second at 30 FPS)
                        frameSize=(width, height),
                    )

            for step in range(steps):
                # Initial forward push at head (forward-fall assumption).
                env.data.xfrc_applied[:, :] = 0.0
                if step < push_steps:
                    env.data.xfrc_applied[env.head_body_id, :3] = np.array(push_cfg.force_xyz, dtype=np.float64)

                _, _, _, _, info = env.step(zero_action)
                env.render()

                torso_q, torso_linvel = _try_get_torso_quat_and_linvel(info)
                torso_angvel_x = float(info.get("torso_angvel_x", 0.0))
                torso_angvel_y = float(info.get("torso_angvel_y", 0.0))
                torso_angvel_z = float(info.get("torso_angvel_z", 0.0))
                torso_angvel = np.array([torso_angvel_x, torso_angvel_y, torso_angvel_z], dtype=np.float64)
                fall_speed = float(np.sqrt(torso_angvel_x**2 + torso_angvel_y**2))

                raw_fall_dir = _estimate_fall_direction_vector_3d(
                    torso_q,
                    torso_linvel,
                )

                if not fall_locked:
                    if float(np.linalg.norm(raw_fall_dir)) > 1e-9:
                        new_dir = _normalize_3d(raw_fall_dir)
                        if float(np.linalg.norm(smoothed_fall_dir)) < 1e-9:
                            smoothed_fall_dir = new_dir
                        else:
                            blended = (1 - fall_dir_smoothing_alpha) * smoothed_fall_dir + fall_dir_smoothing_alpha * new_dir
                            smoothed_fall_dir = _normalize_3d(blended)

                    fall_dir = smoothed_fall_dir
                    if fall_speed > fall_lock_threshold and float(np.linalg.norm(fall_dir)) > 1e-9:
                        fall_locked = True
                        locked_fall_dir = fall_dir.copy()
                else:
                    fall_dir = locked_fall_dir if locked_fall_dir is not None else smoothed_fall_dir

                left_arm_dir, left_critical = _get_arm_vector_and_critical(env, "left", left_ids, right_ids)
                right_arm_dir, right_critical = _get_arm_vector_and_critical(env, "right", left_ids, right_ids)

                # Vector azimuths in XY plane (still useful as angle logs)
                fall_angle = float(np.arctan2(fall_dir[1], fall_dir[0])) if float(np.linalg.norm(fall_dir[:2])) > 1e-9 else 0.0
                left_arm_angle_raw = float(np.arctan2(left_arm_dir[1], left_arm_dir[0])) if float(np.linalg.norm(left_arm_dir[:2])) > 1e-9 else 0.0
                right_arm_angle_raw = float(np.arctan2(right_arm_dir[1], right_arm_dir[0])) if float(np.linalg.norm(right_arm_dir[:2])) > 1e-9 else 0.0

                # Use instantaneous arm azimuths directly for alignment.
                left_arm_angle = left_arm_angle_raw
                right_arm_angle = right_arm_angle_raw

                # Recompute alignment using corrected per-arm angles.
                left_rew, left_align_angle = _alignment_score_from_angles(
                    left_arm_angle,
                    fall_angle,
                    align_cfg.angle_threshold_rad,
                )
                right_rew, right_align_angle = _alignment_score_from_angles(
                    right_arm_angle,
                    fall_angle,
                    align_cfg.angle_threshold_rad,
                )

                if render_mode == "human":
                    # Update arrow positions before rendering
                    _draw_vectors_programmatic(
                        env,
                        fall_dir,
                        left_arm_dir,
                        right_arm_dir,
                        arrow_ids,
                        arrow_positions,
                        torso_angvel_x,
                        torso_angvel_y,
                    )
                
                # Render and optionally record video
                if render_mode == "rgb_array" and video_writer is not None:
                    frame = env.render()
                    if frame is not None:
                        # Convert RGB to BGR for OpenCV
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        video_writer.write(frame_bgr)
                else:
                    env.render()

                writer.writerow(
                    {
                        "step": step,
                        "fall_speed_rad_s": fall_speed,
                        "fall_dir_x": float(fall_dir[0]),
                        "fall_dir_y": float(fall_dir[1]),
                        "fall_dir_z": float(fall_dir[2]),
                        "left_arm_dir_x": float(left_arm_dir[0]),
                        "left_arm_dir_y": float(left_arm_dir[1]),
                        "left_arm_dir_z": float(left_arm_dir[2]),
                        "right_arm_dir_x": float(right_arm_dir[0]),
                        "right_arm_dir_y": float(right_arm_dir[1]),
                        "right_arm_dir_z": float(right_arm_dir[2]),
                        "fall_angle_rad": fall_angle,
                        "left_arm_angle_rad": left_arm_angle,
                        "right_arm_angle_rad": right_arm_angle,
                        "left_align_angle_rad": left_align_angle,
                        "right_align_angle_rad": right_align_angle,
                        "left_align_reward": left_rew,
                        "right_align_reward": right_rew,
                        "left_critical": left_critical,
                        "right_critical": right_critical,
                    }
                )

                if step % 25 == 0 or step == steps - 1:
                    print(
                        f"step={step:03d} "
                        f"F_dir=({fall_dir[0]:+.3f},{fall_dir[1]:+.3f},{fall_dir[2]:+.3f}) "
                        f"L_dir=({left_arm_dir[0]:+.3f},{left_arm_dir[1]:+.3f},{left_arm_dir[2]:+.3f}) "
                        f"R_dir=({right_arm_dir[0]:+.3f},{right_arm_dir[1]:+.3f},{right_arm_dir[2]:+.3f}) | "
                        f"fall={fall_angle:+.3f} rad | "
                        f"L_arm={left_arm_angle:+.3f} rad L_align={left_align_angle:.3f} rew={left_rew:.3f} ({left_critical}) | "
                        f"R_arm={right_arm_angle:+.3f} rad R_align={right_align_angle:.3f} rew={right_rew:.3f} ({right_critical})"
                    )

        finally:
            env.close()
            if video_writer is not None:
                video_writer.release()
        
        # Clean up temporary file if created
        tmp_path_local = locals().get("tmp_path")
        if tmp_path_local is not None:
            Path(tmp_path_local).unlink()

    print(f"Saved alignment logs to: {csv_path}")
    if video_path is not None:
        print(f"Saved video to: {video_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test and log per-hand alignment reward while moving to GOAL_ANGLES_ARMS")
    parser.add_argument("--steps", type=int, default=250, help="Number of simulation steps")
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["human", "rgb_array"],
        help="Rendering mode",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="Optional CSV output path",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="",
        help="Optional video output path (requires render_mode=rgb_array)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = root / csv_path
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = root / "outputs" / "alignment" / f"align_log_{ts}.csv"

    video_path = None
    if args.video:
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = root / video_path
        if args.render_mode == "human":
            print("Warning: video recording requires --render-mode=rgb_array, ignoring --video")
            video_path = None

    run_test(steps=args.steps, render_mode=args.render_mode, csv_path=csv_path, video_path=video_path)


if __name__ == "__main__":
    main()
