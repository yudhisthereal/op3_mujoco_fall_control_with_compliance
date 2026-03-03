"""Example: orient and position arrows defined in robotis_op3/arrow.xml."""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def normalize_3d(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalize a 3D vector."""
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros(3, dtype=np.float64)
    return v / n


def direction_to_quat(direction: np.ndarray) -> np.ndarray:
    """Quaternion [w, x, y, z] that rotates local +X to the given direction."""
    d = normalize_3d(direction)
    if float(np.linalg.norm(d)) < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    default_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    axis = np.cross(default_axis, d)
    axis_norm = float(np.linalg.norm(axis))

    if axis_norm < 1e-9:
        if float(np.dot(default_axis, d)) > 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)

    axis = axis / axis_norm
    angle = float(np.arccos(np.clip(np.dot(default_axis, d), -1.0, 1.0)))
    half = 0.5 * angle
    return np.array(
        [np.cos(half), axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half)],
        dtype=np.float64,
    )


def main():
    # Load scene.xml (which includes arrow.xml under worldbody)
    root = Path(__file__).resolve().parent
    scene_xml = root / "robotis_op3" / "scene.xml"
    
    if not scene_xml.exists():
        print(f"Error: {scene_xml} not found")
        return
    
    # Load model and data
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    
    viewer = mujoco.viewer.launch_passive(model, data)
    
    try:
        # Resolve arrow freejoint qpos addresses from arrow.xml bodies.
        arrow_qpos_addr: dict[str, int] = {}
        arrow_fixed_pos: dict[str, np.ndarray] = {}
        for name in ["fall", "left", "right"]:
            body_name = f"arrow_{name}_shaft"
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                print(f"Error: body '{body_name}' not found. Ensure arrow.xml is included in scene.xml.")
                return
            joint_id = model.body_jntadr[body_id]
            qpos_addr = model.jnt_qposadr[joint_id]
            arrow_qpos_addr[name] = qpos_addr
            arrow_fixed_pos[name] = data.qpos[qpos_addr:qpos_addr + 3].copy()

        # Example directions to demonstrate arrow.xml-based visualization.
        arrow_dirs = {
            "fall": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "left": np.array([0.0, -1.0, 0.2], dtype=np.float64),
            "right": np.array([0.0, 1.0, 0.2], dtype=np.float64),
        }

        print("Using arrows from arrow.xml:")
        for name, d in arrow_dirs.items():
            print(f"  {name:>5}: dir={normalize_3d(d)}")

        # Apply arrow poses once (static arrows).
        # for name, d in arrow_dirs.items():
        #     addr = arrow_qpos_addr[name]
        #     data.qpos[addr:addr + 3] = arrow_fixed_pos[name]
        #     data.qpos[addr + 3:addr + 7] = direction_to_quat(d)
        # mujoco.mj_forward(model, data)
        
        step_count = 0
        max_steps = 500
        
        while viewer.is_running() and step_count < max_steps:
            # Keep scene static; only refresh viewer.
            viewer.sync()
            step_count += 1
        
        print(f"\nSimulation ran for {step_count} steps")
        
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
