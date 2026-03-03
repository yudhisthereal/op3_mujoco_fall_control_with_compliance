from __future__ import annotations

import time
from pathlib import Path

import mujoco

try:
	import mujoco.viewer as mj_viewer
except Exception as exc:  # pragma: no cover
	raise RuntimeError("mujoco.viewer is required for GUI control.") from exc


def run_static_simulation() -> None:
	root = Path(__file__).resolve().parent
	scene_xml = root / "robotis_op3" / "scene.xml"

	model = mujoco.MjModel.from_xml_path(str(scene_xml))
	data = mujoco.MjData(model)

	# Match camera setup used in main.py/tutorial.
	camera = mujoco.MjvCamera()
	mujoco.mjv_defaultFreeCamera(model, camera)
	camera.distance = 1.5
	camera.azimuth = 160.0
	camera.elevation = -20.0
	camera.lookat[:] = (0.0, 0.0, 0.2)

	mujoco.mj_resetData(model, data)
	mujoco.mj_forward(model, data)

	with mj_viewer.launch_passive(model, data) as viewer:
		viewer.cam.distance = camera.distance
		viewer.cam.azimuth = camera.azimuth
		viewer.cam.elevation = camera.elevation
		viewer.cam.lookat[:] = camera.lookat

		step_dt = float(model.opt.timestep)
		while viewer.is_running():
			step_start = time.perf_counter()
			mujoco.mj_step(model, data)
			viewer.sync()

			remaining = step_dt - (time.perf_counter() - step_start)
			if remaining > 0:
				time.sleep(remaining)


if __name__ == "__main__":
	run_static_simulation()
