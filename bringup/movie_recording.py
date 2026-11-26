"""Follow-camera view and synchronized Isaac Sim RGB movie capture."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path


FOLLOW_CAMERA_PATH = "/World/FollowCamera"


class FollowCamera:
    """Track the ROS base_footprint pose: rear-axle XY, world Z=0, yaw only."""

    def __init__(self, stage, vehicle_prim, distance=6.0, height=3.5, lead=3.0,
                 base_link_height=0.8):
        from pxr import Gf, Usd, UsdGeom

        self.vehicle = vehicle_prim
        self.distance = distance
        self.height = height
        self.lead = lead
        self.base_link_height = base_link_height
        self.Gf = Gf
        self.Usd = Usd
        self.UsdGeom = UsdGeom
        camera = UsdGeom.Camera.Define(stage, FOLLOW_CAMERA_PATH)
        self.transform = camera.AddTransformOp()
        aperture = 20.955
        camera.CreateHorizontalApertureAttr(aperture)
        camera.CreateFocalLengthAttr(aperture / (2 * math.tan(math.radians(75) / 2)))
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000.0))
        self.update()

    def update(self) -> None:
        matrix = self.UsdGeom.Xformable(self.vehicle).ComputeLocalToWorldTransform(
            self.Usd.TimeCode.Default())
        Gf = self.Gf
        position = matrix.ExtractTranslation()
        forward = matrix.TransformDir(Gf.Vec3d(1, 0, 0))
        planar = Gf.Vec3d(forward[0], forward[1], 0).GetNormalized()
        # Match ground_truth_localization's level odom -> base_footprint TF.
        # Keep the previous nominal view height, independent of axle bounce.
        footprint = Gf.Vec3d(position[0], position[1], 0)
        eye = (footprint - planar * self.distance +
               Gf.Vec3d(0, 0, self.base_link_height + self.height))
        target = (footprint + planar * self.lead +
                  Gf.Vec3d(0, 0, self.base_link_height + 0.8))
        self.transform.Set(Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse())


def set_default_view(camera_path: str = FOLLOW_CAMERA_PATH) -> None:
    """Select the follow camera in the visible Kit viewport, if any."""
    from omni.kit.viewport.utility import get_active_viewport

    viewport = get_active_viewport()
    if viewport is not None:
        viewport.camera_path = camera_path


class MovieRecorder:
    """Sample multiple render products on one simulation-time schedule."""

    def __init__(self, stage, camera_paths: dict[str, str], output: Path,
                 fps: float, resolution: tuple[int, int]):
        import omni.replicator.core as rep
        from isaacsim.core.utils.extensions import enable_extension

        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("movie fps must be positive and finite")
        if not camera_paths:
            raise ValueError("at least one movie camera is required")
        if any(not stage.GetPrimAtPath(path).IsValid() for path in camera_paths.values()):
            raise ValueError(f"movie camera is missing: {camera_paths}")
        self.output = output.resolve()
        if self.output.suffix.lower() != ".mp4":
            raise ValueError("movie output must end in .mp4")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        enable_extension("omni.videoencoding")
        self.fps = fps
        self.resolution = resolution
        self.camera_paths = camera_paths
        self.products = {}
        self.annotators = {}
        self.next_time = None
        self.frame_times = []
        self.temp = tempfile.TemporaryDirectory(prefix="farm_movie_")
        for name, camera_path in camera_paths.items():
            product = rep.create.render_product(camera_path, resolution)
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach([product])
            self.products[name] = product
            self.annotators[name] = annotator

    def capture_if_due(self, sim_time: float) -> None:
        import numpy as np
        from PIL import Image

        if self.next_time is None:
            self.next_time = sim_time
        if sim_time + 1e-9 < self.next_time:
            return
        frames = {}
        for name, annotator in self.annotators.items():
            pixels = np.asarray(annotator.get_data())
            if pixels.shape[:2] != (self.resolution[1], self.resolution[0]):
                return  # Wait for every render product to produce a valid frame.
            if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
                return
            frames[name] = np.array(pixels, copy=True)
        frame_number = len(self.frame_times)
        for name, pixels in frames.items():
            folder = Path(self.temp.name) / name
            folder.mkdir(exist_ok=True)
            Image.fromarray(pixels).save(folder / f"frame_{frame_number:06d}.png")
        self.frame_times.append(sim_time)
        self.next_time += 1.0 / self.fps
        while self.next_time <= sim_time:
            self.next_time += 1.0 / self.fps

    def close(self) -> dict:
        for name, annotator in self.annotators.items():
            annotator.detach([self.products[name]])
            self.products[name].destroy()
        self.annotators.clear()
        self.products.clear()
        if not self.frame_times:
            self.temp.cleanup()
            raise RuntimeError("no RGB movie frames were captured")
        import video_encoding

        movies = {}
        try:
            for name in self.camera_paths:
                output = self.output.with_name(f"{self.output.stem}_{name}.mp4")
                pattern = str(Path(self.temp.name) / name / "frame_%06d.png")
                if not video_encoding.encode_image_file_sequence(
                    pattern, 0, self.fps, str(output), True
                ):
                    raise RuntimeError(f"failed to encode {name} movie")
                movies[name] = str(output)
            manifest = {
                "movies": movies, "camera_paths": self.camera_paths,
                "fps": self.fps, "resolution": list(self.resolution),
                "frame_count": len(self.frame_times),
                "first_sim_time_s": self.frame_times[0],
                "last_sim_time_s": self.frame_times[-1],
                "frame_sim_times_s": self.frame_times,
            }
            self.output.with_name(f"{self.output.stem}_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n")
            return manifest
        finally:
            self.temp.cleanup()
