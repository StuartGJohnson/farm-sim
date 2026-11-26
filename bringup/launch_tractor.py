"""Compose a farm and tractor in Isaac Sim, with optional ROS 2 cameras."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaacsim import SimulationApp


def parse_args(default_control_source="teleop") -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world", default="debug_out/farms/farm_seed_1_120m2/world/farm.usda"
    )
    parser.add_argument("--tractor", default="debug_out/tractor/tractor.usda")
    parser.add_argument("--config", default="configs/tractor_default.yaml")
    parser.add_argument("--sensor-config", default="configs/sim_sensors.yaml")
    parser.add_argument("--farm-config", default="debug_out/farms/farm_seed_1_120m2/farm.yaml")
    parser.add_argument("--position", nargs=3, type=float, default=(60.0, 60.0, 2.0))
    parser.add_argument("--yaw-deg", type=float, default=0.0,
                        help="Initial tractor heading in world XY, degrees counterclockwise")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--ros2", action="store_true")
    parser.add_argument("--no-ros-cameras", action="store_true",
                        help="keep ROS odometry and clock but omit camera image topics")
    parser.add_argument("--control-source", choices=("teleop", "ros", "none"),
                        default=default_control_source)
    parser.add_argument("--actuation-timeout", type=float, default=0.5)
    parser.add_argument("--max-updates", type=int, default=None, help="stop after this many app updates")
    parser.add_argument("--movie-output", type=Path,
                        help="MP4 output stem; selected cameras get a suffix before .mp4")
    parser.add_argument("--movie-cameras", default="follow",
                        help="comma-separated views: follow and enabled tractor cameras")
    parser.add_argument("--movie-fps", type=float, default=24.0,
                        help="movie frames per simulated second (default: 24)")
    parser.add_argument("--movie-start-on-motion", action="store_true",
                        help="start recording once ROS odometry reports motion above 0.05 m/s")
    parser.add_argument("--movie-resolution", nargs=2, type=int, default=(1280, 720),
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--follow-distance", type=float, default=6.0,
                        help="camera distance behind rear axle in metres")
    parser.add_argument("--follow-height", type=float, default=3.5,
                        help="camera height above rear axle in metres")
    parser.add_argument("--follow-lead", type=float, default=3.0,
                        help="aim point distance ahead of rear axle in metres")
    parser.add_argument(
        "--frame-report",
        action="store_true",
        help="write managed wheel axes after the simulation has initialized",
    )
    return parser.parse_args()


def main(default_control_source="teleop") -> None:
    args = parse_args(default_control_source)
    if args.control_source == "ros" and not args.ros2:
        raise ValueError("ROS control requires --ros2 for odometry and bridge")
    if args.movie_output is not None:
        import math

        if args.movie_output.suffix.lower() != ".mp4":
            raise ValueError("--movie-output must end in .mp4")
        if not math.isfinite(args.movie_fps) or args.movie_fps <= 0:
            raise ValueError("--movie-fps must be positive and finite")
        if any(size <= 0 for size in args.movie_resolution):
            raise ValueError("--movie-resolution requires positive width and height")
    if args.movie_start_on_motion and (args.movie_output is None or args.control_source != "ros"):
        raise ValueError("--movie-start-on-motion requires a movie and ROS control")
    if any(value <= 0 for value in (args.follow_distance, args.follow_height, args.follow_lead)):
        raise ValueError("follow camera distances must be positive")
    app = SimulationApp({"headless": args.headless})
    teleop = None
    actuation = None
    state_streams = None
    movie_recorder = None
    movie_started = False
    shutdown_requested = False
    previous_sigint_handler = None
    try:
        import omni.timeline
        import omni.usd
        from pxr import Gf, UsdGeom

        from tractor_generation.config import load_tractor_config

        context = omni.usd.get_context()
        context.open_stage(str(Path(args.world).resolve()))
        app.update()
        stage = context.get_stage()
        tractor = stage.DefinePrim("/World/Tractor", "Xform")
        tractor.GetReferences().AddReference(str(Path(args.tractor).resolve()))
        tractor_config = load_tractor_config(args.config)
        UsdGeom.Xformable(tractor).AddTranslateOp().Set(Gf.Vec3d(*args.position))
        UsdGeom.Xformable(tractor).AddRotateZOp().Set(args.yaw_deg)

        # The tractor asset contains a scene for convenient manual inspection;
        # the composed farm's /World/PhysicsScene remains authoritative.
        nested_scene = stage.GetPrimAtPath("/World/Tractor/PhysicsScene")
        if nested_scene:
            nested_scene.SetActive(False)

        from bringup.surface_friction import configure_vehicle_surface_friction

        tire_count = configure_vehicle_surface_friction(stage)
        if tire_count:
            print(f"Connected {tire_count} tires to /World/TireFrictionTable")

        if args.ros2:
            from isaacsim.core.utils.extensions import enable_extension

            enable_extension("isaacsim.ros2.bridge")
            app.update()
            from bringup.action_graphs import create_ros2_camera_graphs, create_ros2_clock_graph

            if not args.no_ros_cameras:
                create_ros2_camera_graphs(tractor_config)
            create_ros2_clock_graph()
            from bringup.sim_sensors import TractorStateStreams, load_sensor_config

            sensor_config, reference = load_sensor_config(args.sensor_config, args.farm_config)
            state_streams = TractorStateStreams(stage, sensor_config, reference)

        vehicle = stage.GetPrimAtPath("/World/Tractor/base_link")
        steer_attr = vehicle.GetAttribute("physxVehicleAckermannSteering:maxSteerAngle")
        asset_steer_angle = steer_attr.Get() if steer_attr else None
        print(f"Steering limit: USD={asset_steer_angle}, config={tractor_config.max_steer_angle_rad:.6f}")
        if (asset_steer_angle is None or
                abs(asset_steer_angle - tractor_config.max_steer_angle_rad) > 0.01):
            raise ValueError(
                f"Tractor USD steering limit {asset_steer_angle} does not match "
                f"{args.config} ({tractor_config.max_steer_angle_rad:.4f} rad); "
                "use an asset generated from the selected tractor config")
        from bringup.movie_recording import FOLLOW_CAMERA_PATH, FollowCamera, MovieRecorder, set_default_view

        follow_camera = FollowCamera(stage, vehicle, args.follow_distance,
                                     args.follow_height, args.follow_lead,
                                     tractor_config.rear_axle_z)
        if not args.headless:
            app.update()
            set_default_view()
        if args.movie_output is not None:
            from tractor_generation.names import camera_link

            names = tuple(name.strip() for name in args.movie_cameras.split(","))
            if len(set(names)) != len(names) or any(not name for name in names):
                raise ValueError("--movie-cameras must list distinct, nonempty views")
            available = {"follow": FOLLOW_CAMERA_PATH}
            available.update({
                name: f"/World/Tractor/base_link/{camera_link(name)}/camera"
                for name in tractor_config.enabled_cameras
            })
            missing = sorted(set(names) - available.keys())
            if missing:
                raise ValueError(f"movie cameras unavailable in tractor asset: {missing}")
            movie_recorder = MovieRecorder(
                stage, {name: available[name] for name in names}, args.movie_output,
                args.movie_fps, tuple(args.movie_resolution))
            from isaacsim.core.simulation_manager import SimulationManager

        if args.control_source == "teleop" and not args.headless:
            from bringup.teleop import TractorKeyboardTeleop

            teleop = TractorKeyboardTeleop(vehicle)
            print("W/S throttle and reverse, A/D steer, Space brake")
        elif args.control_source == "ros":
            from bringup.vehicle_actuation import VehicleActuationAdapter

            actuation = VehicleActuationAdapter(vehicle, timeout=args.actuation_timeout)
            print("ROS actuator control on /vehicle/actuation")

        # Generated assets have no animation range. Give this interactive
        # simulation a usable range so Kit does not loop a single frame.
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(stage.GetTimeCodesPerSecond() * 86400)
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_looping(False)
        timeline.play()
        # Kit's SIGINT handler exits before Python's finally block, which would
        # discard recorded frames. Let the loop stop and encode them normally.
        def request_shutdown(_signum, _frame):
            nonlocal shutdown_requested
            shutdown_requested = True

        previous_sigint_handler = signal.signal(signal.SIGINT, request_shutdown)
        update_count = 0
        while app.is_running() and not shutdown_requested:
            if teleop is not None:
                teleop.update()
            if actuation is not None:
                actuation.update()
            follow_camera.update()
            app.update()
            if movie_recorder is not None:
                movie_started = movie_started or not args.movie_start_on_motion or (
                    actuation is not None and actuation.speed is not None
                    and abs(actuation.speed) > 0.05)
                if movie_started:
                    movie_recorder.capture_if_due(SimulationManager.get_simulation_time())
            update_count += 1
            if args.max_updates is not None and update_count >= args.max_updates:
                break
            if args.frame_report and update_count == 30:
                from bringup.runtime_diagnostic import write_wheel_frame_report

                report = write_wheel_frame_report(
                    stage,
                    "/World/Tractor/base_link",
                    "debug_out/tractor/runtime_frame_report.txt",
                )
                print(f"Runtime wheel-frame report written to {report}")
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        if previous_sigint_handler is not None:
            signal.signal(signal.SIGINT, previous_sigint_handler)
        if args.max_updates is not None and state_streams is not None:
            truth_count = state_streams.truth.samples if state_streams.truth else 0
            gnss_count = state_streams.gnss.samples if state_streams.gnss else 0
            print(f"State stream samples: ground_truth={truth_count}, gnss={gnss_count}")
        if teleop is not None:
            teleop.close()
        if actuation is not None:
            actuation.close()
        if state_streams is not None:
            state_streams.close()
        try:
            if movie_recorder is not None:
                manifest = movie_recorder.close()
                print(f"Wrote {manifest['frame_count']} frames per camera: {manifest['movies']}")
        finally:
            app.close()


if __name__ == "__main__":
    main()
