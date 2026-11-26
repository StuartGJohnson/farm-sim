"""Plan a short Field 004 replan across the eastern half of row R003."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from route_planning import FarmTask, ImplementSide, PlannerConfig, Pose, load_farm_map, plan_route
from route_planning.diagnostic import save_route_json, save_route_plot


FARM = Path("debug_out/farms/farm_seed_1_120m2_depot_outlands_fence")
OUTPUT = Path("debug_out/routes/short/short_route_4")
FIELD = "parcel_003"


def lane_midpoint(south, north) -> tuple[float, float]:
    """Center of the common east-west driveable span between adjacent rows."""
    west = max(south.line.bounds[0], north.line.bounds[0])
    east = min(south.line.bounds[2], north.line.bounds[2])
    return ((west + east) / 2, (south.line.centroid.y + north.line.centroid.y) / 2)


def make_task(farm) -> FarmTask:
    south = farm.rows[f"{FIELD}_row_004"]
    middle = farm.rows[f"{FIELD}_row_003"]
    north = farm.rows[f"{FIELD}_row_002"]
    start = Pose(*lane_midpoint(south, middle), 0.0)
    end = Pose(*lane_midpoint(middle, north), math.pi)
    east_half_start = (middle.line.bounds[0] + middle.line.bounds[2]) / 2
    trees = tuple(sorted(tree.id for tree in farm.trees.values()
                         if tree.row_id == middle.id and
                         tree.x >= east_half_start))
    return FarmTask((FIELD,), ImplementSide.BOTH, start, end, tree_ids=trees)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turn-radius-m", type=float, default=2.0,
                        help="minimum Reeds-Shepp turning radius (default: 2.0 m)")
    parser.add_argument("--output-stem", type=Path, default=OUTPUT)
    args = parser.parse_args()
    farm = load_farm_map(FARM)
    task = make_task(farm)
    middle = farm.rows[f"{FIELD}_row_003"]
    start, end, trees = task.start, task.end, task.tree_ids
    config = PlannerConfig(tractor_config_path="configs/tractor_route_2m.yaml",
                           turn_radius_m=args.turn_radius_m, solver_time_s=90)
    route = plan_route(farm, task, config)
    save_route_json(route, farm, args.output_stem.with_suffix(".json"))
    save_route_plot(route, farm, args.output_stem.with_suffix(".png"),
                    bounds=(start.x - 7, start.y - 8, middle.line.bounds[2] + 12, end.y + 8))
    manifest = {
        "farm": str(FARM), "field": FIELD, "selected_row": middle.id,
        "selected_tree_ids": trees, "start_pose": vars(start), "end_pose": vars(end),
        "turn_radius_m": config.turn_radius_m,
        "status": route.optimality_status, "distance_m": route.total_distance_m,
        "reverse_distance_m": sum(segment.reverse_distance_m for segment in route.segments),
        "service_ids": sorted(set().union(*(segment.service_ids for segment in route.segments))),
        "solver_wall_time_s": route.solver_wall_time_s,
    }
    args.output_stem.with_name(args.output_stem.name + "_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
