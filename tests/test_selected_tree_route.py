"""Partial-row service from an in-field start to an in-field end."""

from pathlib import Path

import pytest

from examples.plan_short_route_4 import FARM, make_task
from route_planning import PlannerConfig, load_farm_map, plan_route
from route_planning.diagnostic import load_route_json


@pytest.mark.skipif(not (FARM / "gis/farm.gpkg").exists(), reason="showcase farm not generated")
def test_eastern_half_row_turn():
    farm = load_farm_map(FARM)
    task = make_task(farm)
    route = plan_route(farm, task, PlannerConfig(
        tractor_config_path="configs/tractor_route_2m.yaml", solver_time_s=30))
    assert route.optimality_status == "OPTIMAL"
    assert route.task.start == task.start and route.task.end == task.end
    assert [segment.kind for segment in route.segments] == ["rs", "row", "rs", "row", "rs"]
    assert all(segment.reverse_distance_m == 0 for segment in route.segments)
    assert {service for segment in route.segments for service in segment.service_ids} == {
        f"{tree_id}:{side}" for tree_id in task.tree_ids for side in ("L", "R")}
    turn = route.segments[2]
    assert max(pose.x for pose in turn.poses) > 47.0
    assert min(pose.x for pose in turn.poses) > 45.0


def test_saved_partial_row_task_round_trips():
    path = Path("debug_out/routes/short/short_route_4.json")
    if not path.exists():
        pytest.skip("short_route_4 has not been generated")
    route = load_route_json(path)
    assert route.task.tree_ids == tuple(f"tree_{number:05d}" for number in range(380, 385))
