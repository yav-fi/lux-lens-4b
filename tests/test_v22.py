from lux_frontier.v2.exploration import check_lane, group_stats, rotation_metrics


def group(task, offset, choices):
    order = list(range(1, 9))
    order = order[offset:]+order[:offset]
    valid = {i for i in choices if i is not None}
    return dict(task_id=task, position_offset=offset, candidate_order=order, representation="summary",
                sampling="separate", temperature=.6, prompt_tokens=1000,
                baselines=dict(random_pass=.25, closest_pass=True, physics_pass=True, oracle_pass=True, random_reward=.3),
                completions=[{"grade": dict(starter_id=i, error="invalid_submission" if i is None else None,
                                            final_pass=i in (1,2), reward=.9 if i in (1,2) else .01 if i else 0.)}
                             for i in choices],
                diagnostics=dict(optically_informative=len(valid)>1, unique_valid_actions=len(valid),
                                 valid_action_entropy_bits=1., position_counts={}, action_counts={}))


def test_smoke_gate_requires_optical_variation_on_each_feasible_task():
    mixed = [1, 3]*4
    groups = [group(task, offset, mixed) for task in ("a","b") for offset in (0,4)]
    assert check_lane(groups, {"a","b"})["smoke_gate"]
    groups[-2:] = [group("b", offset, [1]*7+[None]) for offset in (0,4)]
    assert group_stats(groups[-1])["valid_reward_variance"] == 0
    assert not check_lane(groups, {"a","b"})["smoke_gate"]


def test_position_rotation_separates_id_stability_from_location_stability():
    first = group("a", 0, [1,2,3,4,5,6,7,None])
    second = group("a", 4, [1,2,3,4,1,2,3,None])
    result = rotation_metrics([first,second])
    assert (result["valid_pairs"],result["same_id"],result["same_position"]) == (7,4,3)
