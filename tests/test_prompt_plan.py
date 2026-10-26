"""Checks for consequential mistakes in reviewed VPP2 prompt generation."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prompt_plan import (  # noqa: E402
    VIEW_SUFFIX, enrich_grounding, parse_json, render_caption,
    required_rgb_objects, resolve_assignment_modes, validate_plan,
)


def object_at(object_id="bottle", x=200, confidence="high"):
    return {"id": object_id, "label": object_id, "center": [x, 500], "confidence": confidence}


def grounding(*objects):
    return {"objects": list(objects or [object_at()]), "uncertainties": []}


def action(kind="lift", object_id="bottle", **kwargs):
    return {
        "kind": kind, "object_id": object_id, "grasp_part": None,
        "tool_id": None, "destination_id": None, "other_arm": None, **kwargs,
    }


def arm(active=True, mode="proximity", object_id="bottle", phrase="lifts the bottle", evidence="", actions=None):
    return {
        "active": active,
        "assignment": {"mode": mode, "object_id": object_id, "evidence": evidence},
        "actions": [action()] if actions is None else actions,
        "verb_phrase": phrase,
    }


def static_arm():
    return arm(False, "inactive", None, "keeps static", actions=[])


def plan():
    return {
        "requirements": {
            "grasp_parts": [], "destinations": [], "final_states": [],
            "timing": {"kind": "unspecified", "evidence": ""},
        },
        "arms": {"right": arm(), "left": static_arm()},
        "uncertainties": [],
    }


def test_observer_arm_mapping_and_uncertain_coordinates():
    source = grounding(object_at("a", 474), object_at("b", 526), object_at("c", 475), object_at("d", 525, "low"))
    enriched, errors = enrich_grounding(source)
    assert len(errors) == 1 and "low-confidence" in errors[0]
    assert [o["preferred_arm"] for o in enriched["objects"]] == ["right", "left", None, None]
    assert "preferred_arm" not in source["objects"][0]
    invalid = grounding(object_at("a"), object_at("a", 1001))
    _, errors = enrich_grounding(invalid)
    assert any("duplicate id" in error for error in errors)
    assert any("coordinates" in error for error in errors)


def test_valid_plan_renders_fixed_actual_arm_order():
    candidate = plan()
    assert validate_plan(candidate, grounding(), "Lift the bottle.") == []
    assert render_caption(candidate) == (
        "The right robot gripper lifts the bottle, while the left robot gripper keeps static. " + VIEW_SUFFIX
    )
    candidate["arms"]["right"]["verb_phrase"] = "lifts the bottle.@@bad.png"
    with pytest.raises(ValueError, match="@@"):
        render_caption(candidate)


def test_wrong_arm_and_low_confidence_proximity_are_reviewed():
    candidate = plan()
    candidate["arms"]["left"] = candidate["arms"].pop("right")
    candidate["arms"]["right"] = static_arm()
    errors = validate_plan(candidate, grounding(), "Lift the bottle.")
    assert any("proximity heuristic conflicts" in error for error in errors)
    errors = validate_plan(plan(), grounding(object_at(x=500)), "Lift the bottle.")
    assert any("proximity heuristic is unresolved" in error for error in errors)
    errors = validate_plan(plan(), grounding(object_at(confidence="low")), "Lift the bottle.")
    assert any("low-confidence" in error for error in errors)
    task = "Use the right arm to lift the bottle."
    candidate = plan()
    candidate["arms"]["right"]["assignment"].update(mode="explicit", evidence=task)
    assert validate_plan(candidate, grounding(object_at(x=500)), task) == []
    errors = validate_plan(candidate, grounding(object_at(x=500, confidence="medium")), task)
    assert len(errors) == 1 and "medium-confidence" in errors[0]


def test_explicit_arm_prevents_unsupported_extra_participation():
    task = "Use the right arm to lift the bottle."
    candidate = plan()
    candidate["arms"]["right"]["assignment"].update(mode="explicit", evidence=task)
    candidate["arms"]["left"] = arm(mode="proximity", object_id="block", phrase="holds the block", actions=[action("hold", "block")])
    errors = validate_plan(candidate, grounding(object_at(), object_at("block", 800)), task)
    assert any("extra participation" in error for error in errors)
    candidate["arms"]["left"] = static_arm()
    candidate["arms"]["left"]["actions"] = [action("hold")]
    assert any("inactive arm must have no actions" in error for error in validate_plan(candidate, grounding(), task))


def test_handover_preserves_head_and_requires_active_receiver():
    task = "Reach for the mesh-patterned microphone head and move it to the other hand."
    candidate = plan()
    candidate["requirements"]["grasp_parts"] = [{"text": "head", "evidence": "mesh-patterned microphone head"}]
    candidate["arms"]["right"] = arm(
        object_id="microphone", phrase="grasps the microphone by its head and transfers it to the other hand",
        actions=[action("grasp", "microphone", grasp_part="head"), action("transfer", "microphone", other_arm="left")],
    )
    candidate["arms"]["left"] = arm(
        mode="receiver", object_id="microphone", phrase="then receives the microphone from the right robot gripper",
        actions=[action("receive", "microphone", other_arm="right")],
    )
    scene = grounding(object_at("microphone"))
    assert validate_plan(candidate, scene, task) == []
    broken = copy.deepcopy(candidate)
    broken["arms"]["left"] = static_arm()
    assert any("lacks a matching" in error for error in validate_plan(broken, scene, task))
    broken = copy.deepcopy(candidate)
    broken["requirements"]["grasp_parts"] = []
    assert any("task contact part omitted" in error for error in validate_plan(broken, scene, task))


def test_grip_handle_synonym_and_head_orientation_are_not_confused():
    task = "Use the right arm to grasp the hammer by its grip and point the head downward."
    candidate = plan()
    candidate["requirements"]["grasp_parts"] = [{"text": "grip", "evidence": "by its grip"}]
    candidate["requirements"]["final_states"] = [{"text": "head downward", "evidence": "head downward"}]
    candidate["arms"]["right"] = arm(
        mode="explicit", evidence=task, object_id="hammer",
        phrase="grasps the hammer by its handle and points the head downward",
        actions=[action("grasp", "hammer", grasp_part="handle"), action("orient", "hammer")],
    )
    scene = grounding(object_at("hammer"))
    assert validate_plan(candidate, scene, task) == []
    candidate["arms"]["right"]["verb_phrase"] = "grasps the hammer by its handle"
    assert any("required text omitted" in error for error in validate_plan(candidate, scene, task))


def test_simultaneous_actions_survive_rendering():
    task = "Use both hands to lift the two shoes simultaneously."
    candidate = plan()
    candidate["requirements"]["timing"] = {"kind": "simultaneous", "evidence": "simultaneously"}
    candidate["arms"]["right"] = arm(mode="proximity", object_id="shoe_a", phrase="lifts one shoe", actions=[action("lift", "shoe_a")])
    candidate["arms"]["left"] = arm(mode="proximity", object_id="shoe_b", phrase="simultaneously lifts the other shoe", actions=[action("lift", "shoe_b")])
    scene = grounding(object_at("shoe_a", 200), object_at("shoe_b", 800))
    assert validate_plan(candidate, scene, task) == []
    candidate["arms"]["left"]["verb_phrase"] = "lifts the other shoe"
    assert any("omits explicit simultaneous timing" in error for error in validate_plan(candidate, scene, task))
    candidate["requirements"]["timing"]["kind"] = "unspecified"
    assert any("task requires simultaneous" in error for error in validate_plan(candidate, scene, task))


@pytest.mark.parametrize("task", [
    "Take the yellow and black hammer grip and hit the target.",
    "Grab the black and yellow hammer grip and hit the target.",
])
def test_modified_direct_object_grip_cannot_be_silently_dropped(task):
    candidate = plan()
    candidate["arms"]["right"] = arm(object_id="hammer", phrase="grasps the hammer and hits the target", actions=[action("grasp", "hammer")])
    errors = validate_plan(candidate, grounding(object_at("hammer")), task)
    assert any("task contact part omitted: handle" in error for error in errors)


def test_both_mode_is_for_one_shared_object_not_two_independent_objects():
    task = "Use both hands to hold the roller."
    candidate = plan()
    for side in ("right", "left"):
        candidate["arms"][side] = arm(
            mode="both", object_id="roller", evidence="both hands",
            phrase="holds the roller", actions=[action("hold", "roller")],
        )
    assert validate_plan(candidate, grounding(object_at("roller", 500)), task) == []
    candidate["arms"]["left"] = static_arm()
    errors = validate_plan(candidate, grounding(object_at("roller", 500)), task)
    assert any("both mode requires two active arms" in error for error in errors)
    candidate["arms"]["left"] = arm(
        mode="both", object_id="bottle", evidence="both hands",
        phrase="holds the bottle", actions=[action("hold", "bottle")],
    )
    errors = validate_plan(candidate, grounding(object_at("roller"), object_at("bottle", 800)), task)
    assert any("same shared object" in error for error in errors)


@pytest.mark.parametrize("text", [
    'preface {"objects": []}', '{"objects": [], "objects": []}',
    '{"objects": NaN}', '```json\n{}\n```\nMore text', '[]', '{"objects": [}',
])
def test_parse_json_never_repairs_invalid_output(text):
    with pytest.raises(ValueError):
        parse_json(text)


def test_parse_json_accepts_only_raw_or_single_fenced_object():
    assert parse_json('{"objects": []}') == {"objects": []}
    assert parse_json('```json\n{"objects": []}\n```') == {"objects": []}


def test_malformed_schema_returns_flags_instead_of_crashing():
    candidate = plan()
    candidate["arms"]["right"]["assignment"]["mode"] = []
    candidate["arms"]["right"]["actions"][0]["kind"] = []
    candidate["requirements"]["timing"]["kind"] = []
    assert validate_plan(candidate, {"objects": None, "uncertainties": []}, "Lift the bottle.")


@pytest.mark.parametrize("task", [
    "Pick up the roller with both hands.",
    "Use both hands to lift the roller.",
    "Use both hands to elevate the roller.",
    "Use both hands to raise the roller.",
])
def test_lift_task_cannot_collapse_to_only_grasp(task):
    candidate = plan()
    for side in ("right", "left"):
        candidate["arms"][side] = arm(
            mode="both", object_id="roller", evidence="both hands",
            phrase="grasps the roller", actions=[action("grasp", "roller")],
        )
    scene = grounding(object_at("roller", 500))
    errors = validate_plan(candidate, scene, task)
    assert any("caption omits required lifting action" in error for error in errors)
    candidate["arms"]["right"]["verb_phrase"] = "grasps and lifts the roller"
    candidate["arms"]["left"]["verb_phrase"] = "grasps and lifts the roller"
    assert validate_plan(candidate, scene, task) == []


def test_hammer_noun_does_not_count_as_hitting_action():
    task = "Hit the target with the hammer."
    candidate = plan()
    candidate["arms"]["right"] = arm(
        object_id="hammer", phrase="grasps the hammer", actions=[action("grasp", "hammer")],
    )
    scene = grounding(object_at("hammer"))
    assert any("omits required hitting action" in error for error in validate_plan(candidate, scene, task))
    candidate["arms"]["right"]["verb_phrase"] = "grasps the hammer and strikes the target"
    assert validate_plan(candidate, scene, task) == []


def test_handover_task_cannot_leave_both_arms_static_or_omit_transfer_pair():
    task = "Move the bottle to the other hand."
    candidate = plan()
    candidate["arms"] = {"right": static_arm(), "left": static_arm()}
    errors = validate_plan(candidate, grounding(), task)
    assert any("handover task requires both arms" in error for error in errors)
    assert any("handover task requires paired transfer/receive" in error for error in errors)
    candidate["arms"]["right"] = arm()
    candidate["arms"]["left"] = arm(mode="receiver", phrase="receives the bottle", actions=[action("receive", other_arm="right")])
    errors = validate_plan(candidate, grounding(), task)
    assert any("handover task requires paired transfer/receive" in error for error in errors)


@pytest.mark.parametrize("verb", ["Transfer", "Pass"])
def test_transfer_to_container_does_not_authorize_an_extra_arm(verb):
    task = f"{verb} the bread to the basket with the right arm."
    candidate = plan()
    candidate["arms"]["right"] = arm(
        mode="explicit", object_id="bread", evidence="right arm", phrase="places the bread in the basket",
        actions=[action("place", "bread", destination_id="basket")],
    )
    candidate["arms"]["left"] = arm(
        mode="proximity", object_id="basket", phrase="holds the basket", actions=[action("hold", "basket")],
    )
    scene = grounding(object_at("bread"), object_at("basket", 800))
    errors = validate_plan(candidate, scene, task)
    assert any("extra participation" in error for error in errors)
    assert not any("handover task" in error for error in errors)
    candidate["arms"]["left"] = static_arm()
    assert validate_plan(candidate, scene, task) == []


def test_elevate_together_supports_both_arms_on_one_shared_pot():
    task = "Grasp the medium-sized kitchenpot and elevate together."
    candidate = plan()
    candidate["requirements"]["timing"] = {"kind": "simultaneous", "evidence": "together"}
    for side in ("right", "left"):
        candidate["arms"][side] = arm(
            mode="both", object_id="pot", evidence="together",
            phrase="grasps and elevates the kitchenpot together",
            actions=[action("grasp", "pot"), action("lift", "pot")],
        )
    scene = grounding(object_at("pot", 500))
    assert validate_plan(candidate, scene, task) == []
    scene["objects"][0]["confidence"] = "low"
    assert any("low-confidence" in error for error in validate_plan(candidate, scene, task))


def two_bottle_plan():
    candidate = plan()
    candidate["requirements"]["timing"] = {"kind": "simultaneous", "evidence": "simultaneously"}
    candidate["arms"]["right"] = arm(
        mode="both", object_id="bottle_a", evidence="both arms",
        phrase="lifts the first bottle", actions=[action("lift", "bottle_a")],
    )
    candidate["arms"]["left"] = arm(
        mode="explicit", object_id="bottle_b", evidence="both arms",
        phrase="simultaneously lifts the other bottle", actions=[action("lift", "bottle_b")],
    )
    return candidate


def test_assignment_metadata_is_derived_without_rewriting_two_bottle_caption():
    task = "Lift both bottles simultaneously using both arms."
    scene = grounding(object_at("bottle_a", 200), object_at("bottle_b", 800))
    candidate = two_bottle_plan()
    original = copy.deepcopy(candidate)
    original_scene = copy.deepcopy(scene)
    original_caption = render_caption(candidate)
    assert validate_plan(candidate, scene, task)
    resolved = resolve_assignment_modes(candidate, scene, task)
    for side in ("right", "left"):
        assert resolved["arms"][side]["assignment"]["mode"] == "proximity"
        assert resolved["arms"][side]["assignment"]["evidence"] == ""
        assert resolved["arms"][side]["model_assignment"] == original["arms"][side]["assignment"]
        for field in ("active", "actions", "verb_phrase"):
            assert resolved["arms"][side][field] == original["arms"][side][field]
        assert resolved["arms"][side]["assignment"]["object_id"] == original["arms"][side]["assignment"]["object_id"]
    assert render_caption(resolved) == original_caption
    assert validate_plan(resolved, scene, task) == []
    assert candidate == original and scene == original_scene
    assert resolve_assignment_modes(resolved, scene, task) == resolved


def test_assignment_derivation_does_not_hide_wrong_arm_choice():
    task = "Lift both bottles simultaneously using both arms."
    scene = grounding(object_at("bottle_a", 800), object_at("bottle_b", 200))
    candidate = two_bottle_plan()
    resolved = resolve_assignment_modes(candidate, scene, task)
    assert render_caption(resolved) == render_caption(candidate)
    errors = validate_plan(resolved, scene, task)
    assert sum("proximity heuristic conflicts" in error for error in errors) == 2


def test_assignment_derivation_respects_explicit_arm_over_proximity():
    task = "Use the left arm to lift the bottle."
    candidate = plan()
    candidate["arms"]["right"] = static_arm()
    candidate["arms"]["left"] = arm(mode="proximity")
    scene = grounding(object_at(x=200))  # Observer-left would otherwise prefer right.
    resolved = resolve_assignment_modes(candidate, scene, task)
    assert resolved["arms"]["left"]["assignment"]["mode"] == "explicit"
    assert resolved["arms"]["left"]["assignment"]["evidence"] == "left arm"
    assert resolved["arms"]["right"]["assignment"]["mode"] == "inactive"
    assert render_caption(resolved) == render_caption(candidate)
    assert validate_plan(resolved, scene, task) == []


def test_receiver_and_shared_object_assignment_modes_are_derived():
    task = "Use both hands to hold the roller."
    candidate = plan()
    for side in ("right", "left"):
        candidate["arms"][side] = arm(
            mode="proximity", object_id="roller", phrase="holds the roller",
            actions=[action("hold", "roller")],
        )
    scene = grounding(object_at("roller", 500))
    resolved = resolve_assignment_modes(candidate, scene, task)
    assert all(resolved["arms"][side]["assignment"]["mode"] == "both" for side in ("left", "right"))
    assert validate_plan(resolved, scene, task) == []
    candidate["arms"]["left"]["actions"] = [action("receive", "roller", other_arm="right")]
    resolved = resolve_assignment_modes(candidate, scene, task)
    assert resolved["arms"]["left"]["assignment"]["mode"] == "receiver"
    assert resolved["arms"]["left"]["assignment"]["evidence"] == task
    assert any("lacks a matching" in error for error in validate_plan(resolved, scene, task))


def test_non_transfer_action_with_other_arm_is_flagged_not_repaired():
    candidate = plan()
    candidate["arms"]["right"]["actions"][0]["other_arm"] = "left"
    resolved = resolve_assignment_modes(candidate, grounding(), "Lift the bottle.")
    assert resolved["arms"]["right"]["actions"][0]["other_arm"] == "left"
    assert any("only transfer/receive actions" in error for error in validate_plan(resolved, grounding(), "Lift the bottle."))


def test_internal_object_ids_are_flagged_as_whole_words_only():
    candidate = plan()
    candidate["arms"]["right"] = arm(
        object_id="o1", phrase="holds o1", actions=[action("hold", "o1")],
    )
    scene = grounding(object_at("o1"))
    errors = validate_plan(candidate, scene, "Hold the shoe.")
    assert any("internal_object_id_in_caption" in error for error in errors)
    assert not any("internal_object_id_in_caption" in error for error in validate_plan(candidate, scene, "Hold o1."))
    candidate["arms"]["right"]["verb_phrase"] = "holds o10"
    assert not any("internal_object_id_in_caption" in error for error in validate_plan(candidate, scene, "Hold the shoe."))


def test_destination_normalization_allows_into_and_articles_but_preserves_direction():
    task = "Place the bottle in the box."
    candidate = plan()
    candidate["requirements"]["destinations"] = [{"text": "in the box", "evidence": "in the box"}]
    candidate["arms"]["right"] = arm(
        phrase="places the bottle into a box", actions=[action("place", destination_id="box")],
    )
    scene = grounding(object_at(), object_at("box", 500))
    assert validate_plan(candidate, scene, task) == []
    task = "Place the bottle in the left box."
    candidate["requirements"]["destinations"] = [{"text": "in the left box", "evidence": "in the left box"}]
    candidate["arms"]["right"]["verb_phrase"] = "places the bottle into the right box"
    errors = validate_plan(candidate, scene, task)
    assert any("destinations[0]: required text omitted" in error for error in errors)


def test_other_hand_destination_can_name_the_actual_paired_receiver():
    task = "Move the microphone to the other hand."
    candidate = plan()
    candidate["requirements"]["destinations"] = [{"text": "to the other hand", "evidence": "to the other hand"}]
    candidate["arms"]["left"] = arm(
        object_id="microphone", phrase="passes the microphone to the right arm",
        actions=[action("transfer", "microphone", other_arm="right")],
    )
    candidate["arms"]["right"] = arm(
        mode="receiver", object_id="microphone", phrase="receives the microphone from the left robot gripper",
        actions=[action("receive", "microphone", other_arm="left")],
    )
    scene = grounding(object_at("microphone", 800))
    assert validate_plan(candidate, scene, task) == []
    candidate["arms"]["right"]["actions"] = [action("hold", "microphone")]
    errors = validate_plan(candidate, scene, task)
    assert any("destinations[0]: required text omitted" in error for error in errors)


def scanner_plan():
    candidate = plan()
    candidate["arms"]["right"] = arm(
        object_id="box", phrase="holds the box and scans it with the scanner",
        actions=[action("hold", "box"), action("use", "box", tool_id="scanner")],
    )
    candidate["arms"]["left"] = arm(
        object_id="scanner", phrase="grasps and holds the scanner",
        actions=[action("grasp", "scanner"), action("hold", "scanner")],
    )
    return candidate


def test_cross_arm_tool_use_is_flagged_without_rewriting_actions():
    candidate = scanner_plan()
    original = copy.deepcopy(candidate)
    scene = grounding(object_at("box", 200), object_at("scanner", 800))
    errors = validate_plan(candidate, scene, "Scan the box with the scanner.")
    assert len(errors) == 1 and "cross_arm_tool_use" in errors[0]
    assert candidate == original
    # The same tool is explicitly in the acting arm's possession: no exclusive-other-arm conflict.
    candidate["arms"]["right"]["actions"].append(action("hold", "scanner"))
    assert not any("cross_arm_tool_use" in error for error in validate_plan(candidate, scene, "Scan the box with the scanner."))


@pytest.mark.parametrize("transferred_object, expect_conflict", [("scanner", False), ("box", True)])
def test_cross_arm_tool_use_requires_handover_of_that_exact_tool(transferred_object, expect_conflict):
    candidate = scanner_plan()
    candidate["arms"]["left"]["actions"].append(action("transfer", transferred_object, other_arm="right"))
    candidate["arms"]["right"]["actions"].append(action("receive", transferred_object, other_arm="left"))
    scene = grounding(object_at("box", 200), object_at("scanner", 800))
    task = "Pass an object to the other hand and scan the box with the scanner."
    errors = validate_plan(candidate, scene, task)
    assert any("cross_arm_tool_use" in error for error in errors) is expect_conflict
    if not expect_conflict:
        candidate["arms"]["right"]["actions"].pop()
        errors = validate_plan(candidate, scene, task)
        assert any("cross_arm_tool_use" in error for error in errors)


def test_shared_both_arm_tool_is_not_an_exclusive_other_arm_conflict():
    candidate = scanner_plan()
    for side in ("right", "left"):
        candidate["arms"][side]["assignment"] = {
            "mode": "both", "object_id": "scanner", "evidence": "both hands",
        }
    scene = grounding(object_at("box", 200), object_at("scanner", 800))
    task = "Use both hands to scan the box with the scanner."
    assert validate_plan(candidate, scene, task) == []


RGB_TASK = "Arrange red block, green block, and blue block in a row."


def rgb_scene():
    scene = grounding(object_at("o1", 200), object_at("o2", 800), object_at("o3", 300))
    for obj, color in zip(scene["objects"], ("red", "green", "blue")):
        obj["label"] = color + " block"
    return scene


def rgb_plan():
    candidate = plan()
    candidate["arms"]["right"] = arm(
        object_id="o1", phrase="grasps and places the red block, then grasps and places the blue block",
        actions=[action("grasp", "o1"), action("place", "o1"), action("grasp", "o3"), action("place", "o3")],
    )
    candidate["arms"]["left"] = arm(
        object_id="o2", phrase="grasps and places the green block", actions=[action("grasp", "o2"), action("place", "o2")],
    )
    return candidate


def test_rgb_targets_come_from_source_not_task_directory_or_grounding():
    assert required_rgb_objects(RGB_TASK) == ["red", "green", "blue"]
    assert required_rgb_objects("Position blue block, red block and green block in a row.") == ["blue", "red", "green"]
    assert required_rgb_objects("Position red block next to green block.") == []
    assert required_rgb_objects("Lift the bottle.") == []
    assert validate_plan(plan(), grounding(), "Lift the bottle.") == []


def test_rgb_one_arm_can_manipulate_two_blocks_in_sequence():
    candidate = rgb_plan()
    original = copy.deepcopy(candidate)
    assert validate_plan(candidate, rgb_scene(), RGB_TASK) == []
    assert candidate == original
    candidate["arms"]["right"]["verb_phrase"] = "grasps the red and blue blocks in sequence and places them in a row"
    assert validate_plan(candidate, rgb_scene(), RGB_TASK) == []


@pytest.mark.parametrize("reference", ["requirements", "destination", "tool"])
def test_rgb_reference_without_manipulation_object_id_does_not_cover_blue(reference):
    candidate = rgb_plan()
    candidate["arms"]["right"]["actions"] = [action("grasp", "o1"), action("place", "o1")]
    if reference == "requirements":
        candidate["requirements"]["final_states"] = [{"text": "blue block", "evidence": "blue block"}]
    elif reference == "destination":
        candidate["arms"]["right"]["actions"][-1]["destination_id"] = "o3"
    else:
        candidate["arms"]["right"]["actions"][-1]["tool_id"] = "o3"
    errors = validate_plan(candidate, rgb_scene(), RGB_TASK)
    assert any("rgb_object_action_missing: required blue block" in error for error in errors)


def test_rgb_caption_final_order_list_does_not_replace_blue_action():
    candidate = rgb_plan()
    candidate["arms"]["right"]["verb_phrase"] = "grasps and places the red block, resulting in the order red, green, blue from left to right"
    errors = validate_plan(candidate, rgb_scene(), RGB_TASK)
    assert any("rgb_object_caption_action_unverified: blue block" in error for error in errors)
    candidate["arms"]["right"]["verb_phrase"] = "grasps the red block and places it to the left of the blue block"
    errors = validate_plan(candidate, rgb_scene(), RGB_TASK)
    assert any("rgb_object_caption_action_unverified: blue block" in error for error in errors)


def test_rgb_grounding_requires_all_source_objects_and_real_placement_actions():
    candidate = rgb_plan()
    scene = rgb_scene()
    scene["objects"] = scene["objects"][:2]
    errors = validate_plan(candidate, scene, RGB_TASK)
    assert any("rgb_object_grounding_missing: required blue block" in error for error in errors)
    candidate["arms"]["right"]["actions"] = [action("place", "o1"), action("grasp", "o3"), action("hold", "o3")]
    errors = validate_plan(candidate, rgb_scene(), RGB_TASK)
    assert any("rgb_object_placement_action_missing: required blue block" in error for error in errors)


def test_rgb_action_and_caption_must_name_the_same_participating_arm():
    candidate = rgb_plan()
    candidate["arms"]["right"]["verb_phrase"] = "grasps and places the red block"
    candidate["arms"]["left"]["verb_phrase"] = "grasps and places the green block, then grasps and places the blue block"
    errors = validate_plan(candidate, rgb_scene(), RGB_TASK)
    assert any("plan.arms.right: rgb_object_caption_action_unverified: blue block" in error for error in errors)
