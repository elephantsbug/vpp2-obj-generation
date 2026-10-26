"""CPU-only review tests: no torch import or model loading required."""
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "enhance_prompts.py"
spec = importlib.util.spec_from_file_location("enhance_prompts", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def make_dataset(tmp_path):
    dataset = tmp_path / "dataset"
    image = dataset / "images" / "task" / "episode_0000000.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (448, 224), "green").save(image)
    row = {
        "task": "task", "episode": "episode_0000000",
        "view_order": ["observer", "ego"],
        "geometry": {"output_size_wh": [448, 224]},
        "outputs": {"image": str(image.relative_to(dataset))},
        "output_sha256": {"image": runner.sha256(image)},
        "instruction": "Use the right arm to pick up the bottle.",
    }
    (dataset / "manifest.jsonl").write_text(json.dumps(row) + "\n")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps([{"task": row["task"], "episode": row["episode"]}]))
    return dataset, selection, row


def test_selected_image_integrity_and_view_order(tmp_path):
    dataset, selection, row = make_dataset(tmp_path)
    assert runner.select_rows(dataset, selection, False) == [row]
    row["view_order"] = ["ego", "observer"]
    (dataset / "manifest.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="observer-left"):
        runner.select_rows(dataset, selection, False)
    row["view_order"] = ["observer", "ego"]
    (dataset / "manifest.jsonl").write_text(json.dumps(row) + "\n")
    Image.new("RGB", (448, 224), "red").save(dataset / row["outputs"]["image"])
    with pytest.raises(ValueError, match="Changed initial image"):
        runner.select_rows(dataset, selection, False)


def test_preview_limit_and_duplicate_selection(tmp_path):
    dataset, selection, row = make_dataset(tmp_path)
    key = {"task": row["task"], "episode": row["episode"]}
    selection.write_text(json.dumps([key] * 21))
    with pytest.raises(ValueError, match="full generation requires --all"):
        runner.select_rows(dataset, selection, False)
    selection.write_text(json.dumps([key, key]))
    with pytest.raises(ValueError, match="Duplicate preview"):
        runner.select_rows(dataset, selection, False)


def test_arm_identity_suffix_and_static_format():
    valid = "The right robot gripper picks up the bottle, while the left robot gripper keeps static. " + runner.SUFFIX
    assert runner.check_caption(valid) == (valid, [])
    swapped = valid.replace("right robot gripper", "LEFT_PLACEHOLDER").replace("left robot gripper", "right robot gripper").replace("LEFT_PLACEHOLDER", "left robot gripper")
    assert "expected_actual_right_arm_first" in runner.check_caption(swapped)[1]
    assert "expected_one_while_left_arm_clause" in runner.check_caption(swapped)[1]
    assert "use_exact_keeps_static" in runner.check_caption(valid.replace("keeps static", "remains static"))[1]
    assert "unexpected_markup_or_path_separator" in runner.check_caption(valid + "@@images/a.png")[1]


@pytest.mark.parametrize("action", [
    "The right robot gripper picks up the bottle, while the left robot gripper keeps static.",
    "The right robot gripper keeps static, while the left robot gripper picks up the bottle.",
])
def test_static_arm_can_be_either_arm_and_suffix_requires_period(action):
    valid = action + " " + runner.SUFFIX
    assert runner.check_caption(valid) == (valid, [])
    missing_period = action.removesuffix(".") + " " + runner.SUFFIX
    assert "incorrect_view_suffix" in runner.check_caption(missing_period)[1]


@pytest.mark.parametrize("action", [
    "The right robot gripper picks up the hammer, while the left robot gripper keeps static as the right robot gripper strikes the block.",
    "The right robot gripper keeps static as the left robot gripper places the bottle down, while the left robot gripper picks up the bottle.",
])
def test_static_clause_cannot_contain_another_arms_later_actions(action):
    caption = action + " " + runner.SUFFIX
    assert "static_clause_has_extra_actions" in runner.check_caption(caption)[1]


def test_export_retains_raw_failures_and_identical_image_paths(tmp_path):
    dataset, _, row = make_dataset(tmp_path)
    before = {str(p.relative_to(dataset)): runner.sha256(p) for p in dataset.rglob("*") if p.is_file()}
    output = tmp_path / "preview"
    caption = "The right robot gripper picks up the bottle, while the left robot gripper keeps static. " + runner.SUFFIX
    result = {"task": row["task"], "episode": row["episode"], "image": row["outputs"]["image"],
              "original_instruction": row["instruction"], "enhanced_instruction": caption,
              "raw_response": caption, "format_errors": ["generation_did_not_end_with_eos"]}
    runner.write_json(runner.record_path(output, row), result)
    config = {"mode": "preview", "model": "local-model", "gpu_groups": ["4,5", "6,7"]}
    summary = runner.export_review(output, dataset, [row], config)
    assert summary["format_failed"] == 1
    assert (output / "text_prompt_enhanced.txt").read_text() == ""
    assert json.loads((output / "results.json").read_text())[0]["raw_response"] == caption
    assert (output / "text_original.txt").read_text() == f"{row['instruction']}@@{row['outputs']['image']}\n"
    assert (output / result["image"]).resolve() == (dataset / result["image"]).resolve()
    assert "generation_did_not_end_with_eos" in (output / "review.html").read_text()
    after = {str(p.relative_to(dataset)): runner.sha256(p) for p in dataset.rglob("*") if p.is_file()}
    assert before == after

