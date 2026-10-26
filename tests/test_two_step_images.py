"""Regressions for actual Qwen image coordinates and original view integrity."""
from pathlib import Path
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from enhance_prompts_two_step import build_plan_prompt, normalize_observer_pixels, split_views
from prompt_plan import enrich_grounding


def test_pixel_coordinate_mapping_uses_actual_input_size():
    raw = {'coordinate_system': 'observer_pixels_896x896', 'objects': [
        {'id': 'a', 'label': 'object A', 'center': [400, 520], 'confidence': 'high'},
        {'id': 'b', 'label': 'object B', 'center': [600, 700], 'confidence': 'high'}], 'uncertainties': []}
    normalized = normalize_observer_pixels(raw)
    enriched, flags = enrich_grounding(normalized)
    assert not flags
    assert [obj['preferred_arm'] for obj in enriched['objects']] == ['right', 'left']
    assert raw['objects'][0]['center'] == [400, 520]
    assert enriched['objects'][1]['observer_center_px'] == [600, 700]


def test_wrong_coordinate_scale_is_not_silently_guessed():
    with pytest.raises(ValueError, match='896x896'):
        normalize_observer_pixels({'coordinate_system': 'normalized_1000', 'objects': []})
    with pytest.raises(ValueError, match='outside'):
        normalize_observer_pixels({'coordinate_system': 'observer_pixels_896x896', 'objects': [{'center': [950, 200]}]})


def test_unlocalized_object_kept_for_review():
    raw = {'coordinate_system': 'observer_pixels_896x896', 'objects': [
        {'id': 'o', 'label': 'occluded object', 'center': None, 'confidence': 'low'}], 'uncertainties': ['occluded']}
    normalized = normalize_observer_pixels(raw)
    enriched, flags = enrich_grounding(normalized)
    assert flags and enriched['objects'][0]['preferred_arm'] is None


def test_split_preserves_view_order_without_mirroring():
    im = Image.new('RGB', (448, 224))
    im.putpixel((10, 23), (255, 0, 0))
    im.putpixel((224 + 17, 31), (0, 255, 0))
    observer, ego = split_views(im)
    assert observer.getpixel((10, 23)) == (255, 0, 0)
    assert ego.getpixel((17, 31)) == (0, 255, 0)
    reconstructed = Image.new('RGB', im.size)
    reconstructed.paste(observer, (0, 0))
    reconstructed.paste(ego, (224, 0))
    assert reconstructed.tobytes() == im.tobytes()


@pytest.mark.parametrize('arm,other', [('left', 'right'), ('right', 'left')])
def test_explicit_single_arm_is_prominent_before_generation(arm, other):
    task = f'Use the {arm} arm to grasp the hammer handle and hit the block.'
    prompt = build_plan_prompt('{grounding_json}\nTask: {task_instruction}', {'objects': []}, task)
    assert task in prompt
    assert f'"required_active_arms": ["{arm}"]' in prompt
    assert f'"required_inactive_arms": ["{other}"]' in prompt
    assert 'Do not invent an arm-to-arm handover' in prompt


@pytest.mark.parametrize('task', [
    'Grasp the block with the left arm and pass it to the other arm.',
    'Lift the tray with both arms.',
    'Pick up the two bottles together.',
    'Use the left and right arms to lift the roller.',
])
def test_handover_and_two_arm_tasks_not_forced_single_arm(task):
    prompt = build_plan_prompt('{grounding_json}\n{task_instruction}', {'objects': []}, task)
    assert 'required_inactive_arms' not in prompt


def test_object_spatial_direction_does_not_specify_arm():
    prompt = build_plan_prompt('{grounding_json}\n{task_instruction}', {}, 'Place the mouse right of the stapler.')
    assert '"explicit_actual_arms": []' in prompt
    assert 'required_inactive_arms' not in prompt


def test_three_block_prompt_requires_three_actions_not_three_arms():
    scene = {'objects': [{'id': 'a', 'label': 'red block'}, {'id': 'b', 'label': 'green cube'},
                         {'id': 'c', 'label': 'blue block'}]}
    task = 'Using the right arm, the left arm, and the right arm, set red block, green block, and blue block from left to right.'
    prompt = build_plan_prompt('{grounding_json}\n{task_instruction}', scene, task)
    assert task in prompt
    assert all('"task_object": "' + color + ' block"' in prompt for color in ['red', 'green', 'blue'])
    assert '"grounding_object_ids": ["c"]' in prompt
    assert 'required_inactive_arms' not in prompt
    assert 'Each arm can operate multiple blocks in sequence' in prompt


def test_non_rgb_task_does_not_receive_three_block_requirement():
    prompt = build_plan_prompt('{grounding_json}\n{task_instruction}', {}, 'Lift both bottles together.')
    assert 'required_manipulation_targets' not in prompt
