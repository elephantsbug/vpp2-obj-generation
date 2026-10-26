"""Check complete RGB/obj episode pairing without changing dataset contents."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

PROJECT = Path('/data2/niurc/vpp2-obj-generation')
OUT = PROJECT / 'data/robotwin_observer_ego_obj_full_vpp2'
RGB = PROJECT / 'data/robotwin_observer_ego_vpp2'

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def key(row):
    return row['task'], row['episode']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

records = rows(OUT / 'manifest.jsonl')
rgb_records = {key(r): r for r in rows(RGB / 'manifest.jsonl')}
pairs = rows(OUT / 'paired_all.jsonl')
clean = rows(OUT / 'paired_no_flags.jsonl')
flagged = rows(OUT / 'paired_needs_review.jsonl')
prompts_path = PROJECT / 'runs/prompt_all_video_agent_v1/exports/records.jsonl'
prompts = {key(r): r for r in rows(prompts_path)}
prompt_hash = digest(prompts_path)
summary = json.loads((OUT / 'summary.json').read_text())
assert len(records) == len(pairs) == len(rgb_records) == 2500
assert len({key(r) for r in records}) == 2500
assert len({key(r) for r in pairs}) == 2500
assert {key(r) for r in records} == {key(r) for r in pairs} == set(rgb_records) == set(prompts)
assert Counter(r['task'] for r in records) == Counter({task: 50 for task in {r['task'] for r in records}})
assert len({r['task'] for r in records}) == 50
assert clean == [r for r in pairs if not r['prompt_review_flags']]
assert flagged == [r for r in pairs if r['prompt_review_flags']]
assert len(clean) == 2395 and len(flagged) == 105
by_key = {key(r): r for r in records}
for p in pairs:
    r, rgb, prompt = by_key[key(p)], rgb_records[key(p)], prompts[key(p)]
    assert r['output_video'] == rgb['output_video'] == p['video_metadata']
    assert (r['output_video']['width'], r['output_video']['height']) == (448, 224)
    assert r['view_order'] == ['observer', 'ego']
    assert r['content']['source_video_kind'] == p['target_source_kind'] == 'video_object_full'
    assert p['condition_source_kind'] == 'video'
    assert p['condition_video_path'] == str(RGB / rgb['outputs']['video'])
    assert p['condition_episode_first_frame_path'] == str(RGB / rgb['outputs']['image'])
    assert p['target_video_path'] == str(OUT / r['outputs']['video'])
    assert p['target_first_frame_path'] == str(OUT / r['outputs']['image'])
    assert p['timestamps_path'] == str(OUT / r['outputs']['timestamps'])
    assert p['record_path'] == str(OUT / r['outputs']['record'])
    assert json.loads(Path(p['record_path']).read_text()) == r
    assert p['prompt'] == prompt['enhanced_instruction']
    assert p['prompt_review_flags'] == prompt['review_flags']
    assert p['prompt_records_sha256'] == prompt_hash
    assert p['rgb_sha256'] == rgb['output_sha256']
    assert p['obj_sha256'] == r['output_sha256']
    for view in ('observer', 'ego'):
        assert Path(r['inputs'][view + '_video']['path']).parent.name == 'video_object_full'
        assert Path(rgb['inputs'][view + '_video']['path']).parent.name == 'video'
        assert r['source_videos'][view] == rgb['source_videos'][view]
    for media in ('video', 'image', 'timestamps'):
        assert digest(OUT / r['outputs'][media]) == r['output_sha256'][media]
        assert digest(RGB / rgb['outputs'][media]) == rgb['output_sha256'][media]
    assert (OUT / r['outputs']['timestamps']).read_bytes() == (RGB / rgb['outputs']['timestamps']).read_bytes()
    assert r['timing']['rgb_obj_csv_timestamps_equal']
    assert r['timing']['rgb_obj_frame_count_and_fps_equal']
assert summary['failed'] == 0 and summary['paired_episodes'] == 2500
assert summary['total_frames'] == sum(r['output_video']['frame_count'] for r in records)
report = {
    'checked_at': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
    'tasks': 50, 'episodes': 2500, 'obj_video_frames': summary['total_frames'],
    'dimensions_wh': [448, 224], 'view_order': ['observer', 'ego'],
    'without_unresolved_prompt_flags': len(clean), 'with_prompt_flags': len(flagged),
    'all_rgb_obj_media_hashes': 'passed', 'all_simulation_timestamp_files_byte_equal': True,
    'paired_indexes_match_records_and_prompts': True,
    'pixel_composition_integration_tests': '5 passed before full conversion',
    'real_visual_spot_check': 'blocks_ranking_rgb/episode_0000000 frames 0,450,899; source adjust_bottle episode 0 first frames',
    'training_adapter_implemented': False, 'training_started': False,
}
(OUT / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
