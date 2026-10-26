#!/usr/bin/env python3
"""Package agent-authored captions; no model inference or automatic rewriting."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--drafts', type=Path, nargs='+', required=True)
    args = ap.parse_args()
    project, run = args.project.resolve(), args.run.resolve()
    sys.path.insert(0, str(project / 'scripts'))
    from enhance_prompts import check_caption
    from prompt_plan import VIEW_SUFFIX, _rgb_action_mentions
    from build_prompt_review_queue import write_csv

    if (run / 'summary.json').exists() or (run / 'review.html').exists():
        raise ValueError('Refusing to overwrite a published bundle or review annotations')
    inputs = json.loads((run / 'input_manifest.json').read_text())
    sources = {r['episode']: r for r in inputs}
    drafts = [r for path in args.drafts for r in json.loads(path.read_text())]
    assert len(inputs) == len(sources) == len(drafts) == 50
    assert {r['episode'] for r in drafts} == set(sources)
    dataset = project / 'data/robotwin_observer_ego_vpp2'
    manifests = {r['episode']: r for r in map(json.loads, (dataset / 'manifest.jsonl').read_text().splitlines())
                 if r['task'] == 'blocks_ranking_rgb'}
    old_run = project / 'runs/prompt_two_step_full_v1'
    old_hashes, records, items, comparisons = {}, [], [], []
    for draft in sorted(drafts, key=lambda r: r['episode']):
        source = sources[draft['episode']]
        for key in ('task', 'episode', 'image', 'original_instruction'):
            assert draft[key] == source[key], (draft['episode'], key)
        assert digest(Path(source['image_path'])) == source['image_sha256']
        instruction_path = Path(source['instruction_path'])
        assert json.loads(instruction_path.read_text())['seen'][0] == source['original_instruction']
        caption, errors = check_caption(draft['enhanced_instruction'])
        assert not errors, (draft['episode'], errors)
        assert caption == draft['enhanced_instruction']
        assert set(draft['arm_assignment']) == {'red', 'green', 'blue'}
        assert set(draft['visual_observations']) == {'red', 'green', 'blue'}
        body = caption.removesuffix('. ' + VIEW_SUFFIX)
        right, left = body.split(', while the left robot gripper ')
        mentions = {'right': _rgb_action_mentions(right), 'left': _rgb_action_mentions(left)}
        for color, arm in draft['arm_assignment'].items():
            assert arm in ('right', 'left'), (draft['episode'], color, arm)
            assert color in mentions[arm], (draft['episode'], color, 'no direct object action', mentions)
        assert mentions['right'] | mentions['left'] == {'red', 'green', 'blue'}
        old_path = old_run / 'records' / source['task'] / (source['episode'] + '.json')
        old_hashes[str(old_path)] = digest(old_path)
        old = json.loads(old_path.read_text())
        flags = draft['review_flags']
        assert isinstance(flags, list) and all(isinstance(f, str) for f in flags)
        record = {**source, **draft,
                  'authoring_method': 'direct_agent_image_and_task_review',
                  'model_generated_grounding_or_plan_used': False,
                  'instruction_sha256': digest(instruction_path),
                  'review_status': 'needs_source_clarification' if flags else 'agent_reviewed',
                  'user_approved': False,
                  'validation': {'caption_structure': 'passed', 'rgb_direct_action_coverage': 'passed',
                                 'source_image_hash': 'passed', 'source_instruction': 'passed'},
                  'evidence_scope': 'Initial observer+ego image and original seen[0] instruction. Arm choice from proximity is an inference, not a full-video execution audit.'}
        records.append(record)
        record_path = run / 'records' / source['task'] / (source['episode'] + '.json')
        manifest = manifests[source['episode']]
        item = {'id': source['task'] + '/' + source['episode'],
                'task': source['task'], 'episode': source['episode'],
                'priority': 'P1' if flags else 'P3',
                'priority_reason': '原任务的最终位置不明确，请先确认目标位置。' if flags else 'Agent 已逐项核对三物体动作、机械臂、任务方向和顺序；可播放视频继续复核。',
                'issue_labels': ['原任务目标位置待确认'] if flags else ['Agent 已核对'],
                'review_flags': flags, 'original_instruction': source['original_instruction'],
                'enhanced_instruction': caption, 'video_path': source['video_path'],
                'image_path': source['image_path'], 'record_path': str(record_path),
                'observer_video_path': manifest['inputs']['observer_video']['path'],
                'ego_video_path': manifest['inputs']['ego_video']['path'],
                'video_url': f"videos/{source['task']}/{source['episode']}.mp4",
                'image_url': source['image'],
                'record_url': str(record_path.relative_to(run)),
                'has_caption': True, 'decision': '待复核', 'corrected_prompt': '',
                'notes': draft['notes'] if isinstance(draft['notes'], str) else json.dumps(draft['notes'], ensure_ascii=False)}
        items.append(item)
        comparisons.append({'episode': source['episode'], 'original_instruction': source['original_instruction'],
                            'previous_qwen_prompt': old['enhanced_instruction'], 'agent_prompt': caption,
                            'agent_review_flags': ' | '.join(flags), 'video_path': source['video_path']})

    # All validation precedes publication. Old model records are read only.
    for record in records:
        path = run / 'records' / record['task'] / (record['episode'] + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(path, record)
    for path in args.drafts:
        target = run / 'drafts' / path.name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(path.read_bytes())
    for name in ('images', 'videos'):
        (run / name).symlink_to(dataset / name, target_is_directory=True)
    def export(name, rows, key):
        (run / name).write_text(''.join(r[key] + '@@' + r['image'] + '\n' for r in rows))
    reviewed = [r for r in records if not r['review_flags']]
    unresolved = [r for r in records if r['review_flags']]
    export('text_original.txt', records, 'original_instruction')
    export('text_agent_candidates_all.txt', records, 'enhanced_instruction')
    export('text_agent_prompt_enhanced.txt', reviewed, 'enhanced_instruction')
    for name, rows in [('records.jsonl', records), ('needs_review.jsonl', unresolved)]:
        (run / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    write_csv(run / 'review_all.csv', items)
    write_csv(run / 'needs_review.csv', [r for r in items if r['review_flags']])
    with (run / 'old_vs_agent.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparisons[0])); writer.writeheader(); writer.writerows(comparisons)
    run_id = 'agent_rgb_v1_' + digest(run / 'records.jsonl')[:20]
    payload = {'run_id': run_id, 'generated_at': datetime.now(timezone.utc).isoformat(),
               'counts': {'nonempty_review': len(items), 'empty': 0},
               'items': sorted(items, key=lambda r: (r['priority'], r['episode']))}
    save_json(run / 'review_items.json', payload)
    template = (project / 'scripts/review_template.html').read_text()
    template = template.replace('VPP2 · 视频 Prompt 复核', 'VPP2 · Agent 直接编写的 RGB Prompt')
    template = template.replace('候选扩写 Prompt', 'Agent 编写的 Prompt')
    template = template.replace('查看自动检查的原始提示', '查看原任务中的待确认项')
    template = template.replace('自动检查的提示不等于错误结论。', '本页是独立的 Agent 编写版本；无未解决标记的条目也未视为用户已批准。')
    safe = json.dumps(payload, ensure_ascii=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    safe = safe.replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    (run / 'review.html').write_text(template.replace('__REVIEW_DATA_JSON__', safe))
    assert all(digest(Path(p)) == h for p, h in old_hashes.items())
    assert all(Path(i[k]).is_file() for i in items for k in ('video_path', 'image_path', 'observer_video_path', 'ego_video_path', 'record_path'))
    summary = {'run_id': run_id, 'authoring_method': 'direct_agent_image_and_task_review',
               'task': 'blocks_ranking_rgb', 'total': len(records), 'nonempty': len(records),
               'agent_reviewed_without_unresolved_flags': len(reviewed),
               'needs_source_clarification': len(unresolved),
               'unresolved_episodes': [r['episode'] for r in unresolved],
               'rgb_action_coverage': '50/50', 'caption_format_checks': '50/50',
               'source_instruction_and_image_hash_checks': '50/50',
               'all_media_paths_exist': True, 'old_50_records_unchanged': True,
               'old_record_sha256': old_hashes,
               'full_video_execution_audited': False, 'applied_to_existing_training_manifest': False,
               'text_exports': {'text_agent_candidates_all.txt': len(records), 'text_agent_prompt_enhanced.txt': len(reviewed)}}
    save_json(run / 'summary.json', summary)
    (run / 'README.txt').write_text(
        '本目录为 agent 直接查看首帧并根据原任务编写的独立结果，不是 Qwen 重跑。\n'
        f'50 条均有红、绿、蓝三块的明确操作；{len(reviewed)} 条无未解决标记，{len(unresolved)} 条原任务的目标位置需确认。\n'
        'text_agent_candidates_all.txt：全部 50 条，包含待确认条目，供审阅。\n'
        'text_agent_prompt_enhanced.txt：排除待确认条目的独立 VPP2 格式清单。每行 prompt@@原图片相对路径。\n'
        'review.html：首帧、完整视频、原任务、agent prompt 与备注；导出 JSON 保存个人审核。\n'
        'old_vs_agent.csv：旧 Qwen 与新 agent prompt 对照。needs_review.csv：待确认条目。\n'
        '镜像只用于识别臂：observer左=实际右臂；物体目标左右关系沿用原任务。\n'
        '未指定机械臂的任务按首帧接近关系选择；并未对全部视频的实际动作逐帧审计。\n'
        '未覆盖旧模型结果、旧人工审核记录或现有训练清单；网页初始状态不等于用户已批准。\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'old_record_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
