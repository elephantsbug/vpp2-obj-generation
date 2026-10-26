#!/usr/bin/env python3
"""Validate and export already-authored video-grounded RGB prompts."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--drafts', type=Path, nargs='+', required=True)
    args = ap.parse_args()
    project, run = args.project.resolve(), args.run.resolve()
    sys.path.insert(0, str(project / 'scripts'))
    from enhance_prompts import check_caption
    from prompt_plan import _rgb_action_mentions, VIEW_SUFFIX
    from build_prompt_review_queue import write_csv

    assert not (run / 'review.html').exists(), 'Existing review page must not be overwritten'
    inputs = json.loads((run / 'input_manifest.json').read_text())
    sources = {r['episode']: r for r in inputs}
    drafts = [r for p in args.drafts for r in json.loads(p.read_text())]
    assert len(inputs) == len(sources) == len(drafts) == 50
    assert {r['episode'] for r in drafts} == set(sources)
    dataset = project / 'data/robotwin_observer_ego_vpp2'
    old_run = project / 'runs/prompt_rgb_agent_v1'
    manifests = {r['episode']: r for r in map(json.loads, (dataset / 'manifest.jsonl').read_text().splitlines()) if r['task'] == 'blocks_ranking_rgb'}
    records, items, comparison, old_hashes = [], [], [], {}
    cn_arm = {'left': '实际左臂', 'right': '实际右臂'}
    cn_color = {'red': '红块', 'green': '绿块', 'blue': '蓝块'}
    for draft in sorted(drafts, key=lambda r: r['episode']):
        source = sources[draft['episode']]
        ep = source['episode']
        for key in ('task', 'episode', 'image', 'original_instruction'):
            assert draft[key] == source[key], (ep, key)
        for key in source.keys() & draft.keys():
            assert draft[key] == source[key], (ep, 'draft changed source field', key)
        assert sha(source['image_path']) == source['image_sha256'], ep
        assert sha(source['video_path']) == source['video_sha256'], ep
        assert json.loads(Path(source['instruction_path']).read_text())['seen'][0] == source['original_instruction']
        caption, errors = check_caption(draft['enhanced_instruction'])
        assert not errors and caption == draft['enhanced_instruction'], (ep, errors)
        clauses = dict(zip(('right', 'left'), caption.removesuffix('. ' + VIEW_SUFFIX).split(', while the left robot gripper ')))
        mentions = {a: _rgb_action_mentions(s) for a, s in clauses.items()}
        assert set(draft['arm_assignment']) == {'red', 'green', 'blue'}
        for color, arm in draft['arm_assignment'].items():
            assert color in mentions[arm], (ep, color, arm, 'direct action missing')
        sequence = draft['video_sequence']
        assert len(sequence) == 3 and [s['order'] for s in sequence] == [1, 2, 3], ep
        assert {s['object'] for s in sequence} == {'red', 'green', 'blue'}, ep
        assert draft['first_active_arm'] == sequence[0]['arm'], ep
        for step in sequence:
            assert step['arm'] == draft['arm_assignment'][step['object']], ep
            start, end = step['observed_time_range_s']
            assert 0 <= start <= end <= source['duration_s'] + 1.01, (ep, start, end)
            evidence_frames = step['evidence_frames']
            indices = [f['frame'] if isinstance(f, dict) else f for f in evidence_frames]
            assert indices and all(isinstance(i, int) and 0 <= i < source['decoded_frames'] for i in indices), ep
            for evidence in evidence_frames:
                if isinstance(evidence, dict):
                    assert abs(evidence['time_s'] - evidence['frame'] / source['fps']) < 0.04, ep
                    assert Path(evidence['contact_sheet']).is_file(), ep
        for prev, step in zip(sequence, sequence[1:]):
            assert prev['observed_time_range_s'][0] < step['observed_time_range_s'][0], ep
            if prev['arm'] != step['arm']:
                # This checks an explicit dependency exists; binding and semantics need agent review.
                color, previous_arm = prev['object'], prev['arm']
                passive = rf'\b(?:after|once) the {color} block (?:has been|is) (?:placed|set)\b'
                active = rf'\b(?:after|once) the {previous_arm} robot gripper has (?:placed|set) the {color} block\b'
                assert re.search(passive + '|' + active, clauses[step['arm']], re.I), (ep, color, 'missing cross-arm dependency')
        expected_pages = {p['path'] for p in source['contact_sheets']}
        assert expected_pages.issubset(set(draft['evidence_pages_viewed'])), (ep, 'unviewed timeline page')
        assert all(Path(p).is_file() for p in draft['evidence_pages_viewed']), ep
        for key in ('final_observer_order_left_to_right', 'final_ego_order_left_to_right'):
            assert set(draft[key]) == {'red', 'green', 'blue'} and len(draft[key]) == 3, (ep, key)
        assert 'ego view' in caption.removesuffix(VIEW_SUFFIX), (ep, 'missing spatial reference')
        flags = draft['review_flags']
        assert isinstance(flags, list)
        old_path = old_run / 'records' / source['task'] / (ep + '.json')
        old_hashes[str(old_path)] = sha(old_path)
        old = json.loads(old_path.read_text())
        record = {**source, **draft,
                  'authoring_method': 'direct_agent_video_timeline_and_task_review',
                  'evidence_scope': 'Entire clip decoded; agents visually reviewed 1-second samples including first and last frames. Fine temporal windows, where recorded, provide additional evidence. Not every video frame was visually inspected.',
                  'instruction_sha256': sha(source['instruction_path']),
                  'previous_first_frame_review_flags': old['review_flags'],
                  'user_approved': False,
                  'review_status': 'needs_review' if flags else 'video_agent_reviewed',
                  'validation': {'caption_structure': 'passed', 'rgb_direct_actions': 'passed', 'cross_arm_dependency_present': 'passed', 'source_hashes': 'passed', 'all_timeline_pages_declared_viewed': 'passed'}}
        records.append(record)
        seq_text = ' → '.join(cn_arm[s['arm']] + '拾起并放好' + cn_color[s['object']] for s in sequence)
        timing_text = '；'.join(f"{cn_color[s['object']]}约 {s['observed_time_range_s'][0]}–{s['observed_time_range_s'][1]} 秒" for s in sequence)
        record_path = run / 'records' / source['task'] / (ep + '.json')
        manifest = manifests[ep]
        items.append({'id': source['task'] + '/' + ep, 'task': source['task'], 'episode': ep,
                      'priority': 'P1' if flags else 'P3',
                      'priority_reason': '视频核验仍有待确认项。' if flags else '已查看整段视频时间轴，核对实际用臂、三块先后和最终排列。',
                      'issue_labels': ['视频待确认'] if flags else ['视频时序已核验'],
                      'review_flags': flags, 'original_instruction': source['original_instruction'],
                      'enhanced_instruction': caption, 'image_path': source['image_path'], 'video_path': source['video_path'],
                      'record_path': str(record_path), 'observer_video_path': manifest['inputs']['observer_video']['path'],
                      'ego_video_path': manifest['inputs']['ego_video']['path'],
                      'image_url': source['image'], 'video_url': f"videos/{source['task']}/{ep}.mp4",
                      'record_url': str(record_path.relative_to(run)), 'has_caption': True, 'decision': '待复核',
                      'corrected_prompt': '', 'notes': draft['notes'] if isinstance(draft['notes'], str) else json.dumps(draft['notes'], ensure_ascii=False),
                      'sequence_text': seq_text, 'timing_text': timing_text,
                      'evidence_urls': [str(Path(p).relative_to(run)) for p in draft['evidence_pages_viewed']],
                      'final_views_text': 'ego 左→右：' + '、'.join(cn_color[c] for c in draft['final_ego_order_left_to_right']) + '；observer 左→右：' + '、'.join(cn_color[c] for c in draft['final_observer_order_left_to_right'])})
        comparison.append({'episode': ep, 'first_active_arm': draft['first_active_arm'],
                           'video_sequence': seq_text, 'approximate_action_intervals': timing_text,
                           'previous_first_frame_prompt': old['enhanced_instruction'], 'video_prompt': caption,
                           'review_flags': ' | '.join(flags), 'video_path': source['video_path']})
    # Publish only after all records validate; source captions and media are read-only.
    for record in records:
        p = run / 'records' / record['task'] / (record['episode'] + '.json')
        p.parent.mkdir(parents=True, exist_ok=True)
        write_json(p, record)
    (run / 'drafts').mkdir(exist_ok=True)
    for p in args.drafts:
        (run / 'drafts' / p.name).write_bytes(p.read_bytes())
    for name in ('images', 'videos'):
        (run / name).symlink_to(dataset / name, target_is_directory=True)
    reviewed = [r for r in records if not r['review_flags']]
    unresolved = [r for r in records if r['review_flags']]
    for name, rows, key in [('text_original.txt', records, 'original_instruction'),
                            ('text_video_candidates_all.txt', records, 'enhanced_instruction'),
                            ('text_video_prompt_enhanced.txt', reviewed, 'enhanced_instruction')]:
        (run / name).write_text(''.join(r[key] + '@@' + r['image'] + '\n' for r in rows))
    for name, rows in [('records.jsonl', records), ('needs_review.jsonl', unresolved)]:
        (run / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    write_csv(run / 'review_all.csv', items)
    write_csv(run / 'needs_review.csv', [i for i in items if i['review_flags']])
    with (run / 'first_frame_vs_video.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0])); writer.writeheader(); writer.writerows(comparison)
    with (run / 'video_sequence.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', '先动的实际机械臂', '视频动作顺序', '操作阶段近似区间', '最终双视角排列', '视频路径'])
        for r, i in zip(records, items):
            writer.writerow([r['episode'], cn_arm[r['first_active_arm']], i['sequence_text'], i['timing_text'], i['final_views_text'], r['video_path']])
    run_id = 'rgb_video_agent_v1_' + sha(run / 'records.jsonl')[:20]
    payload = {'run_id': run_id, 'generated_at': datetime.now(timezone.utc).isoformat(),
               'counts': {'nonempty_review': len(items), 'empty': 0}, 'items': items}
    write_json(run / 'review_items.json', payload)
    template = (project / 'scripts/review_template.html').read_text()
    template = template.replace('VPP2 · 视频 Prompt 复核', 'VPP2 · RGB 视频时序 Prompt')
    template = template.replace('候选扩写 Prompt', '根据视频重写的 Prompt')
    template = template.replace('查看自动检查的原始提示', '查看视频核验中的待确认项')
    template = template.replace('生成时使用的首帧', '对应视频首帧')
    template = template.replace('自动检查的提示不等于错误结论。', '本版依据完整时长的抽帧序列编写；空间左右以 robot ego view 为参照。')
    template = template.replace('视频用于核验执行情况；新增描述仍须由首帧和原任务支持，不能把后续画面透露的额外细节写入扩写。信息冲突或无法确认时选“暂缓”并记原因。', '本次授权按视频重写：时序、用臂和最终位置以逐段视频证据核对。视频与原任务实质冲突时保留标记；操作时间为抽帧估计区间，不是精确事件时刻。')
    anchor = '<div class="panel"><h3>原任务指令</h3>'
    panel = '<div class="panel"><h3>视频动作顺序</h3><p id="sequence" class="textblock"></p><p id="timing"></p><p id="final-views"></p><p>已完整解码视频，逐页检查每秒抽帧并包含首尾帧；时间为操作阶段的近似区间。</p><div id="evidence-links" class="media-links"></div></div>\n        '
    assert anchor in template
    template = template.replace(anchor, panel + anchor)
    js_anchor = "$('original').textContent = str(item.original_instruction);"
    extra_js = """
      $('sequence').textContent = str(item.sequence_text);
      $('timing').textContent = str(item.timing_text);
      $('final-views').textContent = str(item.final_views_text);
      $('evidence-links').replaceChildren(...array(item.evidence_urls).map((url, index) => {
        const a = document.createElement('a');
        const safe = safeRelativeURL(url);
        if (safe) a.href = safe;
        a.textContent = '查看时间轴第 ' + (index + 1) + ' 页'; a.target = '_blank'; a.rel = 'noopener';
        return a;
      }));
"""
    assert js_anchor in template
    template = template.replace(js_anchor, js_anchor + extra_js)
    safe = json.dumps(payload, ensure_ascii=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    (run / 'review.html').write_text(template.replace('__REVIEW_DATA_JSON__', safe))
    assert all(sha(p) == digest for p, digest in old_hashes.items())
    summary = {'run_id': run_id, 'authoring_method': 'direct_agent_video_timeline_and_task_review',
               'task': 'blocks_ranking_rgb', 'total': len(records), 'nonempty': len(records),
               'video_agent_reviewed_without_flags': len(reviewed), 'needs_review': len(unresolved),
               'first_active_arm_counts': dict(Counter(r['first_active_arm'] for r in records)),
               'video_object_sequence_counts': dict(Counter(' -> '.join(s['object'] for s in r['video_sequence']) for r in records)),
               'decoded_frames': sum(r['decoded_frames'] for r in records),
               'timeline_pages': sum(len(r['contact_sheets']) for r in records),
               'additional_evidence_pages': sum(len(set(r['evidence_pages_viewed']) - {p['path'] for p in r['contact_sheets']}) for r in records),
               'sampling_interval_s': 1.0, 'every_decoded_frame_visually_inspected': False,
               'source_video_image_instruction_checks_passed': 50,
               'caption_format_rgb_action_and_dependency_checks_passed': 50,
               'old_records_unchanged': True, 'old_record_sha256': old_hashes,
               'previous_source_flags_resolved_by_video': [r['episode'] for r in reviewed if r['previous_first_frame_review_flags']],
               'text_exports': {'text_video_candidates_all.txt': len(records), 'text_video_prompt_enhanced.txt': len(reviewed)},
               'existing_training_manifest_updated': False}
    write_json(run / 'summary.json', summary)
    (run / 'README.txt').write_text(
        'RGB 视频时序重标注：本次按用户要求，以视频和原任务为依据。\n'
        f'全部 50 段完整解码，共 {summary["decoded_frames"]} 帧；逐页查看每 1 秒抽帧及首尾帧，共 {summary["timeline_pages"]} 页。不是逐帧视觉审计。\n'
        'text_video_prompt_enhanced.txt：无未解决标记的 VPP2 格式清单；text_video_candidates_all.txt：全部候选。\n'
        'video_sequence.csv：每个 episode 首动臂、三次操作顺序与粗略时间区间。\n'
        'review.html：视频、首帧、时序证据、原任务、新 prompt 和可导出的审核记录。\n'
        'first_frame_vs_video.csv：上一版首帧 prompt 与此次视频 prompt 对照。\n'
        '英文保持实际右臂在前、while 接实际左臂；after 表达跨臂依赖。句序不等于时间顺序。\n'
        'observer 画面左侧为实际右臂；目标位置在英文中明确以 robot ego view 为参照。\n'
        '完整视频是本次新增标注证据，可补充此前未明确的目的地。若真与任务冲突，保留 review_flags。\n'
        '上一版结果和审核记录保持原样。\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'old_record_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
