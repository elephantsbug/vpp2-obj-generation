#!/usr/bin/env python3
"""Build a separate human review package without changing generation records."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote


PROJECT = Path('/data2/niurc/vpp2-obj-generation')
COMPLEX_TASKS = {
    'blocks_ranking_rgb', 'blocks_ranking_size', 'stack_blocks_three',
    'stack_blocks_two', 'stack_bowls_three', 'stack_bowls_two',
    'handover_block', 'handover_mic', 'pick_dual_bottles', 'place_dual_shoes',
    'scan_object', 'put_object_cabinet', 'place_cans_plasticbox',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(record):
    labels = set()
    flags = record['review_flags']
    for flag in flags:
        if 'proximity heuristic is unresolved' in flag:
            labels.add('选臂依据不足')
        if 'proximity heuristic conflicts' in flag:
            labels.add('选臂与位置规则冲突')
        if 'requirements.destinations' in flag and 'required text omitted' in flag:
            labels.add('目的地表述待核对')
        if 'requirements.final_states' in flag and 'required text omitted' in flag:
            labels.add('最终状态表述待核对')
        if '.evidence:' in flag:
            labels.add('任务引用格式')
        if 'grasp_parts' in flag:
            labels.add('抓取部位待核对')
        if 'simultaneous' in flag:
            labels.add('同步或时序待核对')
        if 'cross_arm_tool_use' in flag:
            labels.add('工具操作归属冲突')
        if 'other arm; extra participation' in flag:
            labels.add('可能添加多余机械臂动作')
        if 'omits required lifting action' in flag:
            labels.add('可能遗漏抬起动作')
        if flag.startswith('grounding.') or flag.startswith('pipeline_error:'):
            labels.add('定位或字段格式问题')
        if '.actions[' in flag and ('invalid action kind' in flag or 'unknown object id' in flag or 'only transfer/receive' in flag):
            labels.add('动作计划结构待核对')
        if '.uncertainties:' in flag:
            labels.add('模型或校验反馈中的不确定项')
    if not record['enhanced_instruction']:
        labels.add('空文本（暂缓）')
        return 'P4', '按当前决定暂缓；定位或格式问题，不作为本轮人工写作任务', sorted(labels)
    if record['task'] in COMPLEX_TASKS:
        labels.add('多物体或多阶段任务')
    if not labels:
        labels.add('其他自动检查问题')
    action_labels = {'选臂与位置规则冲突', '工具操作归属冲突', '可能添加多余机械臂动作',
                     '可能遗漏抬起动作', '同步或时序待核对', '抓取部位待核对',
                     '动作计划结构待核对', '多物体或多阶段任务'}
    if labels & action_labels:
        return 'P1', '先查参与机械臂、物体是否齐全、工具、操作和时间顺序；优先级不代表已确认错误', sorted(labels)
    if labels & {'选臂依据不足', '定位或字段格式问题', '模型或校验反馈中的不确定项'}:
        return 'P2', '结合首帧及整段视频确认物体与机械臂；居中不代表必须任选一臂', sorted(labels)
    return 'P3', '核对目的地、最终状态与同义改写；仍需完整检查任务，不能批量视作误报', sorted(labels)


CSV_FIELDS = [
    ('id', '条目ID'), ('priority', '复核优先级'), ('priority_reason', '复核重点'),
    ('task', '任务'), ('episode', 'episode'), ('issue_labels', '问题分类'),
    ('video_path', '拼接视频绝对路径'), ('image_path', '首帧图片绝对路径'),
    ('observer_video_path', '原始observer视频路径'), ('ego_video_path', '原始ego视频路径'),
    ('original_instruction', '原始任务指令'), ('enhanced_instruction', '当前候选prompt'),
    ('review_flags', '自动检查原始标记'), ('record_path', '生成记录JSON路径'),
    ('decision', '审核结论'), ('corrected_prompt', '修改后prompt（仅正文）'), ('notes', '审核备注'),
]


def write_csv(path, items):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([title for _, title in CSV_FIELDS])
        for item in items:
            writer.writerow([' | '.join(item[key]) if isinstance(item[key], list) else item[key]
                             for key, _ in CSV_FIELDS])


def write_workbook(path, active, empty, summary):
    """Provide filters and editable decision columns without macros or formulas."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    workbook.remove(workbook.active)
    order = ['priority', 'task', 'episode', 'issue_labels', 'decision', 'corrected_prompt',
             'notes', 'original_instruction', 'enhanced_instruction', 'video_path', 'image_path',
             'observer_video_path', 'ego_video_path', 'review_flags', 'record_path', 'id', 'priority_reason']
    titles = dict(CSV_FIELDS)
    widths = [12, 26, 22, 36, 18, 65, 40, 65, 75, 55, 55, 55, 55, 65, 55, 45, 65]
    for title, rows in [(f'待复核{len(active)}条', active), (f'空文本暂缓{len(empty)}条', empty)]:
        sheet = workbook.create_sheet(title)
        sheet.append([titles[key] for key in order])
        for row in rows:
            sheet.append([' | '.join(row[key]) if isinstance(row[key], list) else row[key] for key in order])
        sheet.freeze_panes = 'D2'
        sheet.auto_filter.ref = sheet.dimensions
        for index, width in enumerate(widths, 1):
            sheet.column_dimensions[get_column_letter(index)].width = width
        for cell in sheet[1]:
            cell.fill = PatternFill('solid', fgColor='173B57')
            cell.font = Font(color='FFFFFF', bold=True)
            cell.alignment = Alignment(vertical='top', wrap_text=True)
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical='top', wrap_text=True)
                # Text may originate from a model; never let it become an Excel formula.
                if isinstance(cell.value, str):
                    cell.data_type = 's'
            for cell in row[4:7]:
                cell.fill = PatternFill('solid', fgColor='FFF3CC')
            sheet.row_dimensions[row[0].row].height = 70
        dropdown = DataValidation(type='list', formula1='"待复核,通过,修改后通过,排除,暂缓"', allow_blank=False)
        dropdown.errorTitle = '请选择审核结论'
        dropdown.error = '使用待复核、通过、修改后通过、排除或暂缓。'
        dropdown.showErrorMessage = True
        sheet.add_data_validation(dropdown)
        if rows:
            dropdown.add(f'E2:E{len(rows)+1}')
    stats = workbook.create_sheet('任务统计')
    stats.append(['任务', '原始episode数', '自动检查通过', '非空待复核', '空文本暂缓'])
    for task, counts in sorted(summary['tasks'].items()):
        stats.append([task, counts['selected'], counts['automatic_checks_passed'],
                      counts['needs_review_by_checks']-counts['empty_caption'], counts['empty_caption']])
    stats.freeze_panes = 'A2'
    stats.auto_filter.ref = stats.dimensions
    stats.column_dimensions['A'].width = 30
    for col in 'BCDE':
        stats.column_dimensions[col].width = 20
    guide = workbook.create_sheet('复核方法')
    for line in GUIDE.splitlines():
        guide.append([line])
    guide.column_dimensions['A'].width = 110
    workbook.save(path)


GUIDE = '''人工复核说明

本清单针对本轮856条非空待复核prompt。另14条空文本单独列出并默认暂缓。
自动标记是检查线索，不是错误结论；只看有标记的字段也可能漏掉其他错误。

打开 review.html：可按任务、问题分类、优先级和审核状态筛选；每次只加载当前条视频。
若本机无法访问服务器路径，需要在有此目录访问权限的浏览器中打开，或通过SSH转发本地HTTP服务。
CSV可用Excel/WPS打开，包含原视频/拼接视频绝对路径，可复制路径到播放器。
review_checklist.xlsx提供相同清单、筛选器、冻结表头和审核结论下拉选项；黄色列可填写。

推荐每条的复核顺序：
1. 先读原始任务，把工具、物体、指定机械臂、操作、目的地、最终状态列清楚。
2. 看首帧确认物体。左面板observer中，画面左侧机械臂=实际右臂，右侧=实际左臂。
   右面板ego只是同一场景的另一视角，不是额外机械臂。
   机械臂身份固定，不能在交叉移动后根据当时左右位置重新命名。
3. 播放整段视频；必要时用0.5倍速。核对谁抓取/用工具、谁接收、全过程谁参与。
   keeps static只用于全过程无任务动作的臂；接收后放置的臂不是静止臂。
   如任务文本和实际视频冲突，备注“任务-视频冲突”，暂缓或排除，不擅自二选一改写。
4. 核对完整动作：多物体无遗漏，交接传递整个物体，工具由持工具臂操作，
   同时拿起与先后拿起区分，目的地和最终朝向/状态完整，不添加无依据辅助动作。
5. 最后检查文字格式：实际right arm在前、实际left arm在后，用“, while”连接；
   静止臂用“keeps static”；保留observer左/ego右的固定结尾。
   修正prompt栏只写单行英文正文，暂不加@@路径；原记录保持不变。

视频用于验证动作和数据一致性。任务描述与首帧之外、只在未来帧出现的独有细节，
不要直接加进本轮“首帧+原始任务”的prompt；确有需要请备注并另行决定标注规范。
不能为了消除标记而无条件相信模型物体坐标，也不能把right of等物体关系随镜像翻转。

审核结论：
通过：原候选完整准确，无需修改；若是同义改写误报，在备注写出判断依据。
修改后通过：必须填写完整修正版，检查全部任务要素，不能只改触发检查的单词。
排除：任务-视频冲突、演示失败或无法可靠修正，填写原因。
暂缓：证据不足，或14条空文本暂不处理。
待复核：尚未完成完整检查。

保存：网页可能使用浏览器本地存储暂存；同一浏览器、同一路径才能方便恢复。
无论是否启用本地存储，每批结束都点击“导出审核JSON”备份，另可导出CSV。
跨浏览器/设备用“导入审核JSON”恢复。浏览器中的修改不会写回原始记录或训练清单。
CSV和网页审核记录各自独立，不会自动互相同步；建议选择一种作为主要审核方式。
Excel表中的“修改后通过”需要人工检查修正版是否完整，表格不会验证英文语义。

优先级：P1动作/多阶段/物体完整性，P2选臂与定位，P3文本及引用核对，P4空文本暂缓。
优先级只是排序，不是准确率或错误严重程度的人工判定。无需逐条重看所有原始视频：
先看拼接视频，某一面板细节不足时再打开清单中的原始320x240单视角视频。
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=PROJECT/'runs/prompt_two_step_full_v1')
    parser.add_argument('--dataset-root', type=Path, default=PROJECT/'data/robotwin_observer_ego_vpp2')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--template', type=Path, default=PROJECT/'scripts/review_template.html')
    args = parser.parse_args()
    run, dataset = args.run_root.resolve(), args.dataset_root.resolve()
    output = (args.output_root or run/'manual_review_v1').resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Review package already exists; refusing to overwrite possible human annotations')
    if output == run or output == dataset or dataset.is_relative_to(output) or output.is_relative_to(dataset):
        raise ValueError('Use a separate output directory')
    source = run/'needs_review.jsonl'
    source_hash = sha(source)
    records = [json.loads(line) for line in source.read_text().splitlines() if line]
    summary = json.loads((run/'summary.json').read_text())
    if len(records) != summary['needs_review_by_checks']:
        raise ValueError('Review count differs from run summary')
    manifests = {(r['task'], r['episode']): r for r in
                 (json.loads(line) for line in (dataset/'manifest.jsonl').read_text().splitlines())}
    items, record_hashes, ids = [], {}, set()
    for record in records:
        task, episode = record['task'], record['episode']
        key = f'{task}/{episode}'
        if key in ids:
            raise ValueError('Duplicate task/episode')
        ids.add(key)
        manifest = manifests[(task, episode)]
        record_path = run/'records'/task/(episode+'.json')
        if json.loads(record_path.read_text()) != record:
            raise ValueError(f'Aggregate record differs from original: {key}')
        record_hashes[key] = sha(record_path)
        paths = {'video_path': dataset/manifest['outputs']['video'],
                 'image_path': dataset/record['image'], 'record_path': record_path,
                 'observer_video_path': Path(manifest['inputs']['observer_video']['path']),
                 'ego_video_path': Path(manifest['inputs']['ego_video']['path'])}
        if any(not p.is_file() for p in paths.values()):
            raise ValueError(f'Missing media/record path for {key}')
        priority, reason, labels = classify(record)
        has_caption = bool(record['enhanced_instruction'])
        items.append({'id': key, 'task': task, 'episode': episode, 'priority': priority,
                      'priority_reason': reason, 'issue_labels': labels, 'review_flags': record['review_flags'],
                      'original_instruction': record['original_instruction'],
                      'enhanced_instruction': record['enhanced_instruction'],
                      **{k:str(v) for k,v in paths.items()},
                      'video_url': quote(f'media_videos/{task}/{episode}.mp4'),
                      'image_url': quote(f'media_images/{task}/{episode}.png'),
                      'record_url': quote(os.path.relpath(record_path,output)),
                      'has_caption': has_caption, 'decision': '待复核' if has_caption else '暂缓',
                      'corrected_prompt': '', 'notes': '' if has_caption else '空文本：按当前决定暂缓。'})
    items.sort(key=lambda r:(r['priority'], r['task'], r['episode']))
    active = [r for r in items if r['has_caption']]
    empty = [r for r in items if not r['has_caption']]
    if len(empty) != summary['empty_caption']:
        raise ValueError('Empty-caption count differs from run summary')
    template = args.template.read_text()
    if template.count('__REVIEW_DATA_JSON__') != 1:
        raise ValueError('Expected one HTML data placeholder')
    payload = {'run_id': summary['config_fingerprint'], 'generated_at': datetime.now(timezone.utc).isoformat(),
               'counts': {'nonempty_review':len(active),'empty':len(empty)}, 'items': items}
    output.mkdir(parents=True, exist_ok=True)
    (output/'media_videos').symlink_to(dataset/'videos', target_is_directory=True)
    (output/'media_images').symlink_to(dataset/'images', target_is_directory=True)
    for name, rows in [('review_queue.csv',active), ('empty_prompts.csv',empty), ('review_all.csv',items)]:
        write_csv(output/name, rows)
    write_workbook(output/'review_checklist.xlsx',active,empty,summary)
    (output/'video_paths.txt').write_text(''.join(r['video_path']+'\n' for r in active))
    (output/'review_items.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    safe_json = json.dumps(payload,ensure_ascii=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    (output/'review.html').write_text(template.replace('__REVIEW_DATA_JSON__',safe_json))
    (output/'REVIEW_GUIDE.txt').write_text(GUIDE)
    with (output/'task_summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['任务','原始episode数','自动检查通过','非空待复核','空文本暂缓'])
        for task, counts in sorted(summary['tasks'].items()):
            writer.writerow([task,counts['selected'],counts['automatic_checks_passed'],
                             counts['needs_review_by_checks']-counts['empty_caption'],counts['empty_caption']])
    if sha(source) != source_hash or any(sha(run/'records'/Path(key+'.json')) != digest for key,digest in record_hashes.items()):
        raise ValueError('Source review records changed during export')
    checks={'generated_at':payload['generated_at'], 'run_id':payload['run_id'],
            'nonempty_review':len(active),'empty_deferred':len(empty),'all_records':len(items),
            'tasks':len({r['task'] for r in active}),'priority_counts':dict(Counter(r['priority'] for r in active)),
            'issue_counts_nonexclusive':dict(Counter(label for r in active for label in r['issue_labels'])),
            'all_media_paths_exist':True,'unique_ids':True,'source_records_unchanged':True,
            'source_needs_review_sha256':source_hash,'human_decisions_applied_to_training':False}
    (output/'manifest.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
