#!/usr/bin/env python3
"""Check per-task agent annotations and export video-grounded VPP2 captions."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

PROJECT=Path('/data2/niurc/vpp2-obj-generation')
RUN=PROJECT/'runs/prompt_all_video_agent_v1'
DATA=PROJECT/'data/robotwin_observer_ego_vpp2'
sys.path.insert(0,str(PROJECT/'scripts'))
from enhance_prompts import check_caption,SUFFIX
from build_prompt_review_queue import CSV_FIELDS

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic(path, text):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(text);tmp.replace(path)

def save(path, value):atomic(path,json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def write_csv(path, items):
    # This run's flags come from direct video review, not the earlier VLM checks.
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['视频审阅标记' if key=='review_flags' else title for key,title in CSV_FIELDS])
        for item in items:
            writer.writerow([' | '.join(item[key]) if isinstance(item[key],list) else item[key] for key,_ in CSV_FIELDS])

def validate_task(task):
    inputs=json.loads((RUN/'task_inputs'/f'{task}.json').read_text())
    source={r['episode']:r for r in inputs}
    draft_path=RUN/'task_drafts'/f'{task}.json'
    drafts=json.loads(draft_path.read_text())
    assert len(drafts)==50 and len({r['episode'] for r in drafts})==50, 'expected 50 unique episodes'
    assert {r['episode'] for r in drafts}==set(source),'episode set differs'
    records=[]
    for d in sorted(drafts,key=lambda r:r['episode']):
        s=source[d['episode']];ep=d['episode']
        try:
            for k in ('task','episode','image','original_instruction'):assert d[k]==s[k],k+' changed'
            caption,errors=check_caption(d['enhanced_instruction'])
            assert not errors and caption==d['enhanced_instruction'],str(errors)
            assert isinstance(d['review_flags'],list) and all(isinstance(x,str) for x in d['review_flags'])
            assert isinstance(d['notes'],str)
            assert d['temporal_relation'] in ('single_arm','sequential','simultaneous','mixed')
            participation=d['arm_participation']
            assert set(participation)=={'right','left'} and all(type(v) is bool for v in participation.values())
            active={a for a,v in participation.items() if v}
            first=d['first_active_arms']
            assert isinstance(first,list) and len(set(first))==len(first) and set(first)<=active
            assert first or d['review_flags'],'no observed first arm and no uncertainty flag'
            body=caption.removesuffix('. '+SUFFIX)
            right,left=body.split(', while the left robot gripper ')
            phrases={'right':right.removeprefix('The right robot gripper '),'left':left}
            for arm in participation:
                if not participation[arm]:assert phrases[arm]=='keeps static',arm+' inactive clause not exact'
                else:assert not phrases[arm].startswith('keeps static'),arm+' active declared static'
            sequence=d['video_sequence'];assert isinstance(sequence,list) and sequence
            assert [x['order'] for x in sequence]==list(range(1,len(sequence)+1))
            used=set()
            for step in sequence:
                assert isinstance(step['arms'],list) and step['arms'] and set(step['arms'])<=active
                used.update(step['arms'])
                assert isinstance(step['object'],str) and step['object'].strip()
                assert isinstance(step['action'],str) and step['action'].strip()
                lo,hi=step['approx_time_s']
                assert math.isfinite(lo) and math.isfinite(hi) and 0<=lo<=hi<=s['duration_s']+1.01
                frames=step['evidence_frames']
                assert frames and all(type(i) is int and 0<=i<s['decoded_frames'] for i in frames)
            assert used==active,'active arms differ from evidence action arms'
            assert len(active)==1 if d['temporal_relation']=='single_arm' else True
            viewed=d['evidence_pages_viewed'];assert viewed and len(viewed)==len(set(viewed))
            assert all(Path(p).is_file() for p in viewed)
            if d.get('reused_review_record_path'):
                prior=Path(d['reused_review_record_path']);assert sha(prior)==d['reused_review_record_sha256']
                old=json.loads(prior.read_text())
                assert old['enhanced_instruction']==caption
                assert old['video_sha256']==s['video_sha256']==d['reviewed_video_sha256']
                assert old['image_sha256']==s['image_sha256']==d['reviewed_image_sha256']
                assert old['evidence_pages_viewed']==viewed
            else:
                assert {p['path'] for p in s['contact_sheets']}<=set(viewed),'not all timeline pages declared viewed'
            for k in ('video_sha256','image_sha256','video_path','image_path','instruction_path','duration_s','decoded_frames'):
                if k in d:assert d[k]==s[k],'draft changed source field '+k
            assert sha(s['video_path'])==s['video_sha256'],'video hash changed'
            assert sha(s['image_path'])==s['image_sha256'],'image hash changed'
            assert json.loads(Path(s['instruction_path']).read_text())['seen'][0]==s['original_instruction']
            records.append({**s,**d,'authoring_method':'direct_agent_video_and_task_review',
                'instruction_sha256':sha(s['instruction_path']),
                'draft_file_sha256':sha(draft_path),
                'review_status':'needs_review' if d['review_flags'] else 'agent_video_reviewed',
                'user_approved':False,
                'review_scope':'Entire video decoded. Agents reviewed timestamped samples over the full clip, normally every 1 second including endpoints; extra evidence is recorded where used. Not every decoded frame was visually inspected.',
                'structural_validation':'passed; does not by itself prove semantic correctness'})
        except Exception as e:
            raise ValueError(f'{task}/{ep}: {e}') from e
    return records

def export(records, all_tasks):
    assert len(records)==2500 and len({(r['task'],r['episode']) for r in records})==2500
    out=RUN/'exports'
    assert not (out/'review.html').exists(),'Existing exported review page must not be overwritten'
    out.mkdir(exist_ok=True)
    for name in ('images','videos'):(out/name).symlink_to(DATA/name,target_is_directory=True)
    (out/'evidence').symlink_to(RUN/'evidence',target_is_directory=True)
    (out/'rgb_prior_evidence').symlink_to(PROJECT/'runs/prompt_rgb_video_agent_v1/evidence',target_is_directory=True)
    (out/'records').symlink_to(RUN/'records',target_is_directory=True)
    def evidence_url(path):
        p=Path(path)
        if p.is_relative_to(RUN/'evidence'):return 'evidence/'+str(p.relative_to(RUN/'evidence'))
        prior=PROJECT/'runs/prompt_rgb_video_agent_v1/evidence'
        if p.is_relative_to(prior):return 'rgb_prior_evidence/'+str(p.relative_to(prior))
        raise ValueError('Additional evidence must be saved inside the run: '+str(path))
    def lines(rows,key):return ''.join(r[key]+'@@'+r['image']+'\n' for r in rows)
    approved=[r for r in records if not r['review_flags']]
    review=[r for r in records if r['review_flags']]
    atomic(out/'text_video_candidates_all.txt',lines(records,'enhanced_instruction'))
    atomic(out/'text_video_prompt_enhanced.txt',lines(approved,'enhanced_instruction'))
    atomic(out/'text_original.txt',lines(records,'original_instruction'))
    for t in all_tasks:
        selected=[r for r in records if r['task']==t]
        atomic(out/'by_task'/t/'text_video_candidates_all.txt',lines(selected,'enhanced_instruction'))
        atomic(out/'by_task'/t/'text_video_prompt_enhanced.txt',lines([r for r in selected if not r['review_flags']],'enhanced_instruction'))
        atomic(out/'by_task'/t/'text_original.txt',lines(selected,'original_instruction'))
    for name,rows in [('records.jsonl',records),('needs_review.jsonl',review)]:atomic(out/name,''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    items=[]
    for r in records:
        key=r['task']+'/'+r['episode']
        sequence=' → '.join(f"{','.join(s['arms'])}: {s['object']} — {s['action']} (~{s['approx_time_s'][0]}–{s['approx_time_s'][1]}s)" for s in r['video_sequence'])
        item={'id':key,'task':r['task'],'episode':r['episode'],'priority':'P1' if r['review_flags'] else 'P3',
              'priority_reason':'请核对记录中的不确定性或任务与视频差异。' if r['review_flags'] else 'Agent 已依据视频检查用臂、物体、方法、时序和终态。',
              'issue_labels':['待核对'] if r['review_flags'] else ['视频已审阅'],'review_flags':r['review_flags'],
              'original_instruction':r['original_instruction'],'enhanced_instruction':r['enhanced_instruction'],
              'image_path':r['image_path'],'video_path':r['video_path'],'record_path':str(RUN/'records'/r['task']/(r['episode']+'.json')),
              'observer_video_path':r['original_observer_video_path'],'ego_video_path':r['original_ego_video_path'],
              'image_url':r['image'],'video_url':'videos/'+key+'.mp4','record_url':'records/'+key+'.json',
              'has_caption':True,'decision':'待复核','corrected_prompt':'','notes':r['notes'],
              'sequence_text':sequence,'temporal_relation':r['temporal_relation'],
              'first_active_arms':r['first_active_arms'],'evidence_urls':[evidence_url(p) for p in r['evidence_pages_viewed']]}
        items.append(item)
    write_csv(out/'review_all.csv',items);write_csv(out/'needs_review.csv',[i for i in items if i['review_flags']])
    with (out/'video_sequence.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['task','episode','first_active_arms','temporal_relation','observed_sequence','review_flags','video_path'])
        for i in items:w.writerow([i['task'],i['episode'],','.join(i['first_active_arms']),i['temporal_relation'],i['sequence_text'],' | '.join(i['review_flags']),i['video_path']])
    run_id='all_video_agent_v1_'+sha(out/'records.jsonl')[:20]
    payload={'run_id':run_id,'generated_at':datetime.now(timezone.utc).isoformat(),'counts':{'nonempty_review':len(items),'empty':0},'items':items}
    save(out/'review_items.json',payload)
    template=(PROJECT/'scripts/review_template.html').read_text()
    template=template.replace('VPP2 · 视频 Prompt 复核','VPP2 · 50 个任务的视频 Prompt')
    template=template.replace('生成时使用的首帧','视频首帧')
    template=template.replace('自动检查的提示不等于错误结论。','视频审阅标记包含不确定性与任务差异，需逐条核对。')
    template=template.replace('候选扩写 Prompt','依据视频编写的 Prompt')
    template=template.replace('查看自动检查的原始提示','查看视频审阅中的待确认项')
    template=template.replace('视频用于核验执行情况；新增描述仍须由首帧和原任务支持，不能把后续画面透露的额外细节写入扩写。信息冲突或无法确认时选“暂缓”并记原因。','本次授权依据视频编写：按视频核对先后、同时、工具和终态。任务与视频冲突时保留标记；时间为抽帧观察的粗略区间。')
    anchor='<div class="panel"><h3>原任务指令</h3>'
    panel='<div class="panel"><h3>视频动作记录</h3><p id="observed-first"></p><p id="observed-sequence" class="textblock"></p><p>时间为近似观察区间；多个步骤可能重叠，请结合“同时/顺序”关系阅读。</p><div id="timeline-links" class="media-links"></div></div>\n'
    assert anchor in template;template=template.replace(anchor,panel+anchor)
    anchor="$('original').textContent = str(item.original_instruction);"
    js="""
      $('observed-first').textContent = '先参与任务的实际机械臂：' + array(item.first_active_arms).join(', ') + '；关系：' + str(item.temporal_relation);
      $('observed-sequence').textContent = str(item.sequence_text);
      $('timeline-links').replaceChildren(...array(item.evidence_urls).map((url,index)=>{
        const a=document.createElement('a'),safe=safeRelativeURL(url);
        if(safe)a.href=safe;a.target='_blank';a.rel='noopener';a.textContent='视频证据 '+(index+1);return a;
      }));
"""
    assert anchor in template;template=template.replace(anchor,anchor+js)
    safe=json.dumps(payload,ensure_ascii=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    atomic(out/'review.html',template.replace('__REVIEW_DATA_JSON__',safe))
    summary={'run_id':run_id,'status':'completed','tasks':len(all_tasks),'episodes':len(records),
        'agent_reviewed_without_unresolved_flags':len(approved),'needs_review':len(review),'empty':0,
        'task_counts':{t:{'total':sum(r['task']==t for r in records),'needs_review':sum(r['task']==t and bool(r['review_flags']) for r in records)} for t in all_tasks},
        'temporal_relation_counts':dict(Counter(r['temporal_relation'] for r in records)),
        'decoded_frames':sum(r['decoded_frames'] for r in records),'duration_hours':sum(r['duration_s'] for r in records)/3600,
        'reused_verified_rgb':50,'every_frame_visually_inspected':False,'existing_training_manifest_updated':False,
        'text_files':{'text_video_candidates_all.txt':len(records),'text_video_prompt_enhanced.txt':len(approved)}}
    save(out/'summary.json',summary)
    atomic(out/'README.txt',f'全量视频 prompt：50 tasks × 50 episodes，共2500条。\n全部候选：text_video_candidates_all.txt。排除未解决标记：text_video_prompt_enhanced.txt。\n每行严格 prompt@@images/task/episode.png；相对路径以本exports目录或原转换数据集为根，不能以by_task子目录为根。\nby_task/<task>/ 按任务保存同格式文本；图片路径仍相对全量数据集根。\nrecords.jsonl与records/保存逐条动作记录、近似时间、证据、来源哈希。\nvideo_sequence.csv保存动作顺序/同时关系。review.html提供视频、首帧、证据与可导出的人工审核记录。\n本次逐视频查看覆盖全时段的每秒抽帧及首尾，必要时补充证据；不是逐帧视觉审计。RGB 50条复用先前同视频哈希的已完成审阅。\n无未解决标记{len(approved)}条，待复核{len(review)}条。两种情况均保留完整候选，不隐去冲突。\n')
    return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task');ap.add_argument('--finalize',action='store_true');args=ap.parse_args()
    if (RUN/'exports'/'review.html').exists():
        ap.error('This exported run is frozen. Use a new run directory for changes; do not rewrite linked records after export.')
    all_tasks=sorted(p.stem for p in (RUN/'task_inputs').glob('*.json'))
    tasks=[args.task] if args.task else all_tasks
    checked={};records=[]
    for t in tasks:
        path=RUN/'task_drafts'/f'{t}.json'
        if not path.exists():checked[t]={'status':'pending'};continue
        try:
            rows=validate_task(t)
            for row in rows:save(RUN/'records'/t/(row['episode']+'.json'),row)
            records.extend(rows)
            checked[t]={'status':'validated','episodes':len(rows),'review_flags':sum(bool(r['review_flags']) for r in rows),'draft_sha256':sha(path)}
        except Exception as e:checked[t]={'status':'validation_error','error':str(e)}
    status={'updated_at':datetime.now(timezone.utc).isoformat(),'validated_tasks':sum(x['status']=='validated' for x in checked.values()),'validated_episodes':len(records),'tasks':checked}
    save(RUN/'progress'/('checks_'+args.task+'.json' if args.task else 'checks.json'),status)
    if args.finalize:
        assert not args.task
        assert len(all_tasks)==50 and len(records)==2500,str(status)
        print(json.dumps(export(records,all_tasks),ensure_ascii=False,indent=2))
    else:print(json.dumps(status,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
