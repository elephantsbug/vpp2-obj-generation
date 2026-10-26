#!/usr/bin/env python3
"""Export a frozen, <=128-token chronological object-video prompt revision."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys

from obj_chronological_rules import PREFIX, SOURCE, TOKENIZER, render, normalize

PROJECT = Path('/data2/niurc/vpp2-obj-generation')
RGB = PROJECT / 'data/robotwin_observer_ego_vpp2'
OBJ = PROJECT / 'data/robotwin_observer_ego_obj_full_vpp2'
PAIRED = OBJ / 'paired_no_flags.jsonl'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def key(r): return r['task'], r['episode']
def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-root',type=Path,default=PROJECT/'runs/prompt_obj_chronological_128_v1')
    args=ap.parse_args();run=args.output_root.resolve()
    if run.exists(): raise ValueError('Refusing to overwrite an existing prompt revision: '+str(run))
    source_rows=[json.loads(s) for s in SOURCE.read_text().splitlines()]
    selected=[r for r in source_rows if not r['review_flags']]
    excluded=[r for r in source_rows if r['review_flags']]
    pairs={key(r):r for r in [json.loads(s) for s in PAIRED.read_text().splitlines()]}
    assert len(source_rows)==2500 and len(selected)==2395 and len(excluded)==105
    assert len(pairs)==2395 and set(pairs)=={key(r) for r in selected}
    assert len({r['task'] for r in selected})==50
    sys.path.insert(0,str(PROJECT/'third_party/vpp2/src'))
    from vpp2.models.wan21_14b.wan_video_text_encoder import HuggingfaceTokenizer
    tok=HuggingfaceTokenizer(str(TOKENIZER),seq_len=128,clean='whitespace',local_files_only=True)
    import obj_chronological_rules as rules
    records=[]
    for old in selected:
        body=normalize(render(old));prompt=PREFIX+' '+body
        full=tok(prompt,padding=False,truncation=False,add_special_tokens=True)[0].tolist()
        padded,mask=tok(prompt,return_mask=True,add_special_tokens=True)
        assert padded[0][mask[0].bool()].tolist()==full
        assert len(full)<=120,(key(old),len(full))
        assert '\n' not in prompt and '@@' not in prompt and 'keeps static' not in body
        assert 'This is a composed robot video' not in prompt
        previous=tok(old['enhanced_instruction'],padding=False,truncation=False,add_special_tokens=True)[0].tolist()
        source_record=PROJECT/'runs/prompt_all_video_agent_v1/records'/old['task']/(old['episode']+'.json')
        assert json.loads(source_record.read_text())==old
        pair=pairs[key(old)]
        assert pair['prompt']==old['enhanced_instruction'] and pair['prompt_review_flags']==[]
        assert pair['rgb_sha256']['video']==old['video_sha256']
        assert pair['rgb_sha256']['image']==old['image_sha256']
        records.append({
            'task':old['task'],'episode':old['episode'],'image':old['image'],
            'original_instruction':old['original_instruction'],'enhanced_instruction':prompt,
            'task_prefix':PREFIX,'chronological_body':body,'token_count':len(full),
            'token_count_includes_special_tokens':True,'token_limit':128,'truncated':False,
            'source_enhanced_instruction':old['enhanced_instruction'],'source_token_count':len(previous),
            'source_record_path':str(source_record),'source_record_sha256':sha(source_record),
            'source_video_sequence':old['video_sequence'],'first_active_arms':old['first_active_arms'],
            'arm_participation':old['arm_participation'],'temporal_relation':old['temporal_relation'],
            'source_evidence_pages_viewed':old['evidence_pages_viewed'],
            'image_path':old['image_path'],'video_path':old['video_path'],'video_sha256':old['video_sha256'],
            'image_sha256':old['image_sha256'],'target_video_path':pair['target_video_path'],
            'target_video_sha256':pair['obj_sha256']['video'],
            'review_flags':[],'user_approved':False,
            'authoring_method':'agent-authored chronological render rules populated from reviewed episode actions and object descriptions',
            'review_scope':'Reuses the prior video review and evidence; no new claim of visually inspecting the videos in this rewrite.',
            'static_arm_clause_omitted':not all(old['arm_participation'].values()),
        })
    run.mkdir(parents=True);out=run/'exports';out.mkdir()
    snapshot=run/'tokenizer';snapshot.mkdir()
    for name in ('spiece.model','tokenizer.json','special_tokens_map.json','tokenizer_config.json'):
        shutil.copyfile(TOKENIZER/name,snapshot/name)
        assert sha(snapshot/name)==sha(TOKENIZER/name)
    provenance=run/'provenance';provenance.mkdir()
    shutil.copyfile(rules.__file__,provenance/'obj_chronological_rules.py')
    shutil.copyfile(__file__,provenance/'export_obj_chronological_prompts.py')
    contract={
        'source_records_path':str(SOURCE),'source_records_sha256':sha(SOURCE),
        'source_paired_manifest_path':str(PAIRED),'source_paired_manifest_sha256':sha(PAIRED),
        'prefix':PREFIX,'hard_token_limit':128,'authoring_target_max_tokens':120,
        'token_count_includes_special_tokens':True,'text_cleaning':'VPP2 HuggingfaceTokenizer clean=whitespace',
        'tokenizer_source_path':str(TOKENIZER),'tokenizer_snapshot_path':str(snapshot),
        'tokenizer_class':type(tok.tokenizer).__name__,
        'tokenizer_sha256':{p.name:sha(p) for p in snapshot.iterdir()},
        'rules_sha256':sha(rules.__file__),
        'vpp2_tokenizer_code_sha256':sha(PROJECT/'third_party/vpp2/src/vpp2/models/wan21_14b/wan_video_text_encoder.py'),
        'fixed_arm_order':False,'static_arm_clause_required':False,
        'views':['observer','ego'],'target_domain':'video_object_full','condition_domain':'full-scene RGB first frame and history',
        'text_embedding_cache_must_be_rebuilt':True,
        'training_text_usage':'Use the full enhanced_instruction/prompt verbatim, including PREFIX exactly once; do not append the old view suffix or an extra task wrapper. Never tokenize the @@image path.',
        'tokenizer_usage':'Use this snapshot for new text embeddings and training; revalidate all lengths if the tokenizer is changed.',
    }
    save(run/'prompt_contract.json',contract)
    for name,target in [('images',RGB/'images'),('videos',RGB/'videos'),('obj_videos',OBJ/'videos')]:
        (out/name).symlink_to(target,target_is_directory=True)
    def txt(path,rows,field):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(r[field]+'@@'+r['image']+'\n' for r in rows))
    txt(out/'text_prompt_enhanced.txt',records,'enhanced_instruction')
    txt(out/'text_original.txt',records,'original_instruction')
    txt(out/'text_previous_long.txt',records,'source_enhanced_instruction')
    for task in sorted({r['task'] for r in records}):
        txt(out/'by_task'/task/'text_prompt_enhanced.txt',[r for r in records if r['task']==task],'enhanced_instruction')
    for r in records:save(run/'records'/r['task']/(r['episode']+'.json'),r)
    jsonl(out/'records.jsonl',records);jsonl(out/'excluded_105.jsonl',excluded)
    new_records_hash=sha(out/'records.jsonl')
    new_pairs=[]
    for r in records:
        pair=dict(pairs[key(r)])
        pair['source_prompt_records_path']=pair['prompt_records_path']
        pair['source_prompt_records_sha256']=pair['prompt_records_sha256']
        pair.update({'prompt':r['enhanced_instruction'],'prompt_records_path':str(out/'records.jsonl'),
                     'prompt_records_sha256':new_records_hash,'prompt_token_count':r['token_count'],
                     'prompt_tokenizer_path':str(snapshot),'prompt_max_length':128,
                     'prompt_revision':run.name,'text_embedding_cache_must_be_rebuilt':True})
        new_pairs.append(pair)
    jsonl(out/'paired_training.jsonl',new_pairs)
    fields=['task','episode','source_token_count','token_count','temporal_relation','first_active_arms','source_enhanced_instruction','enhanced_instruction','video_path','target_video_path']
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in records:writer.writerow({k:','.join(r[k]) if isinstance(r[k],list) else r[k] for k in fields})
    with (out/'token_lengths.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['task','episode','previous_tokens','new_tokens','limit','includes_special_tokens','truncated'])
        writer.writerows((r['task'],r['episode'],r['source_token_count'],r['token_count'],128,True,False) for r in records)
    new=[r['token_count'] for r in records];old=[r['source_token_count'] for r in records]
    stats=lambda values:{'min':min(values),'median':statistics.median(values),'mean':round(statistics.mean(values),2),'max':max(values),'over128':sum(n>128 for n in values)}
    summary={'status':'completed','created_at':datetime.now(timezone.utc).isoformat(),
             'run_id':run.name+'_'+new_records_hash[:20],'tasks':50,'episodes':len(records),'excluded':len(excluded),
             'source_tokens':stats(old),'new_tokens':stats(new),'empty':0,'truncated':0,
             'task_counts':dict(Counter(r['task'] for r in records)),
             'validation':'VPP2 128-token padded path exactly matches untruncated tokens for every prompt',
             'new_text_embeddings_generated':False,'training_started':False}
    save(out/'summary.json',summary)
    (out/'README.txt').write_text(
        '时序短 prompt：50 tasks，共2395条；原105条待复核记录排除并保存在excluded_105.jsonl。\n'
        'text_prompt_enhanced.txt：每行完整prompt@@images/task/episode.png。图片是全景RGB首帧。\n'
        'by_task/<task>/text_prompt_enhanced.txt：分任务同格式；图片路径仍相对本exports或RGB数据集根。\n'
        'paired_training.jsonl：使用新prompt的全景RGB条件/obj_full目标配对，2395条；训练应读取此版本。\n'
        'comparison.csv：旧长prompt与新时序prompt对照；token_lengths.csv逐条记录token数。\n'
        'records.jsonl与../records保留源动作顺序、证据引用、原长prompt和源文件哈希。\n'
        '全部完整文本（含固定前缀、特殊token）不超过120 token，128-token路径没有截断。\n'
        '分词器固定快照在../tokenizer；后续text embedding和训练应使用同一快照，换分词器须重验。\n'
        '实际输入直接使用完整prompt，不再额外包长模板或旧视角后缀，不把@@图片路径送入分词器。\n'
        '固定前缀说明RGB首帧/历史 -> 黑背景obj未来视频；正文按时序，并保留实际同步动作。\n'
        '取消固定右臂优先、强制while连接和静止臂句；保留物体、工具、抓握/操作、目的地和终态。\n'
        '本次重写复用已有视频审阅记录，没有声称重新逐视频目视检查。\n'
        '原prompt、原配对索引和视频未覆盖；新的文本embedding尚未计算，不能复用旧prompt缓存。\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
