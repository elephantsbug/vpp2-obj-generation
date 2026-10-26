"""Independent checks of exported chronological captions and episode pairing."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from itertools import groupby
import json
from pathlib import Path
import re
import sys

PROJECT=Path('/data2/niurc/vpp2-obj-generation')
RUN=PROJECT/'runs/prompt_obj_chronological_128_v1'
OUT=RUN/'exports'

def read(path):return [json.loads(line) for line in path.read_text().splitlines()]
def key(r):return r['task'],r['episode']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

contract=json.loads((RUN/'prompt_contract.json').read_text())
source_rows=read(Path(contract['source_records_path']))
source={key(r):r for r in source_rows}
selected={key(r) for r in source_rows if not r['review_flags']}
excluded={key(r) for r in source_rows if r['review_flags']}
assert sha(contract['source_records_path'])==contract['source_records_sha256']
assert sha(contract['source_paired_manifest_path'])==contract['source_paired_manifest_sha256']
records=read(OUT/'records.jsonl');pairs=read(OUT/'paired_training.jsonl')
assert len(records)==len(pairs)==len(selected)==2395
assert len({key(r) for r in records})==2395
assert {key(r) for r in records}=={key(r) for r in pairs}==selected
assert {key(r) for r in read(OUT/'excluded_105.jsonl')}==excluded
assert read(OUT/'excluded_105.jsonl')==[r for r in source_rows if r['review_flags']]
assert len(excluded)==105 and not selected.intersection(excluded)
old_pairs={key(r):r for r in read(Path(contract['source_paired_manifest_path']))}
by_key={key(r):r for r in records}
lengths=[];semantic_checks=Counter()
for r in records:
    old=source[key(r)];body=r['chronological_body'];prompt=r['enhanced_instruction'];steps=old['video_sequence']
    assert prompt==contract['prefix']+' '+body
    assert prompt.count(contract['prefix'])==1 and '@@' not in prompt and '\n' not in prompt
    assert 'keeps static' not in body and 'This is a composed robot video' not in prompt
    assert r['source_enhanced_instruction']==old['enhanced_instruction']
    assert r['source_video_sequence']==steps
    assert r['first_active_arms']==old['first_active_arms']
    assert r['arm_participation']==old['arm_participation']
    assert sha(r['source_record_path'])==r['source_record_sha256']
    assert json.loads((RUN/'records'/r['task']/(r['episode']+'.json')).read_text())==r
    assert r['review_flags']==[] and r['user_approved'] is False
    for field in ['image_path','video_path','target_video_path']:
        assert Path(r[field]).is_file()
    assert (OUT/r['image']).resolve()==Path(r['image_path']).resolve()
    if len(old['first_active_arms'])==2:
        assert 'simultaneously' in body.lower(),key(r)
        semantic_checks['initial_parallel_action_preserved']+=1
    if old['temporal_relation']=='single_arm':
        first=re.search(r'\b(left|right) gripper\b',body,re.I)
        assert first and old['arm_participation'][first.group(1).lower()],key(r)
        semantic_checks['single_active_arm_correct']+=1
    t=r['task']
    if t in ('blocks_ranking_rgb','blocks_ranking_size','stack_blocks_two','stack_blocks_three','stack_bowls_two','stack_bowls_three'):
        observed=re.findall(r'\b(left|right) gripper\b',body,re.I)
        source_steps=steps[::2] if t in ('stack_blocks_three','stack_bowls_two') else steps
        assert [x.lower() for x in observed]==[s['arms'][0] for s in source_steps],(key(r),observed)
        semantic_checks['ordered_block_bowl_arm_assignments']+=1
    if t in ('blocks_ranking_rgb','stack_blocks_three'):
        assert body.lower().find('red')<body.lower().find('green')<body.lower().find('blue')
        semantic_checks['all_three_block_colors_and_order']+=1
    if t=='blocks_ranking_size':
        assert body.find('small')<body.find('medium')<body.find('large')
        assert 'large-medium-small row' in body and 'ego view' in body
        semantic_checks['block_size_order_and_destination']+=1
    if t=='handover_mic':
        giver=steps[0]['arms'][0];receiver='right' if giver=='left' else 'left'
        assert body.startswith('The '+giver+' gripper')
        assert f'{receiver.capitalize()} grips the exposed handle; then {giver} releases' in body
        assert f'{receiver} holding the microphone upright' in body
        semantic_checks['microphone_handover_and_final_holder']+=1
    if t=='put_bottles_dustbin':
        groups=[(o,list(g)) for o,g in groupby(steps,key=lambda s:s['object'])]
        names=['clear water bottle' if 'water bottle' in o else 'black soda bottle' if 'soda bottle' in o else 'yellow bottle' for o,_ in groups]
        offsets=[body.find(n) for n in names]
        assert all(i>=0 for i in offsets) and offsets==sorted(offsets)
        directs=[g[0]['arms']==['left'] for _,g in groups]
        if not all(directs):
            assert 'upper body' in body and 'lower body' in body and 'right releases' in body
        if any(directs) and not all(directs):
            for n,d in zip(names,directs):
                assert (f'left lifts the {n} by its body and drops it' if d else f'right passes the {n} to left to drop') in body
        assert 'white-lined black bin at ego-left' in body and 'all three' in body.lower()
        semantic_checks['three_bottle_order_handover_and_bin']+=1
    if t=='place_burger_fries':
        tipped='tips onto its side' in steps[-1]['action']
        assert tipped==('The fries carton tips onto its side.' in body)
        semantic_checks['fries_terminal_tipping_preserved']+=1

records_hash=sha(OUT/'records.jsonl')
for pair in pairs:
    r=by_key[key(pair)];old=old_pairs[key(pair)]
    assert pair['prompt']==r['enhanced_instruction'] and pair['prompt_token_count']==r['token_count']
    assert pair['prompt_records_sha256']==records_hash
    for field in ['condition_video_path','condition_episode_first_frame_path','target_video_path','target_first_frame_path','timestamps_path','rgb_sha256','obj_sha256']:
        assert pair[field]==old[field],(key(pair),field)
    assert pair['text_embedding_cache_must_be_rebuilt'] is True

for folder,rs in [(OUT,records)]+[(OUT/'by_task'/t,[r for r in records if r['task']==t]) for t in sorted({r['task'] for r in records})]:
    lines=(folder/'text_prompt_enhanced.txt').read_text().splitlines()
    assert lines==[r['enhanced_instruction']+'@@'+r['image'] for r in rs]
for file in ['comparison.csv','token_lengths.csv']:
    with (OUT/file).open(encoding='utf-8-sig',newline='') as f:csvrows=list(csv.DictReader(f))
    assert len(csvrows)==2395 and {key(r) for r in csvrows}==selected

sys.path.insert(0,str(PROJECT/'third_party/vpp2/src'))
from vpp2.models.wan21_14b.wan_video_text_encoder import HuggingfaceTokenizer
tok=HuggingfaceTokenizer(str(RUN/'tokenizer'),seq_len=128,clean='whitespace',local_files_only=True)
for name,digest in contract['tokenizer_sha256'].items():assert sha(RUN/'tokenizer'/name)==digest
for r in records:
    full=tok(r['enhanced_instruction'],padding=False,truncation=False,add_special_tokens=True)[0]
    ids,mask=tok(r['enhanced_instruction'],return_mask=True,add_special_tokens=True)
    assert full.tolist()==ids[0][mask[0].bool()].tolist()
    assert len(full)==r['token_count']<=120
    lengths.append(len(full))
report={'checked_at':datetime.now(timezone.utc).isoformat(),'status':'passed','episodes':2395,'excluded':105,'tasks':50,
        'max_tokens':max(lengths),'over128':0,'truncated':0,'all_text_csv_json_pairing_consistent':True,
        'source_records_and_pairing_unchanged':True,'tokenizer_snapshot_and_vpp2_128_path':'passed',
        'semantic_invariant_checks':dict(semantic_checks),
        'scope':'Exact source/format/token checks and selected semantic invariants; not a new visual review of source videos.'}
(OUT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
