#!/usr/bin/env python3
"""Render a separate CPU review of a completed object-video inference batch.

The default left reference is full-scene RGB at the predicted future times.
This is display-only: object-only GT remains the metric/training target.
Original predictions, traces, frozen reviews and metrics are never modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from urllib.parse import quote

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
FRAMES = 16
CHOSEN = (0, 4, 9, 15)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def video_digest(record, field):
    value = record[field]
    value = value["video"] if isinstance(value, dict) else value
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), f"Invalid media digest {field}")
    return value


def link(path, output):
    return quote(os.path.relpath(Path(path).resolve(), output), safe="/.-_")


def font(size):
    from PIL import ImageFont
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def read_prediction(path, width, height, fps):
    import av
    import numpy as np
    frames, times = [], []
    with av.open(str(path)) as container:
        require(len(container.streams.video) == 1, f"{path}: expected one video stream")
        stream = container.streams.video[0]
        stream.thread_count = 1
        require((stream.width, stream.height) == (width, height), f"{path}: unexpected resolution")
        require(stream.average_rate == fps, f"{path}: display FPS differs from trace")
        for frame in container.decode(stream):
            require(frame.pts is not None and frame.time_base is not None, f"{path}: missing PTS")
            frames.append(frame.to_ndarray(format="rgb24"))
            times.append(Fraction(frame.pts) * frame.time_base)
            require(len(frames) <= FRAMES, f"{path}: more than sixteen frames")
    require(len(frames) == FRAMES, f"{path}: expected sixteen frames")
    require(all(b-a == 1/fps for a,b in zip(times,times[1:])), f"{path}: nonuniform playback timestamps")
    return np.stack(frames), {"frames":FRAMES,"width":width,"height":height,"fps":str(fps),
                              "presentation_times_seconds":[float(x) for x in times]}


def read_source(path, ordinals, width, height):
    """Decode raw source ordinals; encoded FPS never selects source frames."""
    import av
    import numpy as np
    wanted, found = set(ordinals), {}
    with av.open(str(path)) as container:
        require(len(container.streams.video) == 1, f"{path}: expected one video stream")
        stream = container.streams.video[0]
        stream.thread_count = 1
        require((stream.width,stream.height) == (width,height), f"{path}: source resolution mismatch")
        for ordinal, frame in enumerate(container.decode(stream)):
            if ordinal in wanted:
                found[ordinal] = frame.to_ndarray(format="rgb24")
                if len(found) == len(wanted):
                    break
    require(set(found) == wanted, f"{path}: missing source ordinals {sorted(wanted-set(found))}")
    return np.stack([found[index] for index in ordinals])


def prepare(run, left_source):
    """Validate original provenance and production sampling without a model."""
    from vpp2_obj.infer import jsonable, load_config, make_dataset, select_episode
    batch_path = run / "batch_trace.json"
    batch = load_json(batch_path)
    require(batch.get("status") == "complete", "Batch inference must be complete")
    results = batch.get("results")
    require(isinstance(results,list) and results, "Batch must have nonempty results")
    require(all(r.get("status") == "complete" for r in results), "Every batch result must be complete")
    require(len({r["id"] for r in results}) == len(results), "Duplicate case IDs")
    require(batch.get("case_count",len(results)) == len(results), "Batch case_count mismatch")
    plan_path, plan_cases = run / "selection.json", {}
    plan_sha = None
    if plan_path.is_file():
        plan = load_json(plan_path)
        plan_sha = digest(plan_path)
        if "plan_sha256" in batch:
            require(batch["plan_sha256"] == plan_sha, "Selection plan SHA mismatch")
        for case in plan.get("cases",[]):
            require(case['id'] not in plan_cases, "Duplicate selection case")
            plan_cases[case['id']] = case
        require(set(plan_cases) == {r['id'] for r in results}, "Selection/batch cases differ")
    datasets, checked = {}, []
    for result in results:
        case_id = result["id"]
        require(isinstance(case_id,str) and re.fullmatch(r"[A-Za-z0-9_-]+",case_id), "Unsafe case ID")
        directory = run / "samples" / case_id
        trace_path, prediction_path = directory / "trace.json", directory / "future.mp4"
        if "trace" in result:
            require(Path(result['trace']).resolve() == trace_path, f"{case_id}: result trace path differs")
        trace = load_json(trace_path)
        expected = {"schema":"vpp2_obj.inference.v1","status":"complete","dry_run":False,
                    "exported_frames":FRAMES,"decoded_frame_slice":[9,25],
                    "target_video_read":False,"target_first_frame_read":False}
        for field,value in expected.items():
            require(trace.get(field) == value, f"{case_id}: invalid trace {field}")
        require(Path(trace['output_directory']).resolve() == directory and Path(trace['video_path']).resolve() == prediction_path, f"{case_id}: trace output paths differ")
        config, manifest = Path(trace['config']).resolve(), Path(trace['manifest']).resolve()
        require(digest(config) == trace['config_sha256'], f"{case_id}: config SHA changed")
        require(digest(manifest) == trace['manifest_sha256'], f"{case_id}: manifest SHA changed")
        if 'test_manifest_sha256' in batch:
            require(trace['manifest_sha256'] == batch['test_manifest_sha256'], "Batch/trace manifest mismatch")
        if 'checkpoint' in batch:
            require(Path(trace['checkpoint']).resolve() == Path(batch['checkpoint']).resolve(), "Batch/trace checkpoint mismatch")
        checkpoint = Path(trace['checkpoint'])
        if checkpoint.is_file():
            stat = checkpoint.stat()
            require(stat.st_size == trace['checkpoint_bytes'] and stat.st_mtime_ns == trace['checkpoint_mtime_ns'], "Checkpoint file metadata changed")
        key = (str(config),str(manifest))
        if key not in datasets:
            datasets[key] = make_dataset(load_config(config),manifest)
        dataset = datasets[key]
        index = select_episode(dataset.records,trace['task'],trace['episode'])
        anchor = trace['anchor_index']
        require(type(anchor) is int and anchor >= 0, "Invalid anchor")
        row = dataset.records[index]
        inference_indices = dataset.window_indices(index,anchor,include_target=False)
        for field,value in inference_indices.items():
            require(trace['sampling'].get(field) == jsonable(value), f"{case_id}: trace sampling mismatch {field}")
        indices = dataset.window_indices(index,anchor,include_target=True)
        require(len(indices['future_source_indices']) == FRAMES, "Expected sixteen future source frames")
        shape = trace['condition_shape']
        require(len(shape) == 4 and shape[:2] == [3,9] and tuple(shape[2:]) == tuple(dataset.video_size), "Condition dimensions mismatch")
        height,width = shape[2:]
        require(width%2 == height%2 == 0, "H.264 requires even frame dimensions")
        fps = Fraction(str(trace['display_fps']))
        require(fps > 0, "FPS must be positive")
        for field in ['condition_video_path','text_embedding_path']:
            require(Path(trace[field]).resolve() == dataset._path(row[field]).resolve(), f"{case_id}: trace {field} mismatch")
        for field in ['prompt','text_embedding_sha256','text_embedding_cache_signature']:
            require(trace.get(field) == row[field], f"{case_id}: trace {field} mismatch")
        selected = plan_cases.get(case_id,{})
        for field,trace_field in [('task','task'),('episode','episode'),('anchor','anchor_index'),('seed','seed'),('steps','num_inference_steps')]:
            if field in selected:
                require(selected[field] == trace[trace_field], f"{case_id}: selection {field} mismatch")
        for field,value in indices.items():
            if field in selected:
                require(selected[field] == jsonable(value), f"{case_id}: selection sampling mismatch {field}")
        media_field,hash_field = ('condition_video_path','rgb_sha256') if left_source == 'full-scene' else ('target_video_path','obj_sha256')
        source_path = dataset._path(row[media_field]).resolve()
        source_sha = video_digest(row,hash_field)
        require(digest(source_path) == source_sha, f"{case_id}: reference media SHA mismatch")
        prediction_sha = digest(prediction_path)
        # Reuse an existing frozen review as an additional identity check only.
        previous = directory / 'review/decode_check.json'
        if previous.is_file():
            old = load_json(previous)
            if 'prediction_sha256' in old:
                require(old['prediction_sha256'] == prediction_sha, f"{case_id}: prediction changed since original review")
            if 'trace_sha256' in old:
                require(old['trace_sha256'] == digest(trace_path), f"{case_id}: trace changed since original review")
        checked.append({'id':case_id,'trace':trace,'trace_path':trace_path,'trace_sha256':digest(trace_path),
                        'prediction_path':prediction_path,'prediction_sha256':prediction_sha,
                        'source_path':source_path,'source_sha256':source_sha,'indices':jsonable(indices),
                        'width':width,'height':height,'fps':fps,'selection':selected,
                        'dedup_index_path':str(dataset._path(row['dedup_index_path'])),
                        'dedup_index_sha256':row['dedup_index_sha256'],
                        'metric_target_video_path':str(dataset._path(row['target_video_path'])),
                        'metric_target_video_sha256':video_digest(row,'obj_sha256')})
    return checked, {'batch_trace':str(batch_path),'batch_trace_sha256':digest(batch_path),
                     'selection_plan':str(plan_path) if plan_path.is_file() else None,'selection_plan_sha256':plan_sha}


def time_label(item,index):
    us = item['indices']['future_timestamps_us'][index]
    anchor_us = item['indices']['condition_timestamps_us'][-1]
    return f"sim t={us/1e6:.3f}s ({(us-anchor_us)/1e6:+.3f}s)"


def comparison(path,item,left_label):
    import av
    import numpy as np
    from PIL import Image, ImageDraw
    width,height,fps = item['width'],item['height'],item['fps']
    with av.open(str(path),'w') as container:
        stream = container.add_stream('libx264',rate=fps)
        stream.width,stream.height,stream.pix_fmt = 2*width,height+68,'yuv420p'
        stream.options = {'crf':'18','preset':'fast','threads':'1'}
        for index in range(FRAMES):
            canvas = Image.new('RGB',(2*width,height+68),'#202936')
            canvas.paste(Image.fromarray(item['reference'][index]),(0,68))
            canvas.paste(Image.fromarray(item['prediction'][index]),(width,68))
            draw = ImageDraw.Draw(canvas)
            draw.text((10,5),left_label,font=font(17),fill='white')
            draw.text((width+10,5),'Object prediction',font=font(17),fill='white')
            draw.text((10,29),f"Future {index+1}/16 | dedup {item['indices']['future_indices'][index]} | {time_label(item,index)}",font=font(16),fill='#d5dfef')
            draw.text((10,49),f"Anchor {item['trace']['anchor_index']} | playback {fps} fps (not source simtime)",font=font(13),fill='#d5dfef')
            frame = av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24')
            frame.pts,frame.time_base = index,1/fps
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def contact(item,left_label):
    from PIL import Image, ImageDraw
    width,height,left,top = item['width'],item['height'],170,102
    canvas = Image.new('RGB',(left+4*width,top+2*height),'#202936')
    draw = ImageDraw.Draw(canvas)
    trace = item['trace']
    draw.text((10,8),f"{item['id']} | {trace['task']} / {trace['episode']} | anchor {trace['anchor_index']}",font=font(22),fill='white')
    for col,index in enumerate(CHOSEN):
        x = left+col*width
        draw.text((x+8,43),f"Future {index+1} | dedup {item['indices']['future_indices'][index]}",font=font(18),fill='white')
        draw.text((x+8,70),time_label(item,index),font=font(17),fill='#d5dfef')
        canvas.paste(Image.fromarray(item['reference'][index]),(x,top))
        canvas.paste(Image.fromarray(item['prediction'][index]),(x,top+height))
    draw.multiline_text((10,top+65),left_label.replace(' ', '\n',1),font=font(19),fill='white',spacing=6)
    draw.multiline_text((10,top+height+65),'Object\nprediction',font=font(19),fill='white',spacing=6)
    return canvas


def overview(items,left_label):
    from PIL import Image, ImageDraw
    width,height,top,row_height = 448,224,78,290
    canvas = Image.new('RGB',(4*width,top+len(items)*row_height),'#202936')
    draw = ImageDraw.Draw(canvas)
    draw.text((10,8),f"{left_label} / object prediction | same future source times",font=font(25),fill='white')
    for col,label in enumerate(['Reference F1','Prediction F1','Reference F16','Prediction F16']):
        draw.text((col*width+10,44),label,font=font(20),fill='#d5dfef')
    for row,item in enumerate(items):
        y=top+row*row_height
        draw.text((10,y),f"{item['id']} | anchor {item['trace']['anchor_index']} | F1 {time_label(item,0)} | F16 {time_label(item,15)}",font=font(18),fill='white')
        for col,frame in enumerate([item['reference'][0],item['prediction'][0],item['reference'][-1],item['prediction'][-1]]):
            image = Image.fromarray(frame)
            image.thumbnail((width,height))
            canvas.paste(image,(col*width+(width-image.width)//2,y+40+(height-image.height)//2))
    return canvas


def page(items,output,run,left_label,metrics):
    esc=html.escape
    metric_links=' · '.join(f'<a href="{link(path,output)}">{esc(path.name)}</a>' for path in metrics) or 'No existing metrics file was found; no metrics were computed.'
    original_link=f'<a href="{link(run/"index.html",output)}">Original review</a> · ' if (run/'index.html').is_file() else ''
    cards=[]
    for item in items:
        base=f"samples/{item['id']}"
        trace=item['trace']
        cards.append(f'''<article><h2>{esc(item['id'])}</h2><p>{esc(trace['task'])} / {esc(trace['episode'])} · anchor {trace['anchor_index']} · seed {trace['seed']} · {trace['num_inference_steps']} steps</p>
<p><strong>Prompt:</strong> {esc(trace['prompt'])}</p><video controls loop playsinline preload="metadata" src="{base}/comparison.mp4"></video>
<p>{esc(left_label)} left; object prediction right. Within each image: observer left, ego right. Playback {item['fps']} fps.</p>
<p><a href="{base}/comparison.mp4">Comparison MP4</a> · <a href="{link(item['prediction_path'],output)}">Original prediction</a> · <a href="{base}/verification.json">Source and pairing checks</a></p>
<details><summary>Future 1, 5, 10 and 16</summary><a href="{base}/contact.png"><img loading="lazy" src="{base}/contact.png" alt="Same-time reference and object prediction"></a></details></article>''')
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Object-video review — {esc(left_label)}</title>
<style>body{{margin:0;background:#151b25;color:#eef3fc;font:16px/1.55 system-ui,sans-serif}}main{{max-width:1450px;margin:auto;padding:24px}}a{{color:#9cc9ff}}article,.note{{padding:18px;background:#253144;margin:24px 0;border-radius:10px}}img{{max-width:100%;height:auto;display:block}}video{{width:100%;max-width:896px;background:black}}p{{overflow-wrap:anywhere}}summary{{cursor:pointer}}</style></head>
<body><main><h1>{esc(left_label)} / object prediction — {len(items)} cases</h1>
<p class="note">左侧显示与预测相同未来时刻的参考帧，右侧为物体预测。全景 RGB 仅用于展示；训练与指标的目标仍是黑底物体 GT（object_only）。此脚本不运行模型、不重算指标。播放 FPS 不代表源仿真时间，精确时间标注在每帧上。选窗方式沿用原 run，展示本身不改变样例代表性。</p>
<p>Existing metrics: {metric_links}</p><p>{original_link}<a href="summary.json">New display verification</a> · <a href="overview.png">Overview</a></p>
<a href="overview.png"><img src="overview.png" alt="Future first and last reference/prediction frames"></a>{''.join(cards)}</main></body></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True,help='Completed batch directory with batch_trace.json and samples/*/{trace.json,future.mp4}')
    parser.add_argument('--output',type=Path,help='New directory; defaults to RUN/full_scene_review_v1 or RUN/object_review_v1')
    parser.add_argument('--left-source',choices=['full-scene','object'],default='full-scene',help='Left reference type (default: full-scene future RGB)')
    args=parser.parse_args()
    run=args.run.expanduser().resolve()
    output=args.output.expanduser().resolve() if args.output else run/('full_scene_review_v1' if args.left_source=='full-scene' else 'object_review_v1')
    require(run.is_dir(),'Run directory does not exist')
    require(not output.exists() and not output.is_symlink(),f'Refusing to overwrite {output}')
    import torch
    torch.set_num_threads(1)
    items,provenance=prepare(run,args.left_source)
    metrics=sorted(path for path in run.glob('*metrics*.json') if path.is_file())
    left_label='Full-scene RGB GT' if args.left_source=='full-scene' else 'Object-only GT'
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=f'.{output.name}-',dir=output.parent))
    try:
        (staging/'samples').mkdir()
        reports=[]
        for item in items:
            target=staging/'samples'/item['id'];target.mkdir()
            item['prediction'],pred_decode=read_prediction(item['prediction_path'],item['width'],item['height'],item['fps'])
            item['reference']=read_source(item['source_path'],item['indices']['future_source_indices'],item['width'],item['height'])
            comparison(target/'comparison.mp4',item,left_label)
            _,comparison_decode=read_prediction(target/'comparison.mp4',2*item['width'],item['height']+68,item['fps'])
            contact(item,left_label).save(target/'contact.png')
            report={'schema':'vpp2_obj.display_review_case.v1','status':'passed','id':item['id'],
                    'display_reference':'full_scene_rgb' if args.left_source=='full-scene' else 'object_only',
                    'metric_target':'object_only','metrics_recomputed':False,'reference_used_as_model_input':False,
                    'task':item['trace']['task'],'episode':item['trace']['episode'],'anchor_index':item['trace']['anchor_index'],
                    'sampling':item['indices'],'source_video':str(item['source_path']),'source_video_sha256':item['source_sha256'],
                    'dedup_index_path':item['dedup_index_path'],'dedup_index_sha256':item['dedup_index_sha256'],
                    'prediction':str(item['prediction_path']),'prediction_sha256':item['prediction_sha256'],
                    'trace':str(item['trace_path']),'trace_sha256':item['trace_sha256'],
                    'manifest':item['trace']['manifest'],'manifest_sha256':item['trace']['manifest_sha256'],
                    'config':item['trace']['config'],'config_sha256':item['trace']['config_sha256'],
                    'metric_target_video_path':item['metric_target_video_path'],
                    'metric_target_video_sha256_from_manifest':item['metric_target_video_sha256'],
                    'prediction_decode':pred_decode,'comparison_decode':comparison_decode,
                    'artifacts':{name:{'sha256':digest(target/name),'bytes':(target/name).stat().st_size} for name in ['comparison.mp4','contact.png']}}
            write_json(target/'verification.json',report)
            reports.append({'id':item['id'],'status':'passed','verification':f"samples/{item['id']}/verification.json",'comparison':f"samples/{item['id']}/comparison.mp4"})
            print(f"Validated and staged {item['id']}",flush=True)
        overview(items,left_label).save(staging/'overview.png')
        (staging/'index.html').write_text(page(items,output,run,left_label,metrics),encoding='utf-8')
        summary={'schema':'vpp2_obj.display_review.v1','status':'passed','created_at_utc':datetime.now(timezone.utc).isoformat(),
                 'run':str(run),'output':str(output),'case_count':len(items),'left_source':args.left_source,
                 'display_reference':'full_scene_rgb' if args.left_source=='full-scene' else 'object_only',
                 'metric_target':'object_only','metrics_recomputed':False,'model_loaded':False,'gpu_used':False,
                 'future_reference_used_as_model_input':False,'source_frames_selected_by':'production future_source_indices; never MP4 playback FPS',
                 'script':str(Path(__file__).resolve()),'script_sha256':digest(__file__),**provenance,
                 'existing_metrics':[{'path':str(path),'relative_link':link(path,output),'sha256':digest(path)} for path in metrics],
                 'cases':reports,'artifacts':{name:{'sha256':digest(staging/name),'bytes':(staging/name).stat().st_size} for name in ['overview.png','index.html']}}
        write_json(staging/'summary.json',summary)
        require(not output.exists() and not output.is_symlink(),f'Output appeared while rendering: {output}')
        os.rename(staging,output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps({'status':'passed','case_count':len(items),'output':str(output),'index':str(output/'index.html')},indent=2))


if __name__=='__main__':
    main()
