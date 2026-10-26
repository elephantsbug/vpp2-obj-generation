#!/usr/bin/env python3
"""Build an offline, synchronized RGB/object review from a paired-video JSONL.

Media paths may be absolute or relative to the manifest directory. This script
only reads metadata and checks file existence; it does not decode or alter media.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote


def require(ok, message):
    if not ok:
        raise ValueError(message)


def escape(value):
    return html.escape(str(value), quote=True)


def resolve_media(value, manifest):
    require(isinstance(value, str) and value.strip(), "Media path must be a nonempty string")
    path = Path(value).expanduser()
    path = path if path.is_absolute() else manifest.parent / path
    path = path.resolve()
    require(path.is_file(), f"Missing media file: {path}")
    return path


def relative_url(path, output):
    return escape(quote(os.path.relpath(path, output), safe="/.-_"))


def integer(value, name):
    require(isinstance(value, int) and not isinstance(value, bool), f"{name} must be an integer")
    return value


def load_rows(manifest):
    rows, seen = [], set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        require(isinstance(row, dict), f"Line {line_number}: expected JSON object")
        identifier = str(row.get("trajectory_id", "")).strip()
        require(identifier and identifier not in seen, f"Line {line_number}: missing or duplicate trajectory_id")
        seen.add(identifier)
        row = dict(row)
        row["trajectory_id"] = identifier
        row["split"] = str(row.get("split", "未标注"))
        count = integer(row.get("frame_count"), "frame_count")
        require(count > 0, f"{identifier}: frame_count must be positive")
        fps = float(Fraction(str(row["playback_fps"])))
        require(math.isfinite(fps) and fps > 0, f"{identifier}: invalid playback_fps")
        row["_fps"] = fps
        for field in ("condition_video_path", "target_video_path"):
            row["_" + field] = resolve_media(row.get(field), manifest)
        require(row["_condition_video_path"] != row["_target_video_path"], f"{identifier}: RGB and object paths must differ")
        row["_preview_path"] = resolve_media(row["preview_path"], manifest) if row.get("preview_path") else None
        start, end = row.get("selected_review_start"), row.get("selected_review_end")
        require((start is None) == (end is None), f"{identifier}: review start/end must both be present or absent")
        if start is not None:
            integer(start, "selected_review_start")
            integer(end, "selected_review_end")
            require(0 <= start <= end < count, f"{identifier}: review segment is out of range")
        segments = row.get("quality", {}).get("shared_valid_segments_inclusive", [])
        require(isinstance(segments, list), f"{identifier}: shared_valid_segments_inclusive must be a list")
        for segment in segments:
            require(isinstance(segment, (list, tuple)) and len(segment) == 2, f"{identifier}: expected inclusive [start, end] segment")
            first, last = segment
            integer(first, "segment start")
            integer(last, "segment end")
            require(0 <= first <= last < count, f"{identifier}: shared segment is out of range")
        if start is not None:
            require(any(first <= start <= end <= last for first, last in segments), f"{identifier}: selected segment is not within a shared valid segment")
        row["_segments"] = segments
        rows.append(row)
    require(rows, "Manifest has no records")
    return rows


STYLE = """
:root { color-scheme:light; --ink:#172b3b; --muted:#596b78; --line:#d9e2e8; --accent:#17684c; }
* { box-sizing:border-box; }
body { margin:0; background:#f3f6f8; color:var(--ink); font:15px/1.65 system-ui,-apple-system,'Segoe UI',sans-serif; }
main { max-width:1240px; margin:auto; padding:28px 22px 60px; }
h1 { font-size:28px; margin:0 0 12px; } h2 { font-size:18px; margin:0; overflow-wrap:anywhere; }
p { margin:8px 0; } a { color:#126b96; } button,select,input { font:inherit; }
button,select { min-height:38px; padding:5px 13px; border:1px solid #bacbd6; border-radius:7px; background:white; color:var(--ink); }
button { cursor:pointer; } button:hover { border-color:var(--accent); } button:focus-visible,a:focus-visible,input:focus-visible { outline:3px solid #57a99c; outline-offset:2px; }
.intro,.toolbar,.card { background:white; border:1px solid var(--line); border-radius:12px; }
.intro { padding:20px; } .intro ul { padding-left:23px; margin:10px 0 0; }
.muted,.meta,.sources { color:var(--muted); font-size:13px; }
.toolbar { margin:18px 0; padding:12px 16px; display:flex; align-items:center; flex-wrap:wrap; gap:15px; position:sticky; top:0; z-index:2; box-shadow:0 2px 8px #172b3b0a; }
.card { margin:18px 0; padding:20px; } .card[hidden] { display:none; }
.heading { display:flex; align-items:center; flex-wrap:wrap; gap:10px; }
.badge { border-radius:5px; background:#eaf0f5; padding:1px 9px; font-size:13px; }
.instruction { margin:13px 0; padding:12px 15px; background:#f4f7f9; border-radius:7px; overflow-wrap:anywhere; }
.review-note { padding:10px 14px; background:#fff5dd; border-left:3px solid #c28a20; border-radius:5px; }
.players { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin:13px 0 6px; }
figure { margin:0; } figcaption { margin-bottom:5px; font-weight:600; }
video { width:100%; height:auto; aspect-ratio:2/1; display:block; background:black; border-radius:7px; }
.controls { margin:12px 0 8px; display:flex; flex-wrap:wrap; gap:9px; align-items:center; }
.controls .play { background:var(--accent); border-color:var(--accent); color:white; }
.timeline { display:flex; gap:12px; align-items:center; } .timeline input { width:100%; min-width:80px; accent-color:var(--accent); }
.frame-display { min-width:135px; font-variant-numeric:tabular-nums; white-space:nowrap; text-align:right; }
.segments { color:#28664c; } .status { color:#8a3a19; min-height:24px; }
.sources { overflow-wrap:anywhere; } details { margin-top:12px; } summary { cursor:pointer; }
.preview { display:block; max-width:100%; height:auto; margin-top:10px; border-radius:7px; }
noscript { display:block; padding:12px; background:#fff3d8; }
@media(max-width:700px) { main { padding:15px 10px; } .card { padding:13px; } .players { grid-template-columns:1fr; } h1 { font-size:23px; } .toolbar { position:static; } }
"""


SCRIPT = r"""
'use strict';
const controllers = [];
for (const card of document.querySelectorAll('.card')) {
  const rgb = card.querySelector('.rgb'), obj = card.querySelector('.obj');
  const videos = [rgb, obj], fps = Number(card.dataset.fps), count = Number(card.dataset.frames);
  const hasSegment = card.dataset.start !== '', start = hasSegment ? Number(card.dataset.start) : 0;
  const end = hasSegment ? Number(card.dataset.end) : count - 1;
  const play = card.querySelector('.play'), slider = card.querySelector('.seek');
  const display = card.querySelector('.frame-display'), status = card.querySelector('.status');
  const mode = card.querySelector('.mode');
  let selected = hasSegment, pendingFrame = start, requestedPlay = false, generation = 0, raf = 0;
  function frameAt(time) { return Math.max(0, Math.min(count - 1, Math.floor(time * fps + 0.0001))); }
  function update() {
    const frame = rgb.readyState ? frameAt(rgb.currentTime) : pendingFrame;
    slider.value = String(frame); display.textContent = `帧 ${frame} / ${count - 1}`;
    mode.textContent = selected ? `有效段 ${start}–${end}` : '全程';
  }
  function seek(frame) {
    pendingFrame = Math.max(0, Math.min(count - 1, Math.round(frame)));
    for (const video of videos) { if (video.readyState >= 1) video.currentTime = pendingFrame / fps; }
    update();
  }
  function pause() {
    requestedPlay = false; generation += 1; cancelAnimationFrame(raf);
    videos.forEach(video => video.pause()); play.textContent = '同步播放';
  }
  function tick() {
    if (!requestedPlay) return;
    if (selected && rgb.currentTime >= (end + 1) / fps) { pause(); seek(end); return; }
    if (!rgb.seeking && !obj.seeking && Math.abs(rgb.currentTime - obj.currentTime) > 0.75 / fps) {
      obj.currentTime = rgb.currentTime;
    }
    update(); raf = requestAnimationFrame(tick);
  }
  async function startPlaying() {
    controllers.forEach(controller => { if (controller.card !== card) controller.pause(); });
    if (selected && (frameAt(rgb.currentTime) < start || frameAt(rgb.currentTime) >= end)) seek(start);
    else if (!selected && (rgb.ended || frameAt(rgb.currentTime) >= count - 1)) seek(0);
    requestedPlay = true; const ticket = ++generation; status.textContent = '加载视频…';
    try {
      await Promise.all(videos.map(video => video.play()));
      if (ticket !== generation) return;
      if (!requestedPlay) { videos.forEach(video => video.pause()); return; }
      status.textContent = ''; play.textContent = '暂停';
      cancelAnimationFrame(raf); raf = requestAnimationFrame(tick);
    } catch (error) {
      if (ticket !== generation) return;
      pause(); status.textContent = '浏览器未能播放，可使用下方“打开全景 / 打开物体”链接检查视频。';
    }
  }
  play.addEventListener('click', () => requestedPlay ? pause() : startPlaying());
  card.querySelector('.all').addEventListener('click', () => { pause(); selected = false; seek(0); });
  const segment = card.querySelector('.segment');
  if (segment) segment.addEventListener('click', () => { pause(); selected = true; seek(start); });
  slider.addEventListener('input', () => { pause(); selected = false; seek(Number(slider.value)); });
  for (const video of videos) {
    video.addEventListener('loadedmetadata', () => { video.currentTime = pendingFrame / fps; update(); });
    video.addEventListener('error', () => { pause(); status.textContent = '视频读取失败，请检查本地文件路径与浏览器支持。'; });
    video.addEventListener('ended', pause);
  }
  rgb.addEventListener('timeupdate', update);
  controllers.push({card, pause}); update();
}
document.querySelector('#split-filter').addEventListener('change', event => {
  let visible = 0;
  controllers.forEach(controller => {
    const hidden = event.target.value !== 'all' && controller.card.dataset.split !== event.target.value;
    controller.card.hidden = hidden;
    if (hidden) controller.pause(); else visible += 1;
  });
  document.querySelector('#visible-count').textContent = String(visible);
});
document.querySelector('#pause-all').addEventListener('click', () => controllers.forEach(controller => controller.pause()));
document.addEventListener('visibilitychange', () => { if (document.hidden) controllers.forEach(controller => controller.pause()); });
"""


def card_html(row, ordinal, output):
    rgb = relative_url(row["_condition_video_path"], output)
    obj = relative_url(row["_target_video_path"], output)
    start, end = row.get("selected_review_start"), row.get("selected_review_end")
    start_attr, end_attr = ("", "") if start is None else (str(start), str(end))
    segments = "；".join(f"{a}–{b}" for a, b in row["_segments"]) or "无共同有效区间"
    selected_button = '<button class="segment" type="button">有效段起点</button>' if start is not None else ""
    preview = ""
    if row["_preview_path"]:
        url = relative_url(row["_preview_path"], output)
        preview = f'<details open><summary>关键帧预览</summary><a href="{url}"><img class="preview" src="{url}" loading="lazy" alt="{escape(row["trajectory_id"])} 关键帧联系表"></a></details>'
    instruction = row.get("instruction_hint") or "（未提供英文指令）"
    notes = "".join(f'<p class="review-note">检查提示：{escape(note)}</p>' for note in row.get("review_notes", []))
    return f"""<article class="card" id="trajectory-{ordinal}" data-split="{escape(row['split'])}" data-fps="{row['_fps']}" data-frames="{row['frame_count']}" data-start="{start_attr}" data-end="{end_attr}">
      <div class="heading"><h2>{ordinal:02d} · {escape(row['trajectory_id'])}</h2><span class="badge">{escape(row['split'])}</span></div>
      <p class="instruction" lang="en">{escape(instruction)}</p>
      {notes}
      <p class="meta">{row['frame_count']} 帧 · 名义播放频率 {row['_fps']:g} fps · 两路视频内部均为左：外部视角 1，右：外部视角 2</p>
      <div class="players">
        <figure><figcaption>全景 RGB</figcaption><video class="rgb" src="{rgb}" preload="metadata" muted playsinline aria-label="全景 RGB 视频"></video></figure>
        <figure><figcaption>物体 · 黑背景</figcaption><video class="obj" src="{obj}" preload="metadata" muted playsinline aria-label="黑背景物体视频"></video></figure>
      </div>
      <div class="controls"><button class="play" type="button">同步播放</button>{selected_button}<button class="all" type="button">全程 / 返回首帧</button><span class="badge mode"></span></div>
      <div class="timeline"><input class="seek" type="range" min="0" max="{row['frame_count'] - 1}" step="1" value="{start or 0}" aria-label="同步跳转视频帧"><output class="frame-display"></output></div>
      <p class="segments">共同有效段（含首尾帧）：{escape(segments)}</p>
      <p class="status" role="status" aria-live="polite"></p>
      <p class="sources">外部视角 1：{escape(row.get('view1_episode_name', '未提供'))}<br>外部视角 2：{escape(row.get('view2_episode_name', '未提供'))}<br><a href="{rgb}">打开全景视频</a> · <a href="{obj}">打开物体视频</a></p>
      {preview}
    </article>"""


def render(rows, manifest, output):
    counts = Counter(row["split"] for row in rows)
    splits = [name for name in ("train", "val") if name in counts]
    splits += sorted(set(counts) - set(splits))
    options = '<option value="all">全部</option>' + "".join(f'<option value="{escape(name)}">{escape(name)} · {counts[name]}</option>' for name in splits)
    cards = "\n".join(card_html(row, i, output) for i, row in enumerate(rows, 1))
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DROID 全景与物体视频检查</title><style>{STYLE}</style></head>
<body><main><section class="intro"><h1>DROID 全景与物体视频检查</h1>
<p>共 {len(rows)} 条轨迹。每条的全景与物体视频同步播放，内部均为<strong>左：外部视角 1，右：外部视角 2</strong>。</p>
<p>完整视频保留全程所有帧，默认从选定有效段起点开始。绿色区间仅代表待选监督段，仍需人工检查；区间外可能出现空 mask 或遮挡。</p>
<p>物体视频保留现有 mask 标注的目标，可能只包含任务涉及物体的一部分。下方英文是原始任务指令，尚未改写为最终训练 prompt。</p>
<ul><li>两视角是否保留同一任务目标，有没有混入无关物体。</li><li>物体是否漏选、断裂或突然消失；结合全景判断是否为真实遮挡。</li><li>物体轮廓、位置与时序是否对应全景，有没有左右错位或帧错位。</li></ul>
<p class="muted">帧号从 0 开始。可切换“有效段 / 全程”，或拖动进度条逐帧检查。<a href="{relative_url(manifest, output)}">查看数据清单</a></p></section>
<noscript>同步播放和筛选需要启用浏览器 JavaScript；仍可使用各条下方链接直接打开视频。</noscript>
<div class="toolbar"><label for="split-filter">数据划分</label><select id="split-filter">{options}</select><span>当前显示 <strong id="visible-count">{len(rows)}</strong> 条</span><button id="pause-all" type="button">全部暂停</button></div>
{cards}</main><script>{SCRIPT}</script></body></html>\n"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--notes", type=Path, help="Optional JSON mapping trajectory IDs to review-note lists")
    args = parser.parse_args()
    manifest, output = args.manifest.expanduser().resolve(), args.output_dir.expanduser().resolve()
    require(manifest.is_file(), f"Missing manifest: {manifest}")
    rows = load_rows(manifest)
    if args.notes:
        notes = json.loads(args.notes.read_text(encoding="utf-8"))
        require(isinstance(notes, dict), "Notes must map trajectory IDs to note lists")
        require(not (set(notes) - {row['trajectory_id'] for row in rows}), "Notes reference unknown trajectories")
        for row in rows:
            value = notes.get(row["trajectory_id"], [])
            require(isinstance(value, list) and all(isinstance(note, str) for note in value), "Review notes must be strings")
            row["review_notes"] = value
    document = render(rows, manifest, output)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / ".index.html.tmp"
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output / "index.html")
    print(json.dumps({"status": "created", "index": str(output / "index.html"), "trajectory_count": len(rows), "split_counts": dict(Counter(row["split"] for row in rows))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
