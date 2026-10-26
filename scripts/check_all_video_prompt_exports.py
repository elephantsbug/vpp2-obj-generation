#!/usr/bin/env python3
"""Independent consistency checks for the completed VPP2 prompt export."""
from collections import Counter
import csv
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import tempfile

RUN = Path('/data2/niurc/vpp2-obj-generation/runs/prompt_all_video_agent_v1')
OUT = RUN / 'exports'

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def parse_text(path):
    lines = path.read_text().splitlines()
    assert all(line and line.count('@@') == 1 for line in lines), path
    entries = [line.split('@@') for line in lines]
    assert len({image for _, image in entries}) == len(entries), path
    return entries

records = read_jsonl(OUT / 'records.jsonl')
assert len(records) == 2500
by_id = {r['task'] + '/' + r['episode']: r for r in records}
assert len(by_id) == 2500
counts = Counter(r['task'] for r in records)
assert len(counts) == 50 and set(counts.values()) == {50}
assert set(counts) == {p.stem for p in (RUN / 'task_inputs').glob('*.json')}
for task in counts:
    assert {r['episode'] for r in records if r['task'] == task} == {f'episode_{i:07d}' for i in range(50)}
clean = [r for r in records if not r['review_flags']]
flagged = [r for r in records if r['review_flags']]
assert read_jsonl(OUT / 'needs_review.jsonl') == flagged

for directory, rows in [(OUT, records)] + [(OUT / 'by_task' / task, [r for r in records if r['task'] == task]) for task in sorted(counts)]:
    for filename, selected, key in [
        ('text_video_candidates_all.txt', rows, 'enhanced_instruction'),
        ('text_video_prompt_enhanced.txt', [r for r in rows if not r['review_flags']], 'enhanced_instruction'),
        ('text_original.txt', rows, 'original_instruction'),
    ]:
        assert parse_text(directory / filename) == [[r[key], r['image']] for r in selected], directory / filename

payload = json.loads((OUT / 'review_items.json').read_text())
assert len(payload['items']) == 2500
assert {i['id'] for i in payload['items']} == set(by_id)
links_checked = 0
for item in payload['items']:
    row = by_id[item['id']]
    assert item['enhanced_instruction'] == row['enhanced_instruction']
    assert item['original_instruction'] == row['original_instruction']
    assert item['review_flags'] == row['review_flags']
    assert item['first_active_arms'] == row['first_active_arms']
    assert item['temporal_relation'] == row['temporal_relation']
    assert item['decision'] == '待复核' and row['user_approved'] is False
    assert len(item['evidence_urls']) == len(row['evidence_pages_viewed'])
    urls = [item['image_url'], item['video_url'], item['record_url'], *item['evidence_urls']]
    for url in urls:
        assert not Path(url).is_absolute() and '..' not in Path(url).parts
        assert (OUT / url).is_file(), url
        links_checked += 1
    assert (OUT / item['image_url']).resolve() == Path(row['image_path']).resolve()
    assert (OUT / item['video_url']).resolve() == Path(row['video_path']).resolve()
    assert json.loads((OUT / item['record_url']).read_text()) == row
    for url, evidence in zip(item['evidence_urls'], row['evidence_pages_viewed']):
        assert (OUT / url).resolve() == Path(evidence).resolve()

for filename, rows in [('review_all.csv', records), ('needs_review.csv', flagged)]:
    with (OUT / filename).open(encoding='utf-8-sig', newline='') as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == len(rows)
    assert {r['条目ID'] for r in csv_rows} == {r['task'] + '/' + r['episode'] for r in rows}
    for row in csv_rows:
        assert row['当前候选prompt'] == by_id[row['条目ID']]['enhanced_instruction']
with (OUT / 'video_sequence.csv').open(encoding='utf-8-sig', newline='') as f:
    sequences = list(csv.DictReader(f))
assert len(sequences) == 2500
assert {r['task'] + '/' + r['episode'] for r in sequences} == set(by_id)

class Scripts(HTMLParser):
    def __init__(self):
        super().__init__(); self.current = None; self.scripts = []
    def handle_starttag(self, tag, attrs):
        if tag == 'script': self.current = [dict(attrs), '']
    def handle_data(self, data):
        if self.current is not None: self.current[1] += data
    def handle_endtag(self, tag):
        if tag == 'script' and self.current is not None:
            self.scripts.append(self.current); self.current = None

html = (OUT / 'review.html').read_text()
assert '__REVIEW_DATA_JSON__' not in html
assert '不能把后续画面透露的额外细节写入扩写' not in html
parser = Scripts(); parser.feed(html)
embedded = [body for attrs, body in parser.scripts if attrs.get('id') == 'review-data']
assert len(embedded) == 1 and json.loads(embedded[0]) == payload
code = '\n'.join(body for attrs, body in parser.scripts if attrs.get('type') != 'application/json')
with tempfile.NamedTemporaryFile(suffix='.js', mode='w') as f:
    f.write(code); f.flush()
    subprocess.run(['node', '--check', f.name], check=True, capture_output=True, text=True)
summary = json.loads((OUT / 'summary.json').read_text())
assert summary['tasks'] == 50 and summary['episodes'] == 2500 and summary['empty'] == 0
assert summary['needs_review'] == len(flagged)
assert summary['agent_reviewed_without_unresolved_flags'] == len(clean)
assert summary['run_id'] == payload['run_id']
report = {
    'checked_at': datetime.now(timezone.utc).isoformat(),
    'status': 'passed', 'tasks': 50, 'episodes': 2500,
    'nonempty_candidates': len(records), 'without_unresolved_flags': len(clean),
    'needs_review': len(flagged), 'resolved_media_and_evidence_links': links_checked,
    'per_task_text_packages': 50, 'text_csv_json_consistency': 'passed',
    'html_embedded_payload': 'passed', 'javascript_syntax': 'passed',
    'browser_render_tested': False,
    'scope': 'Structural and export consistency checks; semantic video review provenance is in per-episode records and cross_review reports.'
}
(RUN / 'progress' / 'final_export_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
