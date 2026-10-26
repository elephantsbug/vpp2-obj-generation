"""Chronological short captions from already video-reviewed per-episode evidence."""
import hashlib
from itertools import groupby
import json
from pathlib import Path
import re

PROJECT = Path('/data2/niurc/vpp2-obj-generation')
SOURCE = PROJECT / 'runs/prompt_all_video_agent_v1/exports/records.jsonl'
TOKENIZER = Path('/data2/niurc/wanvace-object-planner/models/Wan2.2-VACE-Fun-A14B/google/umt5-xxl')
PREFIX = 'Predict future object-only video from full-scene RGB history and the first frame. Black background; observer left, ego right.'
SUFFIX = 'This is a composed robot video with the observer view on the left and the robot ego view on the right.'

def match(pattern, text):
    m = re.search(pattern, text)
    if not m: raise ValueError('No match: ' + pattern + ' IN ' + text)
    return m.group(1)

def clauses(r):
    c = r['enhanced_instruction'].removesuffix(' ' + SUFFIX)
    right, left = c.split(', while the left robot gripper ', 1)
    return {'right': right.removeprefix('The right robot gripper '), 'left': left}

def subject(arm): return 'the ' + arm + ' gripper'

def render(r):
    t, s = r['task'], r['video_sequence']
    c = clauses(r)
    arms = [x['arms'][0] for x in s]
    actor = arms[0]
    obj = s[0]['object']
    if t == 'blocks_ranking_rgb':
        assert [x['object'] for x in s] == ['red block', 'green block', 'blue block']
        return (f'{subject(arms[0])} picks up and places the red block at the left end of a row in ego view; '
                f'then {subject(arms[1])} picks up and places green in the middle; '
                f'then {subject(arms[2])} picks up and places blue at the right end, completing the red-green-blue row.')
    if t == 'blocks_ranking_size':
        assert [x['object'] for x in s] == ['small block', 'medium block', 'large block']
        return (f'At table center, {subject(arms[0])} picks up and places the small block at the row\'s right end in ego view; '
                f'then {subject(arms[1])} places the lifted medium block to its left; '
                f'then {subject(arms[2])} places the lifted large block left of medium, forming a large-medium-small row.')
    if t == 'stack_blocks_two':
        assert [x['object'] for x in s] == ['red block', 'green block']
        return (f'{subject(arms[0])} picks up red and releases it at table center; then {subject(arms[1])} '
                'picks up green, aligns it over red and releases it on top, leaving a stable green-on-red block stack.')
    if t == 'stack_blocks_three':
        assert [x['object'] for x in s] == ['red block','red block','green block','green block','blue block','blue block']
        return (f'{subject(arms[0])} picks up the red block by its sides and releases it upright at table center; '
                f'then {subject(arms[2])} lifts and stacks green on red; then {subject(arms[4])} lifts and stacks blue on green, '
                'using side grasps and releasing each block. The tower remains upright.')
    if t == 'stack_bowls_two':
        second = s[2]['object']
        return (f'{subject(arms[0])} lifts the {obj} by its rim, places it open-side up at table center and releases it; '
                f'then {subject(arms[2])} lifts the remaining {second} by its rim, aligns and lowers it into the first bowl, '
                'then releases it, leaving both bowls nested.')
    if t == 'stack_bowls_three':
        caption = r['enhanced_instruction']
        positions = [match(rf'{word} blue ceramic bowl on the (left|right) side of the observer view', caption)
                     for word in ('first', 'second', 'remaining third')]
        return (f'{subject(arms[0])} lifts the first blue bowl at observer-{positions[0]} by its rim and releases it upright at table center; '
                f'then {subject(arms[1])} nests the second bowl from observer-{positions[1]} inside it; '
                f'then {subject(arms[2])} nests the remaining bowl from observer-{positions[2]} inside the second, '
                'lifting by the rim and releasing each bowl. All three remain upright.')
    if t == 'put_bottles_dustbin':
        groups = [(o, list(g)) for o,g in groupby(s, key=lambda step: step['object'])]
        assert len(groups) == 3
        names = [('clear water bottle' if 'water bottle' in o else 'black soda bottle' if 'soda bottle' in o else 'yellow bottle') for o,_ in groups]
        assert len(set(names)) == 3
        direct = [g[0]['arms']==['left'] for _,g in groups]
        assert all(len(g)==(2 if d else 3) for (_,g),d in zip(groups,direct))
        order = ', then '.join(names)
        if not any(direct):
            return (f'Dispose of the {order}, one bottle at a time. The right gripper lifts each by its upper body; left secures its lower body, '
                    'then right releases. Left drops it into the white-lined black bin at ego-left before the next pickup. All three end inside.')
        if all(direct):
            return (f'The left gripper picks up each bottle by its body and drops it into the white-lined black bin at ego-left, '
                    f'in this order: {order}. Finish each disposal before the next pickup, leaving all three inside.')
        steps = [f'left lifts the {n} by its body and drops it' if d else f'right passes the {n} to left to drop'
                 for n,d in zip(names,direct)]
        return ('The grippers use the white-lined black bin at ego-left. In order, ' + '; then '.join(steps) +
                '. For each pass, right lifts by the upper body; left secures the lower body before right releases. All three bottles end inside.')
    if t == 'place_bread_basket':
        if len(s) == 2:
            return f'{subject(actor)} grasps the {obj} from above, lifts it, lowers it into the breadbasket and releases it inside.'
        if r['temporal_relation'] == 'single_arm':
            return (f'{subject(actor)} picks up the {obj}, lowers it into the breadbasket and releases it; '
                    f'then picks up the remaining {s[1]["object"]} and lowers it into the same basket, leaving both inside.')
        assert s[1]['arms'][0]=='left' and s[2]['arms'][0]=='right'
        return (f'Simultaneously, the left gripper lifts the {s[1]["object"]} and the right lifts the {s[2]["object"]}. '
                'Left lowers and releases its bread into the basket first and withdraws as right holds its bread; '
                'then right lowers its bread inside, leaving both pieces in the basket.')
    if r['temporal_relation'] == 'single_arm':
        assert sum(r['arm_participation'].values()) == 1 and r['arm_participation'][actor]
        other = 'right' if actor=='left' else 'left'
        assert c[other].strip('.') == 'keeps static'
        return f'{subject(actor)} ' + c[actor].replace('robot ego view','ego view')
    if t == 'dump_bin_bigbin':
        assert arms == ['right','left','left','left']
        return (f'The right gripper lifts the {obj} by its upper rim, places it upright at table center and releases it; '
                'then left regrips the rim, lifts the bin over the large bin at observer-right and tilts it to pour the balls inside, '
                'ending with the small bin held tilted over the opening.')
    if t == 'grab_roller':
        return (f'Both grippers simultaneously grasp opposite ends of the {obj}, then lift together, '
                'keeping the single roller held above the table between them.')
    if t == 'lift_pot':
        return ('Both grippers simultaneously grasp opposite side handles of the gray cooking pot, then lift together, '
                'holding the pot upright above the table with its lid in place.')
    if t == 'handover_block':
        assert arms[0]=='left' and arms[-1]=='right'
        return ('The left gripper grasps the upper part of the red block, lifts and presents it upright between the arms. '
                'Right secures its lower part; then left releases and withdraws. Right carries the block to the blue pad '
                'and sets it upright there, remaining beside it.')
    if t == 'handover_mic':
        giver = arms[0]; receiver = 'left' if giver=='right' else 'right'
        return (f'{subject(giver)} lifts the {obj} by its handle and presents it upright near the center. '
                f'{receiver.capitalize()} grips the exposed handle; then {giver} releases and withdraws, leaving {receiver} holding the microphone upright.')
    if t == 'hanging_mug':
        assert arms == ['left','left','right','right']
        rack = match(r'tilts it toward the (.*?), guides', c['right'])
        return (f'The left gripper lifts the {obj} near its rim, rotates it, sets it upright at table center and releases it. '
                f'Then right regrips near the rim, lifts and tilts it toward the {rack}, guides its handle over a protruding rod '
                'and releases it, leaving the mug hanging by its handle.')
    if t == 'pick_diverse_bottles':
        right = match(r'^grasps the (.*?) around its body', c['right'])
        left = match(r'^simultaneously grasps the (.*?) around its body', c['left'])
        return (f'Simultaneously, the right gripper grasps the {right} and left grasps the {left}, both around their bodies. '
                'Both lift their bottles upright and draw them toward the robot, ending with one bottle held upright in each gripper.')
    if t == 'pick_dual_bottles':
        return ('Simultaneously, the right gripper grasps the green bottle with white markings by its body and left grasps '
                'the dark Coca-Cola bottle with a red cap and label by its body. Both lift and draw the bottles inward, holding them upright side by side above the table.')
    if t == 'place_bread_skillet':
        skillet_arm = s[2]['arms'][0]; bread_arm = s[3]['arms'][0]
        bread = match(r'^grasps the (.*?) around its sides', c[bread_arm])
        return (f'Simultaneously, {subject(skillet_arm)} lifts the skillet by its long handle and {bread_arm} lifts the {bread} by its sides. '
                f'{skillet_arm.capitalize()} moves the raised skillet to the center and holds it steady; then {bread_arm} lowers the bread into it, '
                'leaving the bread resting inside the held skillet.')
    if t == 'place_burger_fries':
        burger, fries = s[1]['object'], s[2]['object']
        tray = match(r'onto the (.*?), releases', s[1]['action'])
        extra = ' The fries carton tips onto its side.' if 'tips onto its side' in s[-1]['action'] else ''
        return (f'Simultaneously, the left gripper lifts the {burger} by its sides and right lifts the {fries} by its packaging. '
                f'Left lowers and releases the hamburger on the {tray} and withdraws as right holds the fries; '
                'then right lowers and releases the fries beside it on its ego-view right.' + extra)
    if t == 'place_can_basket':
        assert arms[0]==arms[1] and arms[2]!=arms[0]
        basket = s[2]['object']
        return (f'{subject(arms[0])} grasps the {obj} by its body, lifts and turns it horizontal, lowers and releases it into the {basket}, '
                f'then withdraws. {subject(arms[2]).capitalize()} grips the basket handle and lifts it, ending with the can inside the raised basket.')
    if t == 'place_cans_plasticbox':
        left,right = s[1]['object'],s[2]['object']
        first = (f'Each gripper grasps a {left} by its body on its own ego-view side; both lift simultaneously. ' if left == right else
                 f'Simultaneously, the left gripper lifts the {left} and right lifts the {right}, grasping the can bodies. ')
        return (first +
                'Left places and releases its can upright in the deep blue plastic box and withdraws as right waits holding its can; '
                'then right places and releases its can upright beside it and withdraws.')
    if t == 'place_dual_shoes':
        shoe = match(r'^grasps the (.*?) initially on',c['right'])
        box = match(r'in the (.*?), rotates',c['right']).replace('shoe-box','box')
        return (f'Simultaneously, the right gripper lifts the {shoe} from observer-left and left lifts its match from observer-right, gripping the uppers. '
                f'Left turns and places its shoe in the {box}, releases and withdraws while right holds the other; '
                'then right turns and lowers its shoe beside it. Both end soles down, toes toward ego-left.')
    if t == 'place_object_basket':
        basket = s[2]['object'].removesuffix(' handle')
        return (f'{subject(arms[0])} grasps the {obj} by opposite sides, lifts it, lowers and releases it into the {basket}, then withdraws. '
                f'{subject(arms[2]).capitalize()} grips the basket handle and lifts it, keeping the object inside and the basket held above the table.')
    if t == 'put_object_cabinet':
        object_arm=arms[0]; drawer_arm=s[1]['arms'][0]
        assert drawer_arm != object_arm
        return (f'{subject(object_arm)} grasps the {obj} on the table and waits. '
                f'Then {drawer_arm} grasps the drawer handle, pulls the drawer fully open and holds it open. '
                f'{object_arm.capitalize()} lifts the object, carries it over the open drawer, lowers and releases it inside; '
                f'{drawer_arm} keeps the drawer open.')
    if t == 'scan_object':
        box_arm,scanner_arm=arms[:2]
        box,scanner=s[0]['object'],s[1]['object']
        return (f'Simultaneously, {subject(box_arm)} lifts the {box} and {scanner_arm} lifts the {scanner} by its handle. '
                f'{box_arm.capitalize()} turns and holds the box toward the scanner; then {scanner_arm} brings the scanning face close to it to scan. '
                'Both items remain held and aligned.')
    raise ValueError('Unhandled task: ' + t)


def normalize(body):
    body = ' '.join(body.split()).strip()
    body = body[0].upper() + body[1:]
    if not body.endswith('.'): body += '.'
    return body


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    rows = [json.loads(x) for x in SOURCE.read_text().splitlines()]
    selected = [r for r in rows if not r['review_flags']]
    outputs=[];errors=[]
    for r in selected:
        try:
            body=normalize(render(r)); prompt=PREFIX+' '+body
            ids=tokenizer(prompt,add_special_tokens=True,truncation=False)['input_ids']
            outputs.append({'task':r['task'],'episode':r['episode'],'body':body,'prompt':prompt,'token_count':len(ids),
                            'source_caption':r['enhanced_instruction']})
        except Exception as e:
            errors.append({'task':r['task'],'episode':r['episode'],'error':str(e)})
    Path('/tmp/vpp2_obj_short_candidates.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2)+'\n')
    Path('/tmp/vpp2_obj_short_errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2)+'\n')
    lengths=[r['token_count'] for r in outputs]
    print(json.dumps({'rendered':len(outputs),'errors':len(errors),'max_tokens':max(lengths),'over120':sum(n>120 for n in lengths),'over128':sum(n>128 for n in lengths)}))
    for e in errors[:5]: print(e)
    for r in outputs:
        if r['token_count']>120:print(r['task'],r['episode'],r['token_count'],r['body'])

if __name__=='__main__': main()
