"""Strict parsing, review flags and deterministic rendering for two-step prompts.

These checks enforce declared constraints; they are not a visual/semantic proof.
In particular, observer-side proximity is a reviewable assignment heuristic.
"""

import copy
import json
import re


VIEW_SUFFIX = (
    "This is a composed robot video with the observer view on the left "
    "and the robot ego view on the right."
)
ARMS = ("right", "left")
ACTION_KINDS = {
    "grasp", "lift", "move", "place", "orient", "use", "hold", "support",
    "transfer", "receive", "other",
}
ASSIGNMENT_MODES = {"explicit", "proximity", "both", "receiver", "inactive"}
RGB_COLORS = ("red", "green", "blue")
RGB_OPERATION_RE = re.compile(
    r"\b(?:grasp(?:s|ed|ing)?|grab(?:s|bed|bing)?|pick(?:s|ed|ing)?(?:\s+up)?|"
    r"lift(?:s|ed|ing)?|mov(?:e|es|ed|ing)|plac(?:e|es|ed|ing)|"
    r"position(?:s|ed|ing)?|set(?:s|ting)?|put(?:s|ting)?|"
    r"arrang(?:e|es|ed|ing)|align(?:s|ed|ing)?|reposition(?:s|ed|ing)?|"
    r"push(?:es|ed|ing)?|slid(?:e|es|ing)|hold(?:s|ing)?|held)\b"
)


def _norm(text):
    return " ".join(text.lower().split()) if isinstance(text, str) else ""


def _has_evidence(evidence, task):
    return bool(_norm(evidence)) and _norm(evidence) in _norm(task)


def required_rgb_objects(task):
    """Return source-mentioned RGB block colors, in first-mention order.

    This deliberately narrow check activates only for an operation instruction
    explicitly naming red block, green block and blue block. It does not infer
    required objects or spatial order from a dataset/task directory name.
    """
    text = _norm(task)
    colors = list(dict.fromkeys(re.findall(r"\b(red|green|blue) blocks?\b", text)))
    if set(colors) != set(RGB_COLORS) or not RGB_OPERATION_RE.search(text):
        return []
    return colors


def _rgb_action_mentions(phrase):
    """Recognize limited direct-object/list forms; unrecognized prose is reviewed.

    A color occurring only in a destination, relation, or final ordering list is
    not evidence that the arm manipulates that object. This lexical heuristic
    does not prove that the full action description is semantically correct.
    """
    text = _norm(phrase)
    atom = r"(?:the |a |an )?(?:red|green|blue)(?: (?:block|cube)s?)?"
    direct_objects = re.compile(
        rf"\s*(?P<objects>{atom}(?:(?:,\s*(?:and )?| and | followed by ){atom})*)\b"
    )
    colors = set()
    for verb in RGB_OPERATION_RE.finditer(text):
        # Negated acts do not establish object coverage.
        if re.search(r"\b(?:not|never|without|avoid(?:s)?)\s*$", text[max(0, verb.start() - 18):verb.start()]):
            continue
        match = direct_objects.match(text[verb.end():])
        if match and re.search(r"\b(?:block|cube)s?\b", match.group("objects")):
            colors.update(re.findall(r"\b(?:red|green|blue)\b", match.group("objects")))
    return colors


def _rgb_object_coverage_flags(task, objects, arms, action_by_arm, phrases):
    required = required_rgb_objects(task)
    if not required:
        return []
    errors = []
    mentions = {arm: _rgb_action_mentions(phrase) for arm, phrase in phrases.items()}
    requires_placement = bool(re.search(
        r"\b(?:mov(?:e|es|ed|ing)|plac(?:e|es|ed|ing)|position(?:s|ed|ing)?|"
        r"put(?:s|ting)?|set(?:s|ting)?|arrang(?:e|es|ed|ing)|align(?:s|ed|ing)?)\b",
        _norm(task),
    ))
    for color in required:
        candidates = [
            object_id for object_id, obj in objects.items()
            if re.search(rf"\b{color}\b", _norm(obj.get("label")))
            and re.search(r"\b(?:block|cube)s?\b", _norm(obj.get("label")))
        ]
        if not candidates:
            errors.append(f"grounding: rgb_object_grounding_missing: required {color} block is not identified")
            continue
        if len(candidates) != 1 or len(set(re.findall(
            r"\b(?:red|green|blue)\b", _norm(objects[candidates[0]].get("label"))
        ))) != 1:
            errors.append(f"grounding: rgb_object_grounding_ambiguous: required {color} block needs a distinct identifiable object")
            continue
        object_id = candidates[0]
        participating = [
            arm for arm, actions in action_by_arm.items()
            if isinstance(arms.get(arm), dict) and arms[arm].get("active") is True
            and any(
                action.get("object_id") == object_id
                and isinstance(action.get("kind"), str)
                and action["kind"] in ACTION_KINDS - {"other"}
                for action in actions
            )
        ]
        if not participating:
            errors.append(
                f"plan.arms: rgb_object_action_missing: required {color} block ({object_id}) "
                "needs an active arm's manipulation action.object_id; requirements, tools and destinations do not count"
            )
        elif requires_placement and not any(
            action.get("object_id") == object_id and action.get("kind") in ("move", "place")
            for arm in participating for action in action_by_arm[arm]
        ):
            errors.append(
                f"plan.arms: rgb_object_placement_action_missing: required {color} block ({object_id}) "
                "needs a move/place action for the source arrangement task; grasp/hold alone is insufficient"
            )
        for arm in participating:
            if color not in mentions.get(arm, set()):
                errors.append(
                    f"plan.arms.{arm}: rgb_object_caption_action_unverified: {color} block has a structured action "
                    "but no recognized explicit manipulation in this arm's verb_phrase; a final order list or destination is insufficient"
                )
    return errors


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError(f"Non-JSON constant: {value}")


def parse_json(text):
    """Parse one JSON object, optionally in one complete JSON code fence."""
    if not isinstance(text, str):
        raise ValueError("Model output must be a JSON string")
    text = text.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?[ \t]*\n(.*?)\n```", text, re.S)
        if not match:
            raise ValueError("Expected one complete JSON code fence")
        text = match.group(1)
    try:
        result = json.loads(
            text, object_pairs_hook=_unique_pairs, parse_constant=_bad_constant
        )
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def _uncertainties(value, path, errors):
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        errors.append(f"{path}: expected a list of strings")
    else:
        for item in value:
            if item.strip():
                errors.append(f"{path}: {item.strip()}")


def enrich_grounding(ground):
    """Validate normalized observer coordinates and add the fixed arm mapping."""
    errors = []
    if not isinstance(ground, dict):
        return {}, ["grounding: expected an object"]
    result = copy.deepcopy(ground)
    objects = result.get("objects")
    if not isinstance(objects, list):
        errors.append("grounding.objects: expected a list")
        objects = []
        result["objects"] = []
    seen = set()
    for index, obj in enumerate(objects):
        path = f"grounding.objects[{index}]"
        if not isinstance(obj, dict):
            errors.append(f"{path}: expected an object")
            continue
        object_id = obj.get("id")
        if not isinstance(object_id, str) or not object_id.strip():
            errors.append(f"{path}.id: expected a nonempty string")
        elif object_id in seen:
            errors.append(f"{path}.id: duplicate id {object_id!r}")
        else:
            seen.add(object_id)
        if not isinstance(obj.get("label"), str) or not obj["label"].strip():
            errors.append(f"{path}.label: expected a nonempty string")
        confidence = obj.get("confidence")
        if confidence not in ("high", "medium", "low"):
            errors.append(f"{path}.confidence: expected high, medium or low")
        elif confidence != "high":
            errors.append(f"{path}: {confidence}-confidence object localization needs review")
        center = obj.get("center")
        valid_center = (
            isinstance(center, list) and len(center) == 2
            and all(type(v) in (int, float) and 0 <= v <= 1000 for v in center)
        )
        obj["observer_side"] = None
        obj["preferred_arm"] = None
        if not valid_center:
            errors.append(f"{path}.center: expected two finite coordinates in [0, 1000]")
            continue
        x = center[0]
        side = "left" if x < 475 else "right" if x > 525 else "center"
        obj["observer_side"] = side
        obj["preferred_arm"] = {"left": "right", "right": "left"}.get(side)
    _uncertainties(result.get("uncertainties"), "grounding.uncertainties", errors)
    return result, errors


def _explicit_arms(text):
    text = _norm(text)
    arms = set()
    for arm in ARMS:
        if re.search(
            rf"\b{arm}(?:[- ](?:robot|robotic|mechanical))?[- ](?:arm|hand|gripper)s?\b",
            text,
        ) or re.search(rf"\b(?:arm|hand|gripper) on (?:the )?{arm}\b", text):
            arms.add(arm)
    if re.search(r"\b(?:left and right|right and left) (?:arms|hands|grippers)\b", text):
        arms.update(ARMS)
    return arms


def _both_evidence(text):
    return bool(re.search(
        r"\b(?:both|two) (?:robot )?(?:arms|hands|grippers)\b|\btogether\b",
        _norm(text),
    )) or _explicit_arms(text) == set(ARMS)


def _requires_handover(text):
    """A transfer to a container does not imply a second participating arm."""
    text = _norm(text)
    return bool(re.search(
        r"\b(?:handover|hand[- ]over)\b"
        r"|\b(?:to|into) (?:the |its |your )?(?:other|left|right) "
        r"(?:robot )?(?:arm|hand|gripper)\b"
        r"|\b(?:transfer\w*|pass\w*|give\w*)\b[^.!?;]*"
        r"\b(?:another|other) (?:robot )?(?:arm|hand|gripper)\b"
        r"|\b(?:transfer\w*|pass\w*|give\w*)\b[^.!?;]*"
        r"\bbetween (?:the |both |two )?(?:arms|hands|grippers)\b",
        text,
    ))


def _action_family_flags(task, phrases):
    """Flag clear required action omissions without claiming semantic proof."""
    task = _norm(task)
    captions = [_norm(phrase) for phrase in phrases]
    lift_pattern = r"\b(?:pick(?:s|ed|ing)? up|lift(?:s|ed|ing)?|elevat(?:e|es|ed|ing)|rais(?:e|es|ed|ing))\b"
    errors = []
    if re.search(lift_pattern, task) and not any(re.search(lift_pattern, phrase) for phrase in captions):
        errors.append("plan.arms: caption omits required lifting action (pick up/lift/elevate/raise); grasp alone is insufficient")
    # Bare 'hammer' is usually an object noun; do not count 'grasps the hammer'.
    hit_pattern = r"\b(?:hits?|hitting|strikes?|striking|struck|hammers|hammering|hammered)\b|(?:^|\b(?:to|and|then) )hammer\b"
    if re.search(r"\bhit(?:s|ting)?\b", task) and not any(re.search(hit_pattern, phrase) for phrase in captions):
        errors.append("plan.arms: caption omits required hitting action (hit/strike/hammer)")
    return errors


def _explicit_arm_evidence(task, arm):
    """Return an original task span, preserving its spelling and spacing."""
    patterns = (
        rf"\b{arm}(?:(?:-|\s+)(?:robot|robotic|mechanical))?"
        rf"(?:-|\s+)(?:arm|hand|gripper)s?\b",
        rf"\b(?:arm|hand|gripper)\s+on\s+(?:the\s+)?{arm}\b",
    )
    for pattern in patterns:
        match = re.search(pattern, task, re.I)
        if match:
            return match.group(0)
    # Combined references such as 'left and right arms' may need the full task.
    return task


def resolve_assignment_modes(plan, grounding, task):
    """Derive assignment metadata, preserving all model semantic decisions.

    Localization is intentionally left for validate_plan: this function never
    changes an arm, object id, action, participation decision or caption phrase.
    The original assignment is retained as model_assignment for review.
    """
    result = copy.deepcopy(plan)
    if not isinstance(result, dict) or not isinstance(result.get("arms"), dict):
        return result
    arms = result["arms"]
    task_text = task if isinstance(task, str) else ""
    explicit = _explicit_arms(task_text)
    both_supported = _both_evidence(task_text)
    for arm in ARMS:
        data = arms.get(arm)
        if not isinstance(data, dict) or not isinstance(data.get("assignment"), dict):
            continue
        assignment = data["assignment"]
        # Keep the first raw assignment if a caller resolves an already resolved plan.
        if "model_assignment" not in data:
            data["model_assignment"] = copy.deepcopy(assignment)
        actions = data.get("actions")
        has_receive = isinstance(actions, list) and any(
            isinstance(action, dict) and action.get("kind") == "receive"
            for action in actions
        )
        other = arms.get("left" if arm == "right" else "right")
        other_assignment = other.get("assignment") if isinstance(other, dict) else None
        same_anchor = (
            data.get("active") is True
            and isinstance(other, dict) and other.get("active") is True
            and isinstance(assignment.get("object_id"), str)
            and isinstance(other_assignment, dict)
            and assignment.get("object_id") == other_assignment.get("object_id")
        )
        if data.get("active") is False:
            mode, evidence = "inactive", ""
        elif has_receive:
            mode, evidence = "receiver", task_text
        elif same_anchor and both_supported:
            mode, evidence = "both", task_text
        elif arm in explicit:
            mode, evidence = "explicit", _explicit_arm_evidence(task_text, arm)
        else:
            mode, evidence = "proximity", ""
        assignment["mode"] = mode
        assignment["evidence"] = evidence
    return result


def _paired_handover_receivers(action_by_arm):
    receivers = set()
    for arm, actions in action_by_arm.items():
        other = "left" if arm == "right" else "right"
        for action in actions:
            if (
                action.get("kind") == "transfer"
                and action.get("other_arm") == other
                and isinstance(action.get("object_id"), str)
                and any(
                    counterpart.get("kind") == "receive"
                    and counterpart.get("other_arm") == arm
                    and counterpart.get("object_id") == action.get("object_id")
                    for counterpart in action_by_arm.get(other, [])
                )
            ):
                receivers.add(other)
    return receivers


def _cross_arm_tool_flags(arms, action_by_arm):
    """Detect exclusive tool possession on the opposite arm, without rewriting."""
    errors = []
    for arm, actions in action_by_arm.items():
        other = "left" if arm == "right" else "right"
        for action in actions:
            tool_id = action.get("tool_id")
            if not isinstance(tool_id, str):
                continue
            holders = {
                side for side, side_actions in action_by_arm.items()
                if any(
                    candidate.get("kind") in ("grasp", "hold")
                    and candidate.get("object_id") == tool_id
                    for candidate in side_actions
                )
            }
            if holders != {other}:
                continue
            shared_tool = all(
                isinstance(arms.get(side), dict)
                and arms[side].get("active") is True
                and isinstance(arms[side].get("assignment"), dict)
                and arms[side]["assignment"].get("mode") == "both"
                and arms[side]["assignment"].get("object_id") == tool_id
                for side in ARMS
            )
            if shared_tool:
                continue
            tool_actions = {
                side: [candidate for candidate in side_actions if candidate.get("object_id") == tool_id]
                for side, side_actions in action_by_arm.items()
            }
            if _paired_handover_receivers(tool_actions):
                continue
            errors.append(
                f"plan.arms.{arm}: cross_arm_tool_use: tool {tool_id!r} is explicitly "
                f"grasped/held only by the {other} arm, without a paired handover of that tool"
            )
    return errors


def _destination_norm(text):
    text = re.sub(r"\binto\b", "in", _norm(text))
    return " ".join(re.sub(r"\b(?:a|an|the)\b", " ", text).split())


def _recipient_is_named(text, phrases, receivers):
    if not re.search(r"\bother (?:hand|arm)\b", _norm(text)):
        return False
    joined = _norm(" ".join(phrases.values()))
    for receiver in receivers:
        if re.search(
            rf"\b(?:to|into) (?:the |its )?(?:robot's )?{receiver} "
            rf"(?:robot )?(?:arm|hand|gripper)\b", joined
        ):
            return True
        # The deterministic renderer supplies this receiving arm's subject.
        if re.search(r"\breceives?\b", _norm(phrases.get(receiver, ""))):
            return True
    return False


def _part_tokens(text):
    tokens = set(re.findall(r"\b(?:grip|handle|head)\b", _norm(text)))
    return {"handle" if value == "grip" else value for value in tokens}


def _required_task_parts(task):
    """Conservatively distinguish gripping a head from orienting a head."""
    text = _norm(task)
    found = set()
    # The task names a contact location, e.g. "by its handle".
    for match in re.finditer(
        r"\b(?:by|at|around|on) (?:the |its |their |a |an )?"
        r"(?:[\w'-]+ ){0,2}?(grip|handle|head)\b", text
    ):
        found.update(_part_tokens(match.group(1)))
    # Reaching for a named contact part before a handover still constrains grasp.
    for match in re.finditer(
        r"\breach for (?:the |its |a |an )?(?:[\w'-]+ ){0,3}?"
        r"(grip|handle|head)\b", text
    ):
        found.update(_part_tokens(match.group(1)))
    # A part is the direct object, e.g. "grasp the hammer's head".
    for match in re.finditer(
        r"\b(?:grasp|grab|grip|hold|take|lift|pick up) "
        r"(?:the |its |a |an )?"
        r"(?:(?!(?:by|at|around|on|with|from|to|toward|point|points|pointing)\b)[\w'-]+ ){0,6}?"
        r"(grip|handle|head)\b", text
    ):
        found.update(_part_tokens(match.group(1)))
    return found


def _phrase_errors(phrase, active, path):
    errors = []
    if not isinstance(phrase, str) or not phrase.strip():
        return [f"{path}: expected a nonempty verb phrase"]
    if phrase != phrase.strip():
        errors.append(f"{path}: leading/trailing whitespace is forbidden")
    if "\n" in phrase or "\r" in phrase or "@@" in phrase:
        errors.append(f"{path}: newlines and @@ are forbidden")
    if re.search(r"[.!?]$", phrase):
        errors.append(f"{path}: trailing sentence punctuation is forbidden")
    if re.search(r"\b(?:the )?(?:left|right) robot gripper\b", phrase, re.I) and re.match(
        r"(?:the )?(?:left|right) robot gripper\b", phrase, re.I
    ):
        errors.append(f"{path}: omit the arm subject")
    if "composed robot video" in phrase.lower() or ", while" in phrase.lower():
        errors.append(f"{path}: view suffix and arm connector belong to the renderer")
    if active is False and phrase != "keeps static":
        errors.append(f"{path}: inactive arm must say exactly 'keeps static'")
    if active is True and "keeps static" in _norm(phrase):
        errors.append(f"{path}: participating arm cannot say 'keeps static'")
    return errors


def validate_plan(plan, grounding, task):
    """Return all detected review flags without changing model output."""
    errors = []
    if not isinstance(plan, dict):
        return ["plan: expected an object"]
    if not isinstance(task, str) or not task.strip():
        return ["task: expected a nonempty string"]
    enriched, grounding_errors = enrich_grounding(grounding)
    errors.extend(grounding_errors)
    objects = {
        obj["id"]: obj for obj in enriched.get("objects", [])
        if isinstance(obj, dict) and isinstance(obj.get("id"), str)
    }
    arms = plan.get("arms")
    if not isinstance(arms, dict):
        return errors + ["plan.arms: expected an object with right and left"]
    if set(arms) != set(ARMS):
        errors.append("plan.arms: expected exactly right and left")
    action_by_arm = {}
    phrases = {}
    for arm in ARMS:
        path = f"plan.arms.{arm}"
        data = arms.get(arm)
        if not isinstance(data, dict):
            errors.append(f"{path}: expected an object")
            continue
        active = data.get("active")
        if type(active) is not bool:
            errors.append(f"{path}.active: expected a boolean")
        phrase = data.get("verb_phrase")
        errors.extend(_phrase_errors(phrase, active, f"{path}.verb_phrase"))
        phrases[arm] = phrase if isinstance(phrase, str) else ""
        assignment = data.get("assignment")
        if not isinstance(assignment, dict):
            errors.append(f"{path}.assignment: expected an object")
            assignment = {}
        mode = assignment.get("mode")
        if not isinstance(mode, str) or mode not in ASSIGNMENT_MODES:
            errors.append(f"{path}.assignment.mode: invalid mode")
            mode = None
        if (active is False and mode != "inactive") or (active is True and mode == "inactive"):
            errors.append(f"{path}.assignment: inactive mode must match active=false")
        assignment_id = assignment.get("object_id")
        if assignment_id is not None and (not isinstance(assignment_id, str) or assignment_id not in objects):
            errors.append(f"{path}.assignment.object_id: unknown object")
        evidence = assignment.get("evidence")
        if not isinstance(evidence, str):
            errors.append(f"{path}.assignment.evidence: expected a string")
        elif evidence and not _has_evidence(evidence, task):
            errors.append(f"{path}.assignment.evidence: must be a substring of the task")
        if mode == "explicit" and (
            not _has_evidence(evidence, task) or arm not in _explicit_arms(evidence)
        ):
            errors.append(f"{path}.assignment: explicit assignment needs matching arm evidence from task")
        if mode == "both" and (
            not _has_evidence(evidence, task) or not _both_evidence(evidence)
        ):
            errors.append(f"{path}.assignment: both assignment needs explicit two-arm task evidence")
        if mode == "proximity":
            obj = objects.get(assignment_id) if isinstance(assignment_id, str) else None
            preferred = obj.get("preferred_arm") if obj else None
            if preferred is None:
                errors.append(f"{path}.assignment: proximity heuristic is unresolved for this object")
            elif preferred != arm:
                errors.append(f"{path}.assignment: proximity heuristic conflicts with observer mapping; expected {preferred}")
        actions = data.get("actions")
        if not isinstance(actions, list):
            errors.append(f"{path}.actions: expected a list")
            actions = []
        action_by_arm[arm] = []
        if active is True and not actions:
            errors.append(f"{path}.actions: active arm needs at least one action")
        if active is False and actions:
            errors.append(f"{path}.actions: inactive arm must have no actions")
        for index, action in enumerate(actions):
            action_path = f"{path}.actions[{index}]"
            if not isinstance(action, dict):
                errors.append(f"{action_path}: expected an object")
                continue
            action_by_arm[arm].append(action)
            kind = action.get("kind")
            if not isinstance(kind, str) or kind not in ACTION_KINDS:
                errors.append(f"{action_path}.kind: invalid action kind")
                kind = None
            for field in ("object_id", "tool_id", "destination_id"):
                value = action.get(field)
                if value is None and field == "object_id":
                    errors.append(f"{action_path}.{field}: an object id is required")
                elif value is not None and (not isinstance(value, str) or value not in objects):
                    errors.append(f"{action_path}.{field}: unknown object id")
            part = action.get("grasp_part")
            if part is not None and (not isinstance(part, str) or not part.strip()):
                errors.append(f"{action_path}.grasp_part: expected null or a nonempty string")
            other = action.get("other_arm")
            if other not in (None, "left", "right") or other == arm:
                errors.append(f"{action_path}.other_arm: expected the opposite arm or null")
            if kind in ("transfer", "receive") and other != ("left" if arm == "right" else "right"):
                errors.append(f"{action_path}: transfer/receive must identify the opposite arm")
            if kind not in ("transfer", "receive") and other is not None:
                errors.append(f"{action_path}.other_arm: only transfer/receive actions may identify another arm")
        if mode == "receiver" and not any(a.get("kind") == "receive" for a in action_by_arm[arm]):
            errors.append(f"{path}.assignment: receiver mode needs a receive action")
        if mode == "both":
            other_arm = "left" if arm == "right" else "right"
            other_data = arms.get(other_arm)
            other_assignment = other_data.get("assignment", {}) if isinstance(other_data, dict) else {}
            if (
                not isinstance(other_data, dict) or other_data.get("active") is not True
                or not isinstance(other_assignment, dict)
                or not isinstance(assignment_id, str)
                or other_assignment.get("object_id") != assignment_id
            ):
                errors.append(f"{path}.assignment: both mode requires two active arms assigned to the same shared object")
    errors.extend(_cross_arm_tool_flags(arms, action_by_arm))
    # A handover is a paired interaction involving the same physical object.
    for arm, actions in action_by_arm.items():
        other = "left" if arm == "right" else "right"
        for action in actions:
            action_kind = action.get("kind")
            counterpart = {"transfer": "receive", "receive": "transfer"}.get(action_kind) if isinstance(action_kind, str) else None
            if counterpart and not any(
                candidate.get("kind") == counterpart
                and candidate.get("object_id") == action.get("object_id")
                and candidate.get("other_arm") == arm
                for candidate in action_by_arm.get(other, [])
            ):
                errors.append(f"plan.arms.{arm}: {action.get('kind')} lacks a matching {other}-arm {counterpart}")
    explicit = _explicit_arms(task)
    handover_task = _requires_handover(task)
    multi_arm_task = _both_evidence(task) or handover_task
    if len(explicit) == 1 and not multi_arm_task:
        opposite = (set(ARMS) - explicit).pop()
        if isinstance(arms.get(opposite), dict) and arms[opposite].get("active") is True:
            errors.append(f"plan.arms.{opposite}: task specifies only the other arm; extra participation needs review")
    handover_receivers = _paired_handover_receivers(action_by_arm)
    if handover_task:
        if not all(isinstance(arms.get(arm), dict) and arms[arm].get("active") is True for arm in ARMS):
            errors.append("plan.arms: handover task requires both arms to participate")
        if not handover_receivers:
            errors.append("plan.arms: handover task requires paired transfer/receive actions for the same object")
    requirements = plan.get("requirements")
    if not isinstance(requirements, dict):
        errors.append("plan.requirements: expected an object")
        requirements = {}
    joined_phrase = " ".join(phrases.values())
    errors.extend(_action_family_flags(task, phrases.values()))
    errors.extend(_rgb_object_coverage_flags(task, objects, arms, action_by_arm, phrases))
    for object_id in objects:
        id_pattern = rf"(?<!\w){re.escape(object_id)}(?!\w)"
        if re.search(id_pattern, task, re.I):
            continue
        for arm, phrase in phrases.items():
            if re.search(id_pattern, phrase, re.I):
                errors.append(f"plan.arms.{arm}: internal_object_id_in_caption: {object_id!r}")
    declared_parts = set()
    for field in ("grasp_parts", "destinations", "final_states"):
        values = requirements.get(field)
        if not isinstance(values, list):
            errors.append(f"plan.requirements.{field}: expected a list")
            continue
        for index, value in enumerate(values):
            path = f"plan.requirements.{field}[{index}]"
            if not isinstance(value, dict):
                errors.append(f"{path}: expected an object")
                continue
            text = value.get("text")
            evidence = value.get("evidence")
            if not isinstance(text, str) or not text.strip():
                errors.append(f"{path}.text: expected nonempty text")
                continue
            if not _has_evidence(evidence, task):
                errors.append(f"{path}.evidence: must be a nonempty substring of the task")
            if field == "grasp_parts":
                parts = _part_tokens(text)
                declared_parts.update(parts)
                if parts:
                    if not parts <= _part_tokens(joined_phrase):
                        errors.append(f"{path}: grasp part omitted from caption")
                elif _norm(text) not in _norm(joined_phrase):
                    errors.append(f"{path}: grasp part omitted from caption")
                action_parts = " ".join(
                    a.get("grasp_part") or "" for actions in action_by_arm.values()
                    for a in actions if isinstance(a.get("grasp_part"), (str, type(None)))
                )
                if (parts and not parts <= _part_tokens(action_parts)) or (
                    not parts and _norm(text) not in _norm(action_parts)
                ):
                    errors.append(f"{path}: grasp part omitted from structured actions")
            elif field == "destinations":
                destination = _destination_norm(text)
                present = bool(destination) and destination in _destination_norm(joined_phrase)
                if not present and not _recipient_is_named(text, phrases, handover_receivers):
                    errors.append(f"{path}: required text omitted from caption")
            elif _norm(text) not in _norm(joined_phrase):
                errors.append(f"{path}: required text omitted from caption")
    missing_parts = _required_task_parts(task) - declared_parts
    if missing_parts:
        errors.append(f"plan.requirements.grasp_parts: task contact part omitted: {', '.join(sorted(missing_parts))}")
    timing = requirements.get("timing")
    if not isinstance(timing, dict):
        errors.append("plan.requirements.timing: expected an object")
        timing = {}
    kind = timing.get("kind")
    if kind not in ("simultaneous", "sequential", "unspecified"):
        errors.append("plan.requirements.timing.kind: invalid kind")
    evidence = timing.get("evidence")
    if not isinstance(evidence, str) or (evidence and not _has_evidence(evidence, task)):
        errors.append("plan.requirements.timing.evidence: must be empty or a task substring")
    if kind in ("simultaneous", "sequential") and not _has_evidence(evidence, task):
        errors.append("plan.requirements.timing.evidence: explicit timing needs task evidence")
    task_simultaneous = bool(re.search(r"\b(?:together|simultaneously|at the same time)\b", _norm(task)))
    if task_simultaneous and kind != "simultaneous":
        errors.append("plan.requirements.timing: task requires simultaneous actions")
    if (task_simultaneous or kind == "simultaneous") and not re.search(
        r"\b(?:together|simultaneously|at the same time)\b", _norm(joined_phrase)
    ):
        errors.append("plan.arms: caption omits explicit simultaneous timing")
    _uncertainties(plan.get("uncertainties"), "plan.uncertainties", errors)
    return list(dict.fromkeys(errors))


def render_caption(plan):
    """Render model verb phrases verbatim; reject malformed phrases."""
    phrases = []
    for arm in ARMS:
        try:
            data = plan["arms"][arm]
            phrase = data["verb_phrase"]
            active = data["active"]
        except (TypeError, KeyError) as exc:
            raise ValueError(f"Missing {arm} arm data") from exc
        errors = _phrase_errors(phrase, active, f"plan.arms.{arm}.verb_phrase")
        if type(active) is not bool:
            errors.append(f"plan.arms.{arm}.active: expected a boolean")
        if errors:
            raise ValueError("; ".join(errors))
        phrases.append(phrase)
    return (
        f"The right robot gripper {phrases[0]}, while the left robot gripper "
        f"{phrases[1]}. {VIEW_SUFFIX}"
    )
