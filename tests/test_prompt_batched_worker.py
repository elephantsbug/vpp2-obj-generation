"""CPU protocol tests use the real local Qwen tokenizer/processor, never a model."""
from pathlib import Path
import sys

import pytest
from PIL import Image

sys.path.insert(0, '/data2/niurc/vpp2-obj-generation/scripts')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prompt_batched_worker import decode_batch_output, split_image_grids, trim_generated_tokens


@pytest.mark.parametrize('sequence,prefix,eos,pad,expected,terminated,padding', [
    ([151643, 151643, 44, 45, 70, 71, 151645, 151643, 151643], 4, [151645, 151643], 151643, [70, 71, 151645], True, 2),
    ([151643, 44, 45, 70, 151643, 151643], 3, [151645, 151643], 151643, [70, 151643], True, 1),
    ([99, 70, 0, 71], 1, [2], 0, [70, 0, 71], False, 0),
    ([99, 70, 71, 0], 1, [2], 0, [70, 71, 0], False, 0),
    ([99, 2, 0, 0], 1, 2, 0, [2], True, 2),
    ([99], 1, [2], 0, [], False, 0),
])
def test_eos_and_padding_keep_all_real_generated_tokens(sequence, prefix, eos, pad, expected, terminated, padding):
    result = trim_generated_tokens(sequence, prefix, eos, pad)
    assert result['token_ids'] == expected
    assert result['eos_terminated'] is terminated
    assert result['padding_after_eos'] == padding
    assert result['generated_tokens'] == len(expected)
    assert result['batch_generated_length'] == len(sequence) - prefix


def test_eos_never_silently_discards_nonpadding_tokens():
    with pytest.raises(ValueError, match='refusing to discard real output'):
        trim_generated_tokens([99, 70, 2, 0, 71], 1, [2], 0)
    with pytest.raises(ValueError, match='Invalid padded input'):
        trim_generated_tokens([99, 70], 3, [2], 0)


def test_image_grid_mapping_rejects_missing_or_extra_images():
    grids = [[1, 64, 64], [1, 32, 32], [1, 48, 48]]
    assert split_image_grids(grids, [2, 0, 1]) == [grids[:2], [], grids[2:]]
    with pytest.raises(ValueError, match='Image-grid count'):
        split_image_grids(grids, [2, 2])


@pytest.fixture(scope='module')
def real_processor():
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained('/data2/models/Qwen2.5-VL-72B-Instruct', local_files_only=True,
                                              min_pixels=448 * 448, max_pixels=896 * 896)
    processor.tokenizer.padding_side = 'left'
    return processor


def make_real_inputs(processor, with_images):
    prompts = ['Identify the target.', 'Describe only the target object, using image one to determine its observer coordinates.']
    image_counts = [2, 1] if with_images else [0, 0]
    chats = []
    for prompt, count in zip(prompts, image_counts):
        content = [{'type': 'image'} for _ in range(count)] + [{'type': 'text', 'text': prompt}]
        chats.append(processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True))
    images = [Image.new('RGB', (896, 896), 'red'), Image.new('RGB', (448, 448), 'green'), Image.new('RGB', (448, 448), 'blue')]
    kwargs = {'images': images} if with_images else {}
    inputs = processor(text=chats, padding=True, return_tensors='pt', **kwargs)
    expected = ['Grounding for sample zero.', 'Grounding for sample one with a longer action.']
    generated = [processor.tokenizer.encode(text, add_special_tokens=False) + [eos] for text, eos in zip(expected, [151645, 151643])]
    max_length = max(len(tokens) for tokens in generated)
    prefixes = inputs['input_ids'].tolist()
    sequences = [prefix + tokens + [151643] * (max_length - len(tokens)) for prefix, tokens in zip(prefixes, generated)]
    return inputs, image_counts, expected, generated, sequences


@pytest.mark.parametrize('with_images', [True, False])
def test_real_qwen_processor_preserves_batch_sample_response_and_image_order(real_processor, with_images):
    inputs, image_counts, expected, generated, sequences = make_real_inputs(real_processor, with_images)
    prefixes = inputs['input_ids'].tolist()
    masks = inputs['attention_mask'].tolist()
    flat_grids = inputs['image_grid_thw'].tolist() if with_images else []
    results = decode_batch_output(real_processor, prefixes, masks, sequences, [151645, 151643], 151643,
                                  flat_grids, image_counts, 0.25, 'cpu:ground' if with_images else 'cpu:plan')
    assert [result['raw_response'] for result in results] == expected
    assert [result['token_ids'] for result in results] == generated
    assert [result['batch_member_index'] for result in results] == [0, 1]
    assert [result['input_tokens'] for result in results] == inputs['attention_mask'].sum(dim=1).tolist()
    assert all(result['padded_input_tokens'] == inputs['input_ids'].shape[1] for result in results)
    assert any(result['input_tokens'] < result['padded_input_tokens'] for result in results)
    assert all(result['eos_terminated'] for result in results)
    assert results[0]['padding_after_eos'] > 0
    assert results[1]['padding_after_eos'] == 0
    if with_images:
        assert results[0]['image_grid_thw'] == [[1, 64, 64], [1, 32, 32]]
        assert results[1]['image_grid_thw'] == [[1, 32, 32]]
    else:
        assert [result['image_grid_thw'] for result in results] == [[], []]
    # An accidental response/prefix permutation is rejected rather than mislabeled.
    with pytest.raises(ValueError, match='padded input prefix'):
        decode_batch_output(real_processor, prefixes, masks, sequences[::-1], [151645, 151643], 151643,
                            flat_grids, image_counts, 0.25, 'wrong-order')


def test_right_padding_is_rejected_before_decoding(real_processor):
    with pytest.raises(ValueError, match='left padding'):
        decode_batch_output(real_processor, [[44, 45, 151643]], [[1, 1, 0]], [[44, 45, 151643, 70, 151645]],
                            [151645, 151643], 151643, [], [0], 0.1, 'bad-padding')
