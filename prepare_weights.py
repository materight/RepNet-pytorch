"""Script to download th epre-trained tensorflow weights and convert them to pytorch weights."""
import os
import cv2
import torch
import torchvision.transforms as T
import numpy as np
from tensorflow.python.training import py_checkpoint_reader

from repnet import utils
from repnet.model import RepNet

# Relevant paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_BASE_URL = 'https://storage.googleapis.com/repnet_ckpt'
CHECKPOINT_FILES = ['checkpoint', 'ckpt-88.data-00000-of-00002', 'ckpt-88.data-00001-of-00002', 'ckpt-88.index']
SAMPLE_VIDEO_URL = 'https://www.youtube.com/watch?v=-Q3_7T5w4nE'

# Mapping of ndim -> permutation to go from tf to pytorch
WEIGHTS_PERMUTATION = {
    2: (1, 0),
    4: (3, 2, 0, 1),
    5: (4, 3, 0, 1, 2)
}

# Mapping of tf attributes -> pytorch attributes
ATTR_MAPPING = {
    'kernel':'weight',
    'bias': 'bias',
    'beta': 'bias',
    'gamma': 'weight',
    'moving_mean': 'running_mean',
    'moving_variance': 'running_var'
}

# Mapping of tf checkpoint -> tf model -> pytorch model
WEIGHTS_MAPPING = [
    # Base frame encoder
    ('base_model.layer-2',                'encoder.stem.conv'),
    ('base_model.layer-5',                'encoder.stages.0.blocks.0.norm1'),
    ('base_model.layer-7',                'encoder.stages.0.blocks.0.conv1'),
    ('base_model.layer-8',                'encoder.stages.0.blocks.0.norm2'),
    ('base_model.layer_with_weights-4',   'encoder.stages.0.blocks.0.conv2'),
    ('base_model.layer_with_weights-5',   'encoder.stages.0.blocks.0.norm3'),
    ('base_model.layer_with_weights-6',   'encoder.stages.0.blocks.0.downsample.conv'),
    ('base_model.layer_with_weights-7',   'encoder.stages.0.blocks.0.conv3'),
    ('base_model.layer_with_weights-8',   'encoder.stages.0.blocks.1.norm1'),
    ('base_model.layer_with_weights-9',   'encoder.stages.0.blocks.1.conv1'),
    ('base_model.layer_with_weights-10',  'encoder.stages.0.blocks.1.norm2'),
    ('base_model.layer_with_weights-11',  'encoder.stages.0.blocks.1.conv2'),
    ('base_model.layer_with_weights-12',  'encoder.stages.0.blocks.1.norm3'),
    ('base_model.layer_with_weights-13',  'encoder.stages.0.blocks.1.conv3'),
    ('base_model.layer_with_weights-14',  'encoder.stages.0.blocks.2.norm1'),
    ('base_model.layer_with_weights-15',  'encoder.stages.0.blocks.2.conv1'),
    ('base_model.layer_with_weights-16',  'encoder.stages.0.blocks.2.norm2'),
    ('base_model.layer_with_weights-17',  'encoder.stages.0.blocks.2.conv2'),
    ('base_model.layer_with_weights-18',  'encoder.stages.0.blocks.2.norm3'),
    ('base_model.layer_with_weights-19',  'encoder.stages.0.blocks.2.conv3'),
    ('base_model.layer_with_weights-20',  'encoder.stages.1.blocks.0.norm1'),
    ('base_model.layer_with_weights-21',  'encoder.stages.1.blocks.0.conv1'),
    ('base_model.layer_with_weights-22',  'encoder.stages.1.blocks.0.norm2'),
    ('base_model.layer_with_weights-23',  'encoder.stages.1.blocks.0.conv2'),
    ('base_model.layer-47',               'encoder.stages.1.blocks.0.norm3'),
    ('base_model.layer_with_weights-25',  'encoder.stages.1.blocks.0.downsample.conv'),
    ('base_model.layer_with_weights-26',  'encoder.stages.1.blocks.0.conv3'),
    ('base_model.layer_with_weights-27',  'encoder.stages.1.blocks.1.norm1'),
    ('base_model.layer_with_weights-28',  'encoder.stages.1.blocks.1.conv1'),
    ('base_model.layer_with_weights-29',  'encoder.stages.1.blocks.1.norm2'),
    ('base_model.layer_with_weights-30',  'encoder.stages.1.blocks.1.conv2'),
    ('base_model.layer_with_weights-31',  'encoder.stages.1.blocks.1.norm3'),
    ('base_model.layer-61',               'encoder.stages.1.blocks.1.conv3'),
    ('base_model.layer-63',               'encoder.stages.1.blocks.2.norm1'),
    ('base_model.layer-65',               'encoder.stages.1.blocks.2.conv1'),
    ('base_model.layer-66',               'encoder.stages.1.blocks.2.norm2'),
    ('base_model.layer-69',               'encoder.stages.1.blocks.2.conv2'),
    ('base_model.layer-70',               'encoder.stages.1.blocks.2.norm3'),
    ('base_model.layer_with_weights-38',  'encoder.stages.1.blocks.2.conv3'),
    ('base_model.layer-74',               'encoder.stages.1.blocks.3.norm1'),
    ('base_model.layer_with_weights-40',  'encoder.stages.1.blocks.3.conv1'),
    ('base_model.layer_with_weights-41',  'encoder.stages.1.blocks.3.norm2'),
    ('base_model.layer_with_weights-42',  'encoder.stages.1.blocks.3.conv2'),
    ('base_model.layer_with_weights-43',  'encoder.stages.1.blocks.3.norm3'),
    ('base_model.layer_with_weights-44',  'encoder.stages.1.blocks.3.conv3'),
    ('base_model.layer_with_weights-45',  'encoder.stages.2.blocks.0.norm1'),
    ('base_model.layer_with_weights-46',  'encoder.stages.2.blocks.0.conv1'),
    ('base_model.layer_with_weights-47',  'encoder.stages.2.blocks.0.norm2'),
    ('base_model.layer-92',               'encoder.stages.2.blocks.0.conv2'),
    ('base_model.layer-93',               'encoder.stages.2.blocks.0.norm3'),
    ('base_model.layer-95',               'encoder.stages.2.blocks.0.downsample.conv'),
    ('base_model.layer-96',               'encoder.stages.2.blocks.0.conv3'),
    ('base_model.layer-98',               'encoder.stages.2.blocks.1.norm1'),
    ('base_model.layer-100',              'encoder.stages.2.blocks.1.conv1'),
    ('base_model.layer-101',              'encoder.stages.2.blocks.1.norm2'),
    ('base_model.layer-104',              'encoder.stages.2.blocks.1.conv2'),
    ('base_model.layer-105',              'encoder.stages.2.blocks.1.norm3'),
    ('base_model.layer-107',              'encoder.stages.2.blocks.1.conv3'),
    ('base_model.layer-109',              'encoder.stages.2.blocks.2.norm1'),
    ('base_model.layer-111',              'encoder.stages.2.blocks.2.conv1'),
    ('base_model.layer-112',              'encoder.stages.2.blocks.2.norm2'),
    ('base_model.layer-115',              'encoder.stages.2.blocks.2.conv2'),
    ('base_model.layer-116',              'encoder.stages.2.blocks.2.norm3'),
    ('base_model.layer-118',              'encoder.stages.2.blocks.2.conv3'),
    # Temporal convolution
    ('temporal_conv_layers.0',            'temporal_conv.0'),
    ('temporal_bn_layers.0',              'temporal_conv.1'),
    ('conv_3x3_layer',                    'tsm_conv.0'),
    # Period length head
    ('input_projection',                  'period_length_head.0.input_projection'),
    ('pos_encoding',                      'period_length_head.0.pos_encoding'),
    ('transformer_layers.0.ffn.layer-0',  'period_length_head.0.transformer_layer.linear1'),
    ('transformer_layers.0.ffn.layer-1',  'period_length_head.0.transformer_layer.linear2'),
    ('transformer_layers.0.layernorm1',   'period_length_head.0.transformer_layer.norm1'),
    ('transformer_layers.0.layernorm2',   'period_length_head.0.transformer_layer.norm2'),
    ('transformer_layers.0.mha.w_weight', 'period_length_head.0.transformer_layer.self_attn.in_proj_weight'),
    ('transformer_layers.0.mha.w_bias',   'period_length_head.0.transformer_layer.self_attn.in_proj_bias'),
    ('transformer_layers.0.mha.dense',    'period_length_head.0.transformer_layer.self_attn.out_proj'),
    ('fc_layers.0',                       'period_length_head.1'),
    ('fc_layers.1',                       'period_length_head.3'),
    ('fc_layers.2',                       'period_length_head.5'),
    # Periodicity head
    ('input_projection2',                 'periodicity_head.0.input_projection'),
    ('pos_encoding2',                     'periodicity_head.0.pos_encoding'),
    ('transformer_layers2.0.ffn.layer-0', 'periodicity_head.0.transformer_layer.linear1'),
    ('transformer_layers2.0.ffn.layer-1', 'periodicity_head.0.transformer_layer.linear2'),
    ('transformer_layers2.0.layernorm1',  'periodicity_head.0.transformer_layer.norm1'),
    ('transformer_layers2.0.layernorm2',  'periodicity_head.0.transformer_layer.norm2'),
    ('transformer_layers2.0.mha.w_weight','periodicity_head.0.transformer_layer.self_attn.in_proj_weight'),
    ('transformer_layers2.0.mha.w_bias',  'periodicity_head.0.transformer_layer.self_attn.in_proj_bias'),
    ('transformer_layers2.0.mha.dense',   'periodicity_head.0.transformer_layer.self_attn.out_proj'),
    ('within_period_fc_layers.0',         'periodicity_head.1'),
    ('within_period_fc_layers.1',         'periodicity_head.3'),
    ('within_period_fc_layers.2',         'periodicity_head.5'),
]


if __name__ == '__main__':
    # Download checkpoints
    print('Downloading checkpoints...')
    checkpoints_dir = os.path.join(PROJECT_ROOT, 'checkpoints', 'tf_checkpoint')
    os.makedirs(checkpoints_dir, exist_ok=True)
    for file in CHECKPOINT_FILES:
        dst = os.path.join(checkpoints_dir, file)
        if not os.path.exists(dst):
            utils.download_file(f'{CHECKPOINT_BASE_URL}/{file}', dst)

    # Load tensorflow weights into a dictionary
    print('Loading tensorflow checkpoint...')
    checkpoint_path = os.path.join(checkpoints_dir, 'ckpt-88')
    checkpoint_reader = py_checkpoint_reader.NewCheckpointReader(checkpoint_path)
    shape_map = checkpoint_reader.get_variable_to_shape_map()
    tf_state_dict = {}
    for var_name in sorted(shape_map.keys()):
        var_tensor = checkpoint_reader.get_tensor(var_name)
        if not var_name.startswith('model') or '.OPTIMIZER_SLOT' in var_name:
            # Skip variables that are not part of the model
            continue
        # Split var_name into path
        var_path = var_name.split('/')[1:]  # Remove `model`` key from the path
        var_path = [p for p in var_path if p not in ['.ATTRIBUTES', 'VARIABLE_VALUE']]
        # Map weights into a nested dictionary
        current_dict = tf_state_dict
        for path in var_path[:-1]:
            current_dict = current_dict.setdefault(path, {})
        current_dict[var_path[-1]] = var_tensor

    # Merge transformer self-attention weights into a single tensor
    for k in ['transformer_layers', 'transformer_layers2']:
        v = tf_state_dict[k]['0']['mha']
        v['w_weight'] = np.concatenate([v['wq']['kernel'], v['wk']['kernel'], v['wv']['kernel']], axis=0)
        v['w_bias'] = np.concatenate([v['wq']['bias'], v['wk']['bias'], v['wv']['bias']], axis=0)
        del v['wk'], v['wq'], v['wv']
    tf_state_dict = utils.flatten_dict(tf_state_dict, keep_last=True)
    # Add missing final level for some weights
    for k, v in tf_state_dict.items():
        if not isinstance(v, dict):
            tf_state_dict[k] = {None: v}

    # Convert to a format compatible with PyTorch and save
    print('Converting to PyTorch format...')
    pt_state_dict = {}
    for k_tf, k_pt in WEIGHTS_MAPPING:
        assert k_pt not in pt_state_dict
        pt_state_dict[k_pt] = {}
        for attr in tf_state_dict[k_tf]:
            new_attr = ATTR_MAPPING.get(attr, attr)
            pt_state_dict[k_pt][new_attr] = torch.from_numpy(tf_state_dict[k_tf][attr])
            if attr == 'kernel':
                weights_permutation = WEIGHTS_PERMUTATION[pt_state_dict[k_pt][new_attr].ndim] # Permute weights if needed
                pt_state_dict[k_pt][new_attr] = pt_state_dict[k_pt][new_attr].permute(weights_permutation)
    pt_state_dict = utils.flatten_dict(pt_state_dict, skip_none=True)
    torch.save(pt_state_dict, os.path.join(checkpoints_dir, 'converted_weights.pth'))


    # Initialize the model and try to load the weights
    print('Loading weights into the model...')
    pt_model = RepNet()
    pt_state_dict = torch.load(os.path.join(checkpoints_dir, 'converted_weights.pth'))
    pt_model.load_state_dict(pt_state_dict)
    pt_model.eval()

    # Test the model on a sample video from countix
    print('Test the model output on sample video...')
    video_path = os.path.join(PROJECT_ROOT, 'videos', os.path.basename(SAMPLE_VIDEO_URL) + '.mp4')
    if not os.path.exists(video_path):
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        utils.download_file(SAMPLE_VIDEO_URL, video_path)

    # Read frames
    stride = 5
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=0.5, std=0.5),
    ])
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        frames.append(frame)
    cap.release()

    # Get counts
    with torch.no_grad():
        frames = torch.from_numpy(np.stack(frames, axis=0)).float()
        frames = frames[::stride][:64].unsqueeze(0).movedim(1, 2)
        period_length, periodicity = pt_model(frames)
        period_length, period_length_conf, periodicity = pt_model.get_scores(period_length, periodicity)

    print('Done')
