"""Run the RepNet model on a given video."""
import os
import cv2
import argparse
import torch
import torchvision.transforms as T
import numpy as np

from repnet import utils, plots
from repnet.model import RepNet


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_VISUALIZATIONS_DIR = os.path.join(PROJECT_ROOT, 'visualizations')
SAMPLE_VIDEOS_URLS = [
    'https://www.youtube.com/watch?v=5EYY2J3nb5c', # Cooking
    'https://imgur.com/t/hummingbird/m2e2Nfa', # Hummingbird
    'https://www.reddit.com/r/gifs/comments/4qfif6/cheetah_running_at_63_mph_102_kph', # Cheetah
    'https://www.youtube.com/watch?v=-Q3_7T5w4nE', # Excersise 1 
    'https://www.youtube.com/watch?v=5g1T-ff07kM', # Excersise 2
]

# Script arguments
parser = argparse.ArgumentParser(description='Run the RepNet model on a given video.')
parser.add_argument('--weights', type=str, default=os.path.join(PROJECT_ROOT, 'checkpoints', 'pytorch_weights.pth'), help='Path to the model weights (default: %(default)s).')
parser.add_argument('--sample', type=str, default=SAMPLE_VIDEOS_URLS[0], help='Video to test the model on, either a YouTube/http/local path (default: %(default)s).')
parser.add_argument('--stride', type=int, default=1, help='Temporal stride to use when testing on the sample video (default: %(default)s).')
parser.add_argument('--device', type=str, default='cpu', help='Device to use for inference (default: %(default)s).')


if __name__ == '__main__':
    args = parser.parse_args()

    # Download the video sample if needed
    print(f'Downloading {args.sample}...')
    video_path = os.path.join(PROJECT_ROOT, 'videos', os.path.basename(args.sample) + '.mp4')
    if not os.path.exists(video_path):
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        utils.download_file(args.sample, video_path)

    # Read frames and apply preprocessing
    print(f'Reading video file and pre-processing frames...')
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=0.5, std=0.5),
    ])
    cap = cv2.VideoCapture(video_path)
    fps, width, height = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_frames, frames = [], []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        raw_frames.append(frame)
        frame = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(frame)
    cap.release()

    # Apply stride and limit the number of frames to 64 multiples
    stride = np.clip(args.stride, 1, len(frames) // 64)
    frames = frames[::stride]
    assert len(frames) >= 64, 'The video is too short, at least 64 frames are needed.'
    frames = frames[:(len(frames) // 64) * 64]
    frames = torch.stack(frames, axis=0).unflatten(0, (-1, 64)).movedim(1, 2) # Convert to N x C x D x H x W

    # Load model
    model = RepNet()
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict)
    model.eval()

    # Get counts
    print(f'Running inference on {args.sample}...')
    model, frames = model.to(args.device), frames.to(args.device)
    period_length, periodicity_score, embeddings = [], [], []
    with torch.no_grad():
        for i in range(frames.shape[0]):  # Process each batch separately to avoid OOM
            batch_period_length, batch_periodicity, batch_embeddings = model(frames[i].unsqueeze(0))
            period_length.append(batch_period_length[0].cpu())
            periodicity_score.append(batch_periodicity[0].cpu())
            embeddings.append(batch_embeddings[0].cpu())
    period_length, periodicity_score, embeddings = torch.cat(period_length), torch.cat(periodicity_score), torch.cat(embeddings)
    period_count, periodicity_score = model.get_counts(period_length, periodicity_score, stride)

    # Generate plots and videos
    print(f'Save plots and video with counts to {OUT_VISUALIZATIONS_DIR}...')
    os.makedirs(OUT_VISUALIZATIONS_DIR, exist_ok=True)
    dist = torch.cdist(embeddings, embeddings, p=2)**2
    tsm_img = plots.plot_heatmap(dist.numpy(), log_scale=True)
    cv2.imwrite(os.path.join(OUT_VISUALIZATIONS_DIR, 'tsm.png'), tsm_img)
    rep_frames = plots.plot_repetitions(raw_frames[:len(period_count)], period_count.tolist())
    video = cv2.VideoWriter(os.path.join(OUT_VISUALIZATIONS_DIR, 'repetitions.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in rep_frames:
        video.write(frame)
    video.release()

    print('Done')
