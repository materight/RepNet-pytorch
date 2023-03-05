"""Run the RepNet model on a given video."""
import os
import cv2
import argparse
import torch
import torchvision.transforms as T

from repnet import utils, plots
from repnet.model import RepNet


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_VISUALIZATIONS_DIR = os.path.join(PROJECT_ROOT, 'visualizations')
SAMPLE_VIDEOS_URLS = [
    'https://imgur.com/t/hummingbird/m2e2Nfa', # Hummingbird
    'https://www.youtube.com/watch?v=w0JOoC-5_Lk', # Chopping
    'https://www.youtube.com/watch?v=t9OE3nxnI2Y', # Hammer training
    'https://www.youtube.com/watch?v=aY3TrpiUOqE', # Bouncing ball
    'https://www.youtube.com/watch?v=5EYY2J3nb5c', # Cooking
    'https://www.reddit.com/r/gifs/comments/4qfif6/cheetah_running_at_63_mph_102_kph', # Cheetah
    'https://www.youtube.com/watch?v=cMWb7NvWWuI', # Pendulum
    'https://www.youtube.com/watch?v=5g1T-ff07kM', # Excersise
    'https://www.youtube.com/watch?v=-Q3_7T5w4nE', # Excersise

]

# Script arguments
parser = argparse.ArgumentParser(description='Run the RepNet model on a given video.')
parser.add_argument('--weights', type=str, default=os.path.join(PROJECT_ROOT, 'checkpoints', 'pytorch_weights.pth'), help='Path to the model weights (default: %(default)s).')
parser.add_argument('--video', type=str, default=SAMPLE_VIDEOS_URLS[0], help='Video to test the model on, either a YouTube/http/local path (default: %(default)s).')
parser.add_argument('--strides', nargs='+', type=int, default=[1, 2, 3, 4, 8], help='Temporal strides to try when testing on the sample video (default: %(default)s).')
parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference (default: %(default)s).')
parser.add_argument('--no-score', action='store_true', help='If specified, do not plot the periodicity score.')

if __name__ == '__main__':
    args = parser.parse_args()

    # Download the video sample if needed
    print(f'Downloading {args.video}...')
    video_path = os.path.join(PROJECT_ROOT, 'videos', os.path.basename(args.video) + '.mp4')
    if not os.path.exists(video_path):
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        utils.download_file(args.video, video_path)

    # Read frames and apply preprocessing
    print(f'Reading video file and pre-processing frames...')
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=0.5, std=0.5),
    ])
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    raw_frames, frames = [], []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        raw_frames.append(frame)
        frame = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(frame)
    cap.release()

    # Load model
    model = RepNet()
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(args.device)

    # Test multiple strides and pick the best one
    print('Running inference on multiple stride values...')
    best_stride, best_confidence, best_period_length, best_period_count, best_periodicity_score, best_embeddings = None, None, None, None, None, None
    for stride in args.strides:
        # Apply stride
        stride_frames = frames[::stride]
        stride_frames = stride_frames[:(len(stride_frames) // 64) * 64]
        if len(stride_frames) < 64:
            continue # Skip this stride if there are not enough frames
        stride_frames = torch.stack(stride_frames, axis=0).unflatten(0, (-1, 64)).movedim(1, 2) # Convert to N x C x D x H x W
        stride_frames = stride_frames.to(args.device)
        # Run inference
        raw_period_length, raw_periodicity_score, embeddings = [], [], []
        with torch.no_grad():
            for i in range(stride_frames.shape[0]):  # Process each batch separately to avoid OOM
                batch_period_length, batch_periodicity, batch_embeddings = model(stride_frames[i].unsqueeze(0))
                raw_period_length.append(batch_period_length[0].cpu())
                raw_periodicity_score.append(batch_periodicity[0].cpu())
                embeddings.append(batch_embeddings[0].cpu())
        # Post-process results
        raw_period_length, raw_periodicity_score, embeddings = torch.cat(raw_period_length), torch.cat(raw_periodicity_score), torch.cat(embeddings)
        confidence, period_length, period_count, periodicity_score = model.get_counts(raw_period_length, raw_periodicity_score, stride)
        if best_confidence is None or confidence > best_confidence:
            best_stride, best_confidence, best_period_length, best_period_count, best_periodicity_score, best_embeddings = stride, confidence, period_length, period_count, periodicity_score, embeddings
    if best_stride is None:
        raise RuntimeError('The stride values used are too large and nove 64 video chunk could be sampled. Try different values for --strides.')
    print(f'Predicted a period length of {best_period_length/fps:.1f} seconds (~{int(best_period_length)} frames) with a confidence of {best_confidence:.2f} using a stride of {best_stride} frames.')

    # Generate plots and videos
    print(f'Save plots and video with counts to {OUT_VISUALIZATIONS_DIR}...')
    os.makedirs(OUT_VISUALIZATIONS_DIR, exist_ok=True)
    dist = torch.cdist(best_embeddings, best_embeddings, p=2)**2
    tsm_img = plots.plot_heatmap(dist.numpy(), log_scale=True)
    pca_img = plots.plot_pca(best_embeddings.numpy())
    cv2.imwrite(os.path.join(OUT_VISUALIZATIONS_DIR, 'tsm.png'), tsm_img)
    cv2.imwrite(os.path.join(OUT_VISUALIZATIONS_DIR, 'pca.png'), pca_img)

    # Generate video with counts
    rep_frames = plots.plot_repetitions(raw_frames[:len(best_period_count)], best_period_count.tolist(), best_periodicity_score.tolist() if not args.no_score else None)
    video = cv2.VideoWriter(os.path.join(OUT_VISUALIZATIONS_DIR, 'repetitions.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, rep_frames[0].shape[:2][::-1])
    for frame in rep_frames:
        video.write(frame)
    video.release()

    print('Done')
