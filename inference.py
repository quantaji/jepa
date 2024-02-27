import os
import sys

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu

sys.path.append(".")

import src.models.vision_transformer as vit
from evals.video_classification_frozen.eval import init_model
from evals.video_classification_frozen.utils import ClipAggregation, FrameAggregation, make_transforms
from src.models.attentive_pooler import AttentiveClassifier


def load_video_decord(file_pth, clip_start_frame, clip_end_frame):
    # code modified from src/datasets/video_dataset.py
    # config
    num_partitions = 8
    frames_per_partition = 16
    frame_step = 4  # frame_step
    random_clip_sampling = True
    allow_partitions_overlap = (True,)
    resolution = 224

    fpc = frames_per_partition
    fstp = frame_step

    vr = VideoReader(file_pth, num_threads=-1, ctx=cpu(0))
    vr.seek(0)

    partition_len = (clip_end_frame - clip_start_frame) // num_partitions

    clip_len = int(fpc * fstp)

    all_indices, partitions_indices = [], []
    for i in range(num_partitions):
        if partition_len > clip_len:
            # If partition_len > clip len, then sample a random window of
            # clip_len frames within the partitions
            end_indx = clip_end_frame
            if random_clip_sampling:
                end_indx = np.random.randint(clip_len, partition_len)
            start_indx = end_indx - clip_len
            indices = np.linspace(start_indx, end_indx, num=fpc)
            indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
            # --
            indices = indices + i * partition_len + clip_start_frame

        else:
            # If partition overlap not allowed and partition_len < clip_len
            # then repeatedly append the last frame in the partitions until
            # we reach the desired clip length
            if not allow_partitions_overlap:
                indices = np.linspace(0, partition_len, num=partition_len // fstp)
                indices = np.concatenate(
                    (
                        indices,
                        np.ones(fpc - partition_len // fstp) * partition_len,
                    )
                )
                indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
                # --
                indices = indices + i * partition_len + clip_start_frame

            # If partition overlap is allowed and partition_len < clip_len
            # then start_indx of partition i+1 will lie within partition i
            else:
                sample_len = min(clip_len, len(vr)) - 1
                indices = np.linspace(0, sample_len, num=sample_len // fstp)
                indices = np.concatenate(
                    (
                        indices,
                        np.ones(fpc - sample_len // fstp) * sample_len,
                    )
                )
                indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
                # --
                clip_step = 0
                if len(vr) > clip_len:
                    clip_step = (len(vr) - clip_len) // (num_partitions - 1)
                indices = indices + i * clip_step + clip_start_frame

        partitions_indices.append(indices)
        all_indices.extend(list(indices))

    buffer = vr.get_batch(all_indices).asnumpy()

    def split_into_clips(video):
        """Split video into a list of clips"""
        return [video[i * fpc : (i + 1) * fpc] for i in range(num_partitions)]

    transform = make_transforms(
        training=False,
        num_views_per_clip=1,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=resolution,
        normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )

    buffer = [transform(np.expand_dims(clip, axis=0)) for clip in buffer]

    # buffer now is a list of torch tensor of shape (C, 1, H, W)

    return buffer, partitions_indices


def load_pretrained(encoder, pretrained, checkpoint_key="target_encoder"):
    checkpoint = torch.load(pretrained, map_location="cpu")
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint["encoder"]

    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}

    del checkpoint
    return encoder


def init_model(
    device,
    pretrained,
    model_name,
    patch_size=16,
    crop_size=224,
    # Video specific parameters
    frames_per_clip=16,
    tubelet_size=2,
    use_sdpa=False,
    use_SiLU=False,
    tight_SiLU=True,
    uniform_power=False,
    checkpoint_key="target_encoder",
):
    encoder = vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
    )

    encoder.to(device)
    encoder = load_pretrained(encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key)
    return encoder


def load_model(
    device: str,
    encoder_ckpt: str,
    classifier_ckpt: str,
    checkpoint_key="target_encoder",
    resolution=384,
    patch_size=16,
    tubelet_size=2,
    model_name="vit_huge",
    pretrain_frames_per_clip=1,
    uniform_power=True,
    use_SiLU=False,
    tight_SiLU=False,
    use_sdpa=True,
    attend_across_segments=True,
    num_classes=174,
):
    encoder = init_model(
        crop_size=resolution,
        device=device,
        pretrained=encoder_ckpt,
        model_name=model_name,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        frames_per_clip=pretrain_frames_per_clip,
        uniform_power=uniform_power,
        checkpoint_key=checkpoint_key,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=use_sdpa,
    )

    if pretrain_frames_per_clip == 1:
        # Process each frame independently and aggregate
        encoder = FrameAggregation(encoder).to(device)
    else:
        # Process each video clip independently and aggregate
        encoder = ClipAggregation(encoder, tubelet_size=tubelet_size, attend_across_segments=attend_across_segments).to(device)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    num_classes = 174
    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=encoder.num_heads,
        depth=1,
        num_classes=num_classes,
    ).to(device)

    classifier_state_dict = torch.load(classifier_ckpt)
    restore_dict = {}
    for key in classifier_state_dict["classifier"].keys():
        restore_dict[key.replace("module.", "")] = classifier_state_dict["classifier"][key]

    classifier.load_state_dict(restore_dict)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    return encoder, classifier


def inference_video(
    video_pth,
    clip_start_frame,
    clip_end_frame,
    encoder,
    classifier,
    attend_across_segments,
):
    buffer, partitions_indices = load_video_decord(
        video_pth,
        clip_start_frame,
        clip_end_frame,
    )

    clips = [[dij.to(encoder.device, non_blocking=True).unsqueeze(0) for dij in di] for di in buffer]

    clip_indices = [torch.from_numpy(d).to(encoder.device, non_blocking=True) for d in partitions_indices]

    with torch.no_grad():
        outputs = encoder(clips, clip_indices)

    if attend_across_segments:
        outputs = [classifier(o) for o in outputs]
    else:
        outputs = [[classifier(ost) for ost in os] for os in outputs]

    index = outputs[0].reshape(-1).argmax().item()

    return index
