import os
import json
import shutil
import logging
import argparse
import sys
from idlelib.colorizer import prog_group_name_to_tag
from pathlib import Path
from typing import Callable
from functools import partial
from math import ceil
from copy import deepcopy

import cv2
import h5py
import torch
import einops
import numpy as np
from PIL import Image
from numba.core.typing.builtins import Print
from tqdm import tqdm
from pprint import pformat
from tqdm.contrib.concurrent import process_map
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import (
    STATS_PATH,
    check_timestamps_sync,
    get_episode_data_index,
    serialize_dict,
    write_json,
)

# --------------------------------------------------------------------------------------------------------------------------- #
# hdf5:
# ['master', 'puppet', 'observations']
#       'master':             ['joint_position']
#       'puppet':             ['end_effector', 'joint_position']
#       'observations':       ['depth_images', 'rgb_images']
#               'rgb_images':         ['camera_left', 'camera_right', 'camera_top']
#               'depth_images':       ['camera_left', 'camera_right', 'camera_top']
# --------------------------------------------------------------------------------------------------------------------------- #

FEATURES = {
    "observation.images.camera_left": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.camera_right": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.camera_top": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.camera_left_depth": {
        "dtype": "image",
        "shape": [480, 640, 1],
        "names": ["height", "width", "channel"],
    },
    "observation.images.camera_right_depth": {
        "dtype": "image",
        "shape": [480, 640, 1],
        "names": ["height", "width", "channel"],
    },
    "observation.images.camera_top_depth": {
        "dtype": "image",
        "shape": [480, 640, 1],
        "names": ["height", "width", "channel"],
    },
    # "observation.images.camera_left_depth": {
    #     "dtype": "uint16",
    #     "shape": [480 * 640],
    #     "names": None,
    # },
    # "observation.images.camera_right_depth": {
    #     "dtype": "uint16",
    #     "shape": [480 * 640],
    #     "names": None,
    # },
    # "observation.images.camera_top_depth": {
    #     "dtype": "uint16",
    #     "shape": [480 * 640],
    #     "names": None,
    # },
    "observation.state": {
        "dtype": "float32",
        "shape": [8],
    },
    "action": {
        "dtype": "float32",
        "shape": [8],
    },
    "episode_index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "frame_index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "task_index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
}


def get_stats_einops_patterns(dataset, num_workers=0):
    """These einops patterns will be used to aggregate batches and compute statistics.

    Note: We assume the images are in channel first format
    """

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(dataloader))

    stats_patterns = {}

    for key in dataset.features:
        # sanity check that tensors are not float64
        assert batch[key].dtype != torch.float64

        # if isinstance(feats_type, (VideoFrame, Image)):
        if key in dataset.meta.camera_keys:
            # sanity check that images are channel first
            _, c, h, w = batch[key].shape
            assert c < h and c < w, f"expect channel first images, but instead {batch[key].shape}"

            # sanity check that images are float32 in range [0,1]
            assert batch[key].dtype == torch.float32, f"expect torch.float32, but instead {batch[key].dtype=}"
            # assert batch[key].max() <= 1, f"expect pixels lower than 1, but instead {batch[key].max()=}"
            # assert batch[key].min() >= 0, f"expect pixels greater than 1, but instead {batch[key].min()=}"

            stats_patterns[key] = "b c h w -> c 1 1"
        elif batch[key].ndim == 2:
            stats_patterns[key] = "b c -> c "
        elif batch[key].ndim == 1:
            stats_patterns[key] = "b -> 1"
        else:
            raise ValueError(f"{key}, {batch[key].shape}")

    return stats_patterns


def compute_stats(dataset, batch_size=8, num_workers=4, max_num_samples=None):
    """Compute mean/std and min/max statistics of all data keys in a LeRobotDataset."""
    if max_num_samples is None:
        max_num_samples = len(dataset)

    # for more info on why we need to set the same number of workers, see `load_from_videos`
    stats_patterns = get_stats_einops_patterns(dataset, num_workers)

    # mean and std will be computed incrementally while max and min will track the running value.
    mean, std, max, min = {}, {}, {}, {}
    for key in stats_patterns:
        mean[key] = torch.tensor(0.0).float()
        std[key] = torch.tensor(0.0).float()
        max[key] = torch.tensor(-float("inf")).float()
        min[key] = torch.tensor(float("inf")).float()

    def create_seeded_dataloader(dataset, batch_size, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        return dataloader

    # Note: Due to be refactored soon. The point of storing `first_batch` is to make sure we don't get
    # surprises when rerunning the sampler.
    first_batch = None
    running_item_count = 0  # for online mean computation
    dataloader = create_seeded_dataloader(dataset, batch_size, seed=1337)
    for i, batch in enumerate(
            tqdm(
                dataloader,
                total=ceil(max_num_samples / batch_size),
                desc="Compute mean, min, max",
            )
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        if first_batch is None:
            first_batch = deepcopy(batch)
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float()
            # Numerically stable update step for mean computation.
            batch_mean = einops.reduce(batch[key], pattern, "mean")
            # Hint: to update the mean we need x̄ₙ = (Nₙ₋₁x̄ₙ₋₁ + Bₙxₙ) / Nₙ, where the subscript represents
            # the update step, N is the running item count, B is this batch size, x̄ is the running mean,
            # and x is the current batch mean. Some rearrangement is then required to avoid risking
            # numerical overflow. Another hint: Nₙ₋₁ = Nₙ - Bₙ. Rearrangement yields
            # x̄ₙ = x̄ₙ₋₁ + Bₙ * (xₙ - x̄ₙ₋₁) / Nₙ
            mean[key] = (
                    mean[key]
                    + this_batch_size * (batch_mean - mean[key]) / running_item_count
            )
            max[key] = torch.maximum(
                max[key], einops.reduce(batch[key], pattern, "max")
            )
            min[key] = torch.minimum(
                min[key], einops.reduce(batch[key], pattern, "min")
            )

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    first_batch_ = None
    running_item_count = 0  # for online std computation
    dataloader = create_seeded_dataloader(dataset, batch_size, seed=1337)
    for i, batch in enumerate(
            tqdm(dataloader, total=ceil(max_num_samples / batch_size), desc="Compute std")
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        # Sanity check to make sure the batches are still in the same order as before.
        if first_batch_ is None:
            first_batch_ = deepcopy(batch)
            for key in stats_patterns:
                assert torch.equal(first_batch_[key], first_batch[key])
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float()
            # Numerically stable update step for mean computation (where the mean is over squared
            # residuals).See notes in the mean computation loop above.
            batch_std = einops.reduce((batch[key] - mean[key]) ** 2, pattern, "mean")
            std[key] = (
                    std[key] + this_batch_size * (batch_std - std[key]) / running_item_count
            )

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    for key in stats_patterns:
        std[key] = torch.sqrt(std[key])

    stats = {}
    for key in stats_patterns:
        stats[key] = {
            "mean": mean[key],
            "std": std[key],
            "max": max[key],
            "min": min[key],
        }
    return stats


class FrankaDataset(LeRobotDataset):
    def __init__(
            self,
            repo_id: str,
            root: str | Path | None = None,
            episodes: list[int] | None = None,
            image_transforms: Callable | None = None,
            delta_timestamps: dict[list[float]] | None = None,
            tolerance_s: float = 1e-4,
            download_videos: bool = True,
            local_files_only: bool = False,
            video_backend: str | None = None,
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            download_videos=download_videos,
            local_files_only=local_files_only,
            video_backend=video_backend,
        )

    def add_frame(self, frame: dict) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # TODO(aliberts, rcadene): Add sanity check for the input, check it's numpy or torch,
        # check the dtype and shape matches, etc.

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        frame_index = self.episode_buffer["size"]
        timestamp = (frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps)
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        for key in frame:
            if key not in self.features:
                raise ValueError(key)

            if self.features[key]["dtype"] not in ["video"]:
                item = frame[key].numpy() if isinstance(frame[key], torch.Tensor) else frame[key]
                self.episode_buffer[key].append(item)
            elif self.features[key]["dtype"] in ["video"]:
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"], image_key=key, frame_index=frame_index
                )
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_image(frame[key], img_path)
                self.episode_buffer[key].append(str(img_path))

        self.episode_buffer["size"] += 1

    def consolidate(self, run_compute_stats: bool = True, keep_image_files: bool = False) -> None:
        self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        check_timestamps_sync(self.hf_dataset, self.episode_data_index, self.fps, self.tolerance_s)

        if len(self.meta.video_keys) > 0:
            self.encode_videos()
            self.meta.write_video_info()

        if not keep_image_files:
            img_dir = self.root / "images"
            if img_dir.is_dir():
                shutil.rmtree(self.root / "images")

        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes

        if run_compute_stats:
            self.stop_image_writer()
            self.meta.stats = compute_stats(self)
            serialized_stats = serialize_dict(self.meta.stats)
            write_json(serialized_stats, self.root / STATS_PATH)
            self.consolidated = True
        else:
            logging.warning(
                "Skipping computation of the dataset statistics, dataset is not fully consolidated."
            )


if __name__ == '__main__':
    tgt_path = "/media/a/data1/dataset-mani/FrankaDataLerobot/"
    repo_id = "241223_upright_cup"
    # repo_id = "place_in_bread_on_plate_2"

    # --------------------------------------------------------------------------------------------------------------------------- #
    # 初始化 数据集
    # --------------------------------------------------------------------------------------------------------------------------- #
    dataset = FrankaDataset.create(
        repo_id=repo_id,
        root=f"{tgt_path}/{repo_id}",
        fps=12,
        robot_type="franka",
        features=FEATURES,
    )

    # [
    #     "master",
    #     "master/joint_position",
    #     "observations",
    #     "observations/depth_images",
    #     "observations/depth_images/camera_left",
    #     "observations/depth_images/camera_right",
    #     "observations/depth_images/camera_top",
    #     "observations/rgb_images",
    #     "observations/rgb_images/camera_left",
    #     "observations/rgb_images/camera_right",
    #     "observations/rgb_images/camera_top",
    #     "puppet",
    #     "puppet/end_effector",
    #     "puppet/joint_position"
    # ]

    src_root_path = "/media/a/data1/dataset-mani/FrankaDataOri/241223_upright_cup/"
    # src_root_path = "//media/a/data1/dataset-mani/FrankaDataOri/place_in_bread_on_plate_2/success_episodes/train/"

    # --------------------------------------------------------------------------------------------------------------------------- #
    # 开始批量转换
    # --------------------------------------------------------------------------------------------------------------------------- #
    for eidx, episode in enumerate(os.listdir(src_root_path)[:]):
        src_episode_path = os.path.join(src_root_path, episode, "data/", "trajectory.hdf5")

        # --------------------------------------------------------------------------------------------------------------------------- #
        # 读取当前episode的所有frame
        # --------------------------------------------------------------------------------------------------------------------------- #
        with h5py.File(src_episode_path, 'r') as file:
            # [t, 8]
            puppet_state = np.array(file["puppet/joint_position"])
            # [t, 6]
            puppet_effector = np.array(file["puppet/end_effector"])
            # [t, 8]
            master_state = np.array(file["master/joint_position"])
            # (t, 480, 640, 3)
            rgb_camera_left = np.stack([cv2.cvtColor(cv2.imdecode(img_compressed, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for img_compressed in file["observations/rgb_images/camera_left"]])
            # (t, 480, 640, 3)
            rgb_camera_right = np.stack([cv2.cvtColor(cv2.imdecode(img_compressed, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for img_compressed in file["observations/rgb_images/camera_right"]])
            # (t, 720, 1280, 3)
            rgb_camera_top = np.stack([cv2.resize(cv2.cvtColor(cv2.imdecode(img_compressed, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), (640, 480)) for img_compressed in file["observations/rgb_images/camera_top"]])
            # (t, 480, 640)
            depth_camera_left = np.stack([cv2.imdecode(img_compressed, cv2.IMREAD_UNCHANGED) for img_compressed in file["observations/depth_images/camera_left"]]) / 1000.0
            # (t, 480, 640)
            depth_camera_right = np.stack([cv2.imdecode(img_compressed, cv2.IMREAD_UNCHANGED) for img_compressed in file["observations/depth_images/camera_right"]]) / 1000.0
            # (t, 720, 1280)
            depth_camera_top = np.stack([cv2.resize(cv2.imdecode(img_compressed, cv2.IMREAD_UNCHANGED), (640, 480)) for img_compressed in file["observations/depth_images/camera_top"]]) / 1000.0

        # 0.0 -> 65.535
        # print(depth_camera_top.dtype)
        # print(np.min(depth_camera_top))
        # print(np.max(depth_camera_top))
        # sys.exit(1)
        # --------------------------------------------------------------------------------------------------------------------------- #
        # 对于每一个frame数据
        # --------------------------------------------------------------------------------------------------------------------------- #
        num_frames = len(puppet_state)
        for i in tqdm(range(num_frames), total=num_frames, position=eidx):
            frame_data = {
                "action": puppet_state[i],
                "observation.state": puppet_state[i],
                "observation.images.camera_left": rgb_camera_left[i],
                "observation.images.camera_right": rgb_camera_right[i],
                "observation.images.camera_top": rgb_camera_top[i],
                "observation.images.camera_left_depth": depth_camera_left[i],  # .flatten()
                "observation.images.camera_right_depth": depth_camera_right[i],
                "observation.images.camera_top_depth": depth_camera_top[i],
            }

            dataset.add_frame(frame_data)

        # --------------------------------------------------------------------------------------------------------------------------- #
        # save episode
        # --------------------------------------------------------------------------------------------------------------------------- #
        dataset.save_episode("upright_cup")

    # --------------------------------------------------------------------------------------------------------------------------- #
    # 最后统计
    # --------------------------------------------------------------------------------------------------------------------------- #
    dataset.consolidate()
