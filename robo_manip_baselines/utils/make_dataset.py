import sys
import os
import glob
import json
import argparse
import numpy as np
from multiprocessing import Pool
from pathlib import Path
import cv2
from array_utils import calc_minmax, stack_arrays_with_padding
from robo_manip_baselines.common import DataKey, DataManager
from robo_manip_baselines.utils.dataset_split import (
    deterministic_split,
    validate_split,
)

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--train_ratio", type=float, default=0.8)
parser.add_argument("--test_ratio", type=float, required=False)
parser.add_argument("--train_keywords", nargs="*", required=False)
parser.add_argument("--test_keywords", nargs="*", required=False)
parser.add_argument("--skip", type=int, default=6)
parser.add_argument("--cropped_img_size", type=int, default=480)
parser.add_argument("--resized_img_size", type=int, default=128)
parser.add_argument("--split_seed", type=int, default=0)
parser.add_argument("-j", "--nproc", type=int, default=1)
parser.add_argument("-q", "--quiet", action="store_true")
args = parser.parse_args()
print("[make_dataset] arguments:")
for k, v in vars(args).items():
    print(" " * 4 + f"{k}: {v}")
print()


def load_skip_resize_data(file_info):
    skip, resized_img_size, filename = file_info
    print(" " * 4 + filename)
    data_manager = DataManager(env=None)
    data_manager.load_data(filename)
    try:
        _front_images = data_manager.get_data(DataKey.get_rgb_image_key("front"))[
            ::skip
        ]
        _side_images = data_manager.get_data(DataKey.get_rgb_image_key("side"))[::skip]
        if args.cropped_img_size is not None:
            [fro_lef, fro_top, sid_lef, sid_top] = [
                (images_shape[ax] - args.cropped_img_size) // 2
                for images_shape in [_front_images.shape, _side_images.shape]
                for ax in [1, 2]
            ]
            [fro_rig, fro_bot, sid_rig, sid_bot] = [
                (p + args.cropped_img_size)
                for p in [fro_lef, fro_top, sid_lef, sid_top]
            ]
            _front_images = _front_images[:, fro_lef:fro_rig, fro_top:fro_bot, :]
            _side_images = _side_images[:, sid_lef:sid_rig, sid_top:sid_bot, :]
        if resized_img_size is not None:
            _front_images = np.array(
                [
                    cv2.resize(image, (resized_img_size, resized_img_size))
                    for image in _front_images
                ]
            )
            _side_images = np.array(
                [
                    cv2.resize(image, (resized_img_size, resized_img_size))
                    for image in _side_images
                ]
            )
        _wrenches = data_manager.get_data(DataKey.MEASURED_EEF_WRENCH)[::skip]
        _joints = data_manager.get_data(DataKey.MEASURED_JOINT_POS)[::skip]
        _actions = data_manager.get_data(DataKey.COMMAND_JOINT_POS)[::skip]
    except KeyError as e:
        print(f"{e.__class__.__name__}: filename={filename}")
        raise
    assert len(_front_images) == len(_side_images)
    assert len(_front_images) == len(_wrenches)
    assert len(_front_images) == len(_joints)
    assert len(_front_images) == len(_actions)
    return (_front_images, _side_images, _wrenches, _joints, _actions)


def save_arr(out_base_name, out_subpath_name, arr_data, quiet):
    """almost an alias for np.save()"""
    out_path = Path(out_base_name, out_subpath_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr_data)
    if not quiet:
        print(" " * 4 + f"{out_path}:\t{arr_data.shape}")


def load_data(in_dir, skip, resized_img_size, nproc):
    front_images = []
    side_images = []
    wrenches = []
    joints = []
    actions = []
    masks = []

    in_file_names = glob.glob(os.path.join(in_dir, "**/*.npz"), recursive=True)
    in_file_names.sort()
    try:
        assert len(in_file_names) >= 1, f"{len(in_file_names)=}"
    except AssertionError:
        sys.stderr.write(f"{sys.stderr.name} {in_dir=}\n")
        raise
    with Pool(nproc) as pool:
        loaded_data = pool.map(
            load_skip_resize_data,
            [(skip, resized_img_size, in_file_name) for in_file_name in in_file_names],
        )

    seq_length = []
    for _front_images, _side_images, _wrenches, _joints, _actions in loaded_data:
        seq_length.append(len(_joints))
    max_seq = max(seq_length)

    for _front_images, _side_images, _wrenches, _joints, _actions in loaded_data:
        front_images.append(_front_images)
        side_images.append(_side_images)
        wrenches.append(_wrenches)
        joints.append(_joints)
        actions.append(_actions)
        masks.append(
            np.concatenate((np.ones(len(_joints)), np.zeros(max_seq - len(_joints))))
        )

    front_images = stack_arrays_with_padding(front_images, max_seq)
    side_images = stack_arrays_with_padding(side_images, max_seq)
    wrenches = stack_arrays_with_padding(wrenches, max_seq)
    joints = stack_arrays_with_padding(joints, max_seq)
    actions = stack_arrays_with_padding(actions, max_seq)
    masks = np.stack(masks)

    return front_images, side_images, wrenches, joints, actions, masks, in_file_names


if __name__ == "__main__":
    # Load data
    if not args.quiet:
        print("[make_dataset] input files:")
    front_images, side_images, wrenches, joints, actions, masks, in_file_names = (
        load_data(args.in_dir, args.skip, args.resized_img_size, args.nproc)
    )

    # Set dataset index
    if args.train_keywords is None:
        train_idx_list, test_idx_list = deterministic_split(
            len(in_file_names),
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            seed=args.split_seed,
        )
    else:
        train_idx_list = [
            index
            for index, filename in enumerate(in_file_names)
            if any(keyword in filename for keyword in args.train_keywords)
        ]
        test_idx_list = [
            index for index in range(len(in_file_names)) if index not in train_idx_list
        ]
    if args.test_keywords is not None:
        test_idx_list = [
            index
            for index, filename in enumerate(in_file_names)
            if any(keyword in filename for keyword in args.test_keywords)
        ]
    train_idx_list, test_idx_list = validate_split(train_idx_list, test_idx_list)
    if not args.quiet:
        print()
        print(
            "\n".join(
                ["[make_dataset] train files:"]
                + [(" " * 4 + in_file_names[idx]) for idx in train_idx_list]
            )
        )
        print(
            "\n".join(
                ["[make_dataset] test files:"]
                + [(" " * 4 + in_file_names[idx]) for idx in test_idx_list]
            )
        )

    # save
    if not args.quiet:
        print()
        print("[make_dataset] output files:")
    save_arr(
        args.out_dir,
        "train/masks.npy",
        masks[train_idx_list].astype(np.float32),
        args.quiet,
    )
    save_arr(
        args.out_dir,
        "train/front_images.npy",
        front_images[train_idx_list].astype(np.uint8),
        args.quiet,
    )
    save_arr(
        args.out_dir,
        "train/side_images.npy",
        side_images[train_idx_list].astype(np.uint8),
        args.quiet,
    )
    save_arr(
        args.out_dir,
        "train/wrenches.npy",
        wrenches[train_idx_list].astype(np.float32),
        args.quiet,
    )
    save_arr(
        args.out_dir,
        "train/joints.npy",
        joints[train_idx_list].astype(np.float32),
        args.quiet,
    )
    save_arr(
        args.out_dir,
        "train/actions.npy",
        actions[train_idx_list].astype(np.float32),
        args.quiet,
    )
    save_arr(
        args.out_dir,
        "test/masks.npy",
        masks[test_idx_list].astype(np.float32),
        args.quiet,
    )
    save_arr(
        args.out_dir,
        "test/front_images.npy",
        front_images[test_idx_list].astype(np.uint8),
        args.quiet,
    )
    save_arr(
        args.out_dir,
        "test/side_images.npy",
        side_images[test_idx_list].astype(np.uint8),
        args.quiet,
    )
    save_arr(
        args.out_dir,
        "test/wrenches.npy",
        wrenches[test_idx_list].astype(np.float32),
        args.quiet,
    )
    save_arr(
        args.out_dir,
        "test/joints.npy",
        joints[test_idx_list].astype(np.float32),
        args.quiet,
    )
    save_arr(
        args.out_dir,
        "test/actions.npy",
        actions[test_idx_list].astype(np.float32),
        args.quiet,
    )

    # save bounds
    save_arr(args.out_dir, "wrench_bounds.npy", calc_minmax(wrenches), args.quiet)
    save_arr(args.out_dir, "joint_bounds.npy", calc_minmax(joints), args.quiet)
    save_arr(args.out_dir, "action_bounds.npy", calc_minmax(actions), args.quiet)

    source_paths = [
        os.path.relpath(filename, start=args.in_dir) for filename in in_file_names
    ]
    manifest = {
        "schema_version": 1,
        "preprocessing": {
            "crop": args.cropped_img_size,
            "resize": args.resized_img_size,
            "skip": args.skip,
            "split_seed": args.split_seed,
            "train_ratio": args.train_ratio,
            "test_ratio": args.test_ratio,
        },
        "source_files": source_paths,
        "splits": {
            "train": {
                "indices": train_idx_list,
                "source_files": [source_paths[index] for index in train_idx_list],
                "count": len(train_idx_list),
            },
            "test": {
                "indices": test_idx_list,
                "source_files": [source_paths[index] for index in test_idx_list],
                "count": len(test_idx_list),
            },
        },
        "arrays": {
            "front_images": {
                "shape": list(front_images.shape),
                "dtype": str(front_images.dtype),
            },
            "actions": {
                "shape": list(actions.shape),
                "dtype": str(actions.dtype),
            },
            "masks": {
                "shape": list(masks.shape),
                "dtype": str(masks.dtype),
            },
        },
    }
    manifest_path = Path(args.out_dir) / "dataset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    if not args.quiet:
        print(" " * 4 + str(manifest_path))
