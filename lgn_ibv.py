"""
Same code as colab_vsc.py just more organized and easier to read.
LGN-IBV: Lateral Geniculate Nucleus - Image-Based Vision Model
Generates binocular spontaneous activity and estimates depth from autostereograms.
"""

import hashlib
import json
import os
import random
import time
from datetime import date
from pathlib import Path
from random import randint

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import pylab
from PIL import Image, ImageFilter, ImageOps
from scipy import signal
from scipy.interpolate import griddata
from sklearn.decomposition import FastICA
from sklearn.feature_extraction import image as skimage


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def pixel_distance(x0, y0, x1, y1) -> float:
    """Euclidean distance between two 2-D points."""
    return np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)


def open_norm(path: str, verbose: bool = False):
    """
    Open a grayscale image and z-score normalise it.

    Returns (raw, normalised) when verbose=True, else just the normalised array.
    """
    raw = np.array(Image.open(path).convert("L"))
    normalised = (raw - raw.mean()) / raw.std()
    return (raw, normalised) if verbose else normalised


def save_array_as_image(array: np.ndarray, path: str) -> None:
    """Scale a float array to [0, 255] and save it as a PNG."""
    scaled = (255.0 / array.max() * (array - array.min())).astype(np.uint8)
    Image.fromarray(scaled).save(path)


def calculate_optimal_p(t: float, r: float, a: float) -> float:
    """Return the recruitment probability p that balances wave propagation."""
    return t / ((np.pi * r ** 2 / 2) * (1 + a))


def generate_ident_hash(*args) -> str:
    """SHA-256 hash (first 20 chars) of the stringified experiment parameters."""
    raw = "".join(f"{v:f}" for v in args)
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def resolve_existing_path(base: Path, candidates: list) -> Path:
    """Return the first candidate path (relative to *base*) that exists."""
    for candidate in candidates:
        full = base / candidate
        if full.exists():
            return full
    searched = [str(base / c) for c in candidates]
    raise FileNotFoundError(f"Could not find required file. Checked: {searched}")


# ---------------------------------------------------------------------------
# LGN model
# ---------------------------------------------------------------------------

class LGN:
    """
    Lateral Geniculate Nucleus model.

    Simulates binocular spontaneous retinal wave activity on a 2-D grid with
    `num_layers` layers (default 2 = left eye + right eye).
    """

    def __init__(
        self,
        width: int = 128,
        p: float = 0.5,
        r: float = 1.0,
        t: int = 1,
        trans: float = 0.0,
        num_layers: int = 2,
        make_wave: bool = True,
        random_seed: int = 0,
    ):
        random.seed(random_seed)
        self.width = width
        self.p = p
        self.r = r
        self.t = t
        self.trans = trans
        self.num_layers = num_layers

        if make_wave:
            self.reset_wave()

    # ------------------------------------------------------------------
    # Wave initialisation
    # ------------------------------------------------------------------

    def reset_wave(self) -> None:
        """Reinitialise the grid and grow a new random wave."""
        w = self.width
        self.recruitable = np.random.rand(self.num_layers, w, w) < self.p
        self.tot_recruitable = int(self.recruitable.sum())
        self.tot_recruitable_active = 0
        self.tot_active = 0
        self.active = np.zeros((self.num_layers, w, w), bool)
        self.active_neighbors = np.zeros((self.num_layers, w, w), int)
        self.activated = []  # recently activated nodes pending propagation

        if self.tot_recruitable > 0:
            while self.fraction_active() < 0.20:
                self.activate()

    # ------------------------------------------------------------------
    # Wave dynamics
    # ------------------------------------------------------------------

    def fraction_active(self) -> float:
        """Fraction of recruitable cells that are currently active."""
        if self.tot_recruitable > 0:
            return self.tot_recruitable_active / self.tot_recruitable
        return float("nan")

    def propagate(self) -> None:
        """
        Spread activity from every node in `self.activated` to its neighbours
        within radius `r`.  Cross-layer transfer occurs with probability `trans`.
        """
        while self.activated:
            act_l, act_x, act_y = self.activated.pop()
            self.active[act_l, act_x, act_y] = True
            self.tot_active += 1
            self.tot_recruitable_active += 1

            for l in range(self.num_layers):
                for x in range(int(act_x - self.r), int(act_x + self.r + 1)):
                    for y in range(int(act_y - self.r), int(act_y + self.r + 1)):
                        if pixel_distance(act_x, act_y, x, y) > self.r:
                            continue

                        xi, yi = x % self.width, y % self.width

                        if l != act_l:
                            # Cross-layer spread (stochastic)
                            if np.random.rand() < self.trans:
                                self.active_neighbors[l, xi, yi] += 1
                        else:
                            self.active_neighbors[l, xi, yi] += 1

                        already_active = self.active[l, xi, yi]
                        threshold_reached = self.active_neighbors[l, xi, yi] == self.t

                        if threshold_reached and not already_active:
                            if self.recruitable[l, xi, yi]:
                                self.activated.append([l, xi, yi])
                            else:
                                # Activate but do NOT propagate further
                                self.active[l, xi, yi] = True
                                self.tot_active += 1

    def activate(self) -> None:
        """Seed activity at a random recruitable, currently-inactive node."""
        if self.fraction_active() > 0.95:
            return

        while True:
            l = np.random.randint(0, self.num_layers)
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.width)
            if self.recruitable[l, x, y] and not self.active[l, x, y]:
                break

        self.activated.append([l, x, y])
        self.propagate()

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def correlation(self) -> float:
        """Pearson correlation between left-eye and right-eye activity maps."""
        if self.num_layers < 2:
            print("Monocular models have no inter-eye correlation.")
            return 0.0

        w = self.width
        binary = np.zeros((2, w, w), int)
        binary[np.where(self.active)] = 1

        mean0, mean1 = binary[0].mean(), binary[1].mean()
        std0, std1 = binary[0].std(), binary[1].std()
        cov = ((binary[0] - mean0) * (binary[1] - mean1)).mean()
        return cov / (std0 * std1)

    def make_img_mat(self) -> np.ndarray:
        """
        Return a (num_layers, width, width) binary float array of active cells.
        """
        w = self.width
        img_array = np.zeros((self.num_layers, w, w))
        for l in range(self.num_layers):
            for x in range(w - 1):
                for y in range(w - 1):
                    if self.active[l, x, y]:
                        img_array[l, x, y] = 1.0
        return img_array


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_lr_activity(layer_activity: np.ndarray, patch_count: int,
                     ident_hash: str, parent_path: str) -> None:
    """Save left- and right-eye activity images to disk."""
    for side, idx in (("r", 0), ("l", 1)):
        folder = Path(parent_path) / "images" / "activity" / ident_hash / side
        folder.mkdir(parents=True, exist_ok=True)
        filename = f"{side}{patch_count}.png"
        save_array_as_image(layer_activity[idx], str(folder / filename))

    print(f"SAVING ACTIVITY TO: {parent_path}/images/activity/{ident_hash}")


def save_lr_filters(first_eye: np.ndarray, second_eye: np.ndarray,
                    ident_hash: str, parent_path: str) -> None:
    """Save the first 100 left- and right-eye filter images to disk."""
    for side, eye in (("r", first_eye), ("l", second_eye)):
        folder = Path(parent_path) / "images" / "filters" / ident_hash / side
        folder.mkdir(parents=True, exist_ok=True)

    for i in range(min(100, first_eye.shape[0])):
        for side, eye in (("r", first_eye), ("l", second_eye)):
            folder = Path(parent_path) / "images" / "filters" / ident_hash / side
            save_array_as_image(eye[i], str(folder / f"r{i}.png"))

    print(f"SAVING FILTERS TO: {parent_path}/images/filters/{ident_hash}")


# ---------------------------------------------------------------------------
# Patch generation & ICA
# ---------------------------------------------------------------------------

def generate_patches(num_patches: int, patch_size: int, lgn_width: int,
                     lgn_p: float, lgn_r: float, lgn_t: int, lgn_a: float,
                     ident_hash: str, parent_path: str):
    """
    Generate at least `num_patches` binocular patch pairs from LGN waves.

    Returns (patch_array, last_layer_activity).
    """
    half_comp = patch_size ** 2
    patch_base = None

    while (patch_base is None or patch_base.shape[0] < num_patches):
        lgn = LGN(
            width=lgn_width, p=lgn_p, r=lgn_r,
            t=lgn_t, trans=lgn_a, num_layers=2,
        )
        layer_activity = lgn.make_img_mat()
        save_lr_activity(layer_activity, 0 if patch_base is None else patch_base.shape[0],
                         ident_hash, parent_path)

        patches_l = skimage.extract_patches_2d(layer_activity[0], (patch_size, patch_size))
        patches_r = skimage.extract_patches_2d(layer_activity[1], (patch_size, patch_size))

        flat_l = patches_l.reshape(len(patches_l), -1)
        flat_r = patches_r.reshape(len(patches_r), -1)
        composite = np.concatenate((flat_l, flat_r), axis=1)

        # Remove patches where either eye has zero variance (blank patches)
        valid = [
            i for i in range(len(composite))
            if composite[i, :half_comp].std() > 0 and composite[i, half_comp:].std() > 0
        ]
        composite = composite[valid]

        patch_base = composite if patch_base is None else np.vstack((patch_base, composite))

    return patch_base[:num_patches], layer_activity


def perform_ica(num_components: int, patches: np.ndarray) -> np.ndarray:
    """Fit FastICA on `patches` and return the component matrix."""
    ica = FastICA(
        n_components=num_components,
        random_state=1,
        max_iter=1_000_000,
        whiten="unit-variance",
    )
    return ica.fit(patches).components_


def generate_filters(num_filters: int, num_components: int, num_patches: int,
                     patch_size: int, lgn_width: int, lgn_p: float, lgn_r: float,
                     lgn_t: int, lgn_a: float, ident_hash: str, parent_path: str):
    """
    Accumulate `num_filters` ICA-derived binocular filters.

    Returns (filter_array, patches, last_layer_activity).
    """
    print("GENERATING FILTERS")
    bar = progressbar.ProgressBar(max_value=num_filters)
    filter_base = None
    last_patches = None
    last_activity = None

    while filter_base is None or filter_base.shape[0] < num_filters:
        patches, last_activity = generate_patches(
            num_patches, patch_size, lgn_width,
            lgn_p, lgn_r, lgn_t, lgn_a, ident_hash, parent_path,
        )
        last_patches = patches
        new_filters = perform_ica(num_components, patches)

        filter_base = (
            new_filters if filter_base is None
            else np.vstack((filter_base, new_filters))
        )
        count = filter_base.shape[0]
        bar.update(min(count, num_filters))
        print(count, end=" ")

    return filter_base[:num_filters], last_patches, last_activity


def unpack_filters(filters: np.ndarray, ident_hash: str, parent_path: str):
    """
    Split the concatenated binocular filter matrix into per-eye 2-D filter arrays.

    Returns (first_eye_filters, second_eye_filters).
    """
    half = filters.shape[1] // 2
    dim = int(np.sqrt(half))

    first_eye = filters[:, :half].reshape(-1, dim, dim)
    second_eye = filters[:, half:].reshape(-1, dim, dim)

    save_lr_filters(first_eye, second_eye, ident_hash, parent_path)
    return first_eye, second_eye


# ---------------------------------------------------------------------------
# Disparity estimation
# ---------------------------------------------------------------------------

def linear_convolution(center: np.ndarray, slide: np.ndarray) -> np.ndarray:
    """
    Slide `slide` across a padded version of `center` and return the
    absolute dot-product at each offset.
    """
    if center.shape != slide.shape:
        return None

    w = center.shape[1]
    padded = np.zeros((center.shape[0], w * 3))
    padded[:, w: w * 2] = center

    estimate = np.array([
        np.sum(padded[:, x: x + w] * slide)
        for x in range(w * 2)
    ])
    return np.abs(estimate)


def linear_disparity(first_eye: np.ndarray, second_eye: np.ndarray) -> np.ndarray:
    """Compute a disparity tuning curve for each filter pair."""
    disparity_map = np.empty((first_eye.shape[0], first_eye.shape[1] * 2))
    for i in range(first_eye.shape[0]):
        disparity_map[i] = linear_convolution(first_eye[i], second_eye[i])
    return disparity_map


def normalize_disparity(disparity_map: np.ndarray) -> np.ndarray:
    """Normalise each disparity curve by its mean across filters."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return disparity_map / np.mean(disparity_map, axis=0)


# ---------------------------------------------------------------------------
# Activity & depth estimation
# ---------------------------------------------------------------------------

def double_convolve(normal: np.ndarray, shifted: np.ndarray,
                    image: np.ndarray, pupillary_distance: int) -> np.ndarray:
    """
    Convolve `image` with `normal` and `shifted` filters separately,
    align by `pupillary_distance`, multiply, threshold negatives to zero,
    and return the absolute result.
    """
    conv_normal = signal.convolve2d(image, normal, boundary="symm", mode="same")
    conv_shifted = signal.convolve2d(image, shifted, boundary="symm", mode="same")

    # Crop to align the two convolutions
    aligned_normal = conv_normal[:, :-pupillary_distance]
    aligned_shifted = conv_shifted[:, pupillary_distance:]

    product = aligned_normal * aligned_shifted
    product[product < 0] = 0  # Threshold sub-zero values

    # Place result back into the original image shape
    result = np.zeros(image.shape)
    result[:, pupillary_distance:] = product
    return np.abs(product)


def scale_disparity(activity_map: np.ndarray,
                    disparity_map: np.ndarray) -> np.ndarray:
    """Weight `disparity_map` by the local `activity_map` at each pixel."""
    scaled = np.zeros((activity_map.shape[0], activity_map.shape[1], disparity_map.shape[0]))
    scaled[:, :] = disparity_map
    for x in range(activity_map.shape[0]):
        for y in range(activity_map.shape[1]):
            scaled[x, y] *= activity_map[x, y]
    return scaled


def generate_activity(autostereogram: np.ndarray, asg_patch_size: int,
                      first_eye: np.ndarray, second_eye: np.ndarray,
                      disparity_map: np.ndarray) -> np.ndarray:
    """Sum scaled disparity activity across all filter pairs."""
    print("\nCALCULATING ACTIVITY")
    bar = progressbar.ProgressBar(max_value=first_eye.shape[0])
    summed = None

    for i in range(first_eye.shape[0]):
        conv = double_convolve(first_eye[i], second_eye[i], autostereogram, asg_patch_size)
        contribution = scale_disparity(conv, disparity_map[i])
        summed = contribution if summed is None else summed + contribution
        bar.update(i)

    bar.update(first_eye.shape[0])
    return summed


def estimate_depth(activity: np.ndarray) -> np.ndarray:
    """Pick the disparity peak per pixel to form a depth map."""
    print("\nESTIMATING DEPTH")
    depth = np.zeros((activity.shape[0], activity.shape[1]))
    bar = progressbar.ProgressBar(max_value=activity.shape[0])
    half_depth = activity.shape[2] // 2

    for x in range(activity.shape[0]):
        for y in range(activity.shape[1]):
            depth[x, y] = abs(int(np.nanargmax(activity[x, y])) - half_depth)
        bar.update(x)

    return depth


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    num_filters: int, num_components: int, num_patches: int, patch_size: int,
    lgn_width: int, lgn_p: float, lgn_r: float, lgn_t: int, lgn_a: float,
    autostereogram: np.ndarray, asg_patch_size: int,
    groundtruth: np.ndarray, experiment_folder: str,
) -> dict:
    """
    Run a full experiment: generate filters, estimate depth, compare to ground
    truth, save outputs, and return a parameter dict with the correlation score.
    """
    ident_hash = generate_ident_hash(
        num_filters, num_components, num_patches, patch_size,
        lgn_width, lgn_p, lgn_r, lgn_t, lgn_a, time.time(),
    )

    filters, _, _ = generate_filters(
        num_filters, num_components, num_patches, patch_size,
        lgn_width, lgn_p, lgn_r, lgn_t, lgn_a, ident_hash, experiment_folder,
    )
    first_eye, second_eye = unpack_filters(filters, ident_hash, experiment_folder)

    disparity_map = normalize_disparity(linear_disparity(first_eye, second_eye))
    activity = generate_activity(autostereogram, asg_patch_size, first_eye, second_eye, disparity_map)
    depth_estimate = estimate_depth(activity)

    correlation = abs(np.corrcoef(depth_estimate.flatten(), groundtruth.flatten())[0, 1])
    current_time = time.localtime()

    # Save depth map image
    depth_image_path = f"{experiment_folder}/images/depthmaps/{ident_hash}.png"
    save_array_as_image(depth_estimate, depth_image_path)

    # Save experiment parameters + result as JSON
    params = {
        "num_filters": num_filters,
        "num_components": num_components,
        "num_patches": num_patches,
        "patch_size": patch_size,
        "lgn_width": lgn_width,
        "lgn_p": lgn_p,
        "lgn_r": lgn_r,
        "lgn_t": lgn_t,
        "lgn_a": lgn_a,
        "corr": correlation,
        "time": time.strftime("%a, %d %b %Y %H:%M:%S GMT", current_time),
        "id": ident_hash,
    }
    json_path = f"{experiment_folder}/json/{ident_hash}.json"
    with open(json_path, "w") as f:
        json.dump(params, f, indent=2)

    return params


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    today = date.today()
    main_path = Path(os.getcwd())
    parent_path = str(main_path / str(today))

    # Create required output directories
    for subdir in ("images", "json", "images/depthmaps"):
        (Path(parent_path) / subdir).mkdir(parents=True, exist_ok=True)

    # Locate input files
    auto_path = resolve_existing_path(main_path, [Path("shift5_70patch.png")])
    gt_path = resolve_existing_path(main_path, [Path("dm.png")])

    autostereogram = open_norm(str(auto_path))
    groundtruth = np.array(Image.open(gt_path).convert("L"))

    # Sweep over (r, t, a) parameter combinations
    p_shift = 0.01
    for r in range(2, 5):
        for t in range(3, 5):
            for a in np.arange(0.1, 0.7, 0.1):
                p = calculate_optimal_p(t, r, a) + p_shift
                print("-------------------------------------")
                print(f"r={r}  t={t}  a={a:.1f}  p={p:.4f}")
                print("-------------------------------------")
                result = run_experiment(
                    num_filters=100, num_components=20,
                    num_patches=10000, patch_size=9,
                    lgn_width=256, lgn_p=p,
                    lgn_r=r, lgn_t=t, lgn_a=a,
                    autostereogram=autostereogram,
                    asg_patch_size=70,
                    groundtruth=groundtruth,
                    experiment_folder=parent_path,
                )
                print(f"Correlation: {result['corr']:.4f}  |  ID: {result['id']}")


if __name__ == "__main__":
    main()