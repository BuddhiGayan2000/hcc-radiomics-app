"""
Radiomic feature extraction from pixel arrays and ROI masks.

Computes 25 features from a region of interest:
- 7 morphological features (volume, area, diameter, etc.)
- 8 first-order intensity statistics (mean, median, std, etc.)
- 10 texture features (GLCM, run-length, etc.)

These are computed directly from pixel arrays in the backend,
used when the frontend sends a selected slice + ROI polygon.
"""

import numpy as np
from typing import Dict, List, Tuple


def shannon_entropy(vals: np.ndarray, n_bins: int = 32) -> float:
    """Compute Shannon entropy of a distribution."""
    if len(vals) == 0:
        return np.nan

    min_val = np.min(vals)
    max_val = np.max(vals)

    if max_val == min_val:
        return 0.0

    counts = np.zeros(n_bins)
    for v in vals:
        b = int(np.floor(((v - min_val) / (max_val - min_val)) * n_bins))
        if b >= n_bins:
            b = n_bins - 1
        counts[b] += 1

    n = len(vals)
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            h -= p * np.log2(p)

    return h


def first_order_stats(vals: np.ndarray) -> Dict[str, float]:
    """Compute first-order intensity statistics."""
    if len(vals) == 0:
        return {
            "Mean": np.nan,
            "Median": np.nan,
            "Min": np.nan,
            "Max": np.nan,
            "Std": np.nan,
            "Skewness": np.nan,
            "Kurtosis": np.nan,
            "Entropy": np.nan,
        }

    sorted_vals = np.sort(vals)
    mean = np.mean(vals)
    variance = np.var(vals)
    std = np.sqrt(variance)

    if std > 0:
        skew = np.mean(((vals - mean) / std) ** 3)
        kurt = np.mean(((vals - mean) / std) ** 4)
    else:
        skew = 0.0
        kurt = 0.0

    return {
        "Mean": float(mean),
        "Median": float(sorted_vals[len(sorted_vals) // 2]),
        "Min": float(sorted_vals[0]),
        "Max": float(sorted_vals[-1]),
        "Std": float(std),
        "Skewness": float(skew),
        "Kurtosis": float(kurt),
        "Entropy": float(shannon_entropy(vals)),
    }


def shape_stats(mask: np.ndarray, width: int, height: int) -> Dict[str, float]:
    """Compute morphological shape statistics."""
    area = 0
    sum_x = 0.0
    sum_y = 0.0
    pts = []

    for y in range(height):
        for x in range(width):
            if mask[y * width + x]:
                area += 1
                sum_x += x
                sum_y += y
                pts.append((x, y))

    if area == 0:
        return {
            "Volume": 0.0,
            "Area": 0.0,
            "MaxDiameter": 0.0,
            "SurfaceArea": 0.0,
            "Sphericity": 0.0,
            "Compactness": 0.0,
            "Elongation": 1.0,
        }

    cx = sum_x / area
    cy = sum_y / area

    # Calculate perimeter
    perimeter = 0
    boundary = []

    for y in range(height):
        for x in range(width):
            if mask[y * width + x]:
                up = 1 if (y > 0 and mask[(y - 1) * width + x]) else 0
                down = 1 if (y < height - 1 and mask[(y + 1) * width + x]) else 0
                left = 1 if (x > 0 and mask[y * width + x - 1]) else 0
                right = 1 if (x < width - 1 and mask[y * width + x + 1]) else 0

                if not (up and down and left and right):
                    perimeter += 1
                    boundary.append((x, y))

    # Calculate max diameter
    step = max(1, len(boundary) // 150)
    sample = boundary[::step]

    max_diameter = 0.0
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            dx = sample[i][0] - sample[j][0]
            dy = sample[i][1] - sample[j][1]
            d = np.sqrt(dx * dx + dy * dy)
            if d > max_diameter:
                max_diameter = d

    # Calculate PCA (elongation)
    sxx = 0.0
    syy = 0.0
    sxy = 0.0

    for x, y in pts:
        sxx += (x - cx) ** 2
        syy += (y - cy) ** 2
        sxy += (x - cx) * (y - cy)

    sxx /= area
    syy /= area
    sxy /= area

    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = np.sqrt(max(0, (tr * tr) / 4 - det))

    l1 = tr / 2 + disc
    l2 = max(1e-6, tr / 2 - disc)
    elongation = np.sqrt(l1 / l2)

    # Calculate sphericity and compactness
    sphericity = (4 * np.pi * area) / (perimeter * perimeter + 1e-9)
    compactness = area / (perimeter * perimeter + 1e-9)

    return {
        "Volume": float(area),
        "Area": float(area),
        "MaxDiameter": float(max_diameter),
        "SurfaceArea": float(perimeter),
        "Sphericity": float(sphericity),
        "Compactness": float(compactness),
        "Elongation": float(elongation),
    }


def bbox(mask: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
    """Get bounding box of mask."""
    min_x = width
    max_x = -1
    min_y = height
    max_y = -1

    for y in range(height):
        for x in range(width):
            if mask[y * width + x]:
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

    if max_x < 0:
        return None

    return (min_x, max_x, min_y, max_y)


def glcm_features(raw: np.ndarray, mask: np.ndarray, width: int, height: int, n_levels: int = 24) -> Dict[str, float]:
    """Compute Gray-Level Co-occurrence Matrix features."""
    bb = bbox(mask, width, height)

    if bb is None:
        return {
            "GLCM_Contrast": np.nan,
            "GLCM_Correlation": np.nan,
            "GLCM_Homogeneity": np.nan,
            "GLCM_Energy": np.nan,
            "GLCM_Entropy": np.nan,
        }

    min_x, max_x, min_y, max_y = bb

    roi_vals = []
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if mask[y * width + x]:
                roi_vals.append(raw[y, x])

    if len(roi_vals) == 0:
        return {
            "GLCM_Contrast": np.nan,
            "GLCM_Correlation": np.nan,
            "GLCM_Homogeneity": np.nan,
            "GLCM_Energy": np.nan,
            "GLCM_Entropy": np.nan,
        }

    min_val = np.min(roi_vals)
    max_val = np.max(roi_vals)

    if max_val == min_val:
        return {
            "GLCM_Contrast": np.nan,
            "GLCM_Correlation": np.nan,
            "GLCM_Homogeneity": np.nan,
            "GLCM_Energy": np.nan,
            "GLCM_Entropy": np.nan,
        }

    # Quantize to levels
    q = np.zeros((height, width), dtype=np.int16) - 1

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if mask[y * width + x]:
                lvl = int(np.floor(((raw[y, x] - min_val) / (max_val - min_val)) * (n_levels - 1)))
                lvl = max(0, min(n_levels - 1, lvl))
                q[y, x] = lvl

    # Compute GLCM
    glcm = np.zeros((n_levels, n_levels))
    offsets = [(1, 0), (1, 1), (0, 1), (-1, 1)]
    total = 0

    for dx, dy in offsets:
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                a = q[y, x]
                if a < 0:
                    continue

                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue

                b = q[ny, nx]
                if b < 0:
                    continue

                glcm[int(a), int(b)] += 1
                glcm[int(b), int(a)] += 1
                total += 2

    if total == 0:
        return {
            "GLCM_Contrast": np.nan,
            "GLCM_Correlation": np.nan,
            "GLCM_Homogeneity": np.nan,
            "GLCM_Energy": np.nan,
            "GLCM_Entropy": np.nan,
        }

    glcm = glcm / total

    # Compute features
    contrast = 0.0
    energy = 0.0
    entropy = 0.0
    mean_i = 0.0
    mean_j = 0.0

    for i in range(n_levels):
        for j in range(n_levels):
            p = glcm[i, j]
            contrast += p * ((i - j) ** 2)
            energy += p * p
            if p > 0:
                entropy -= p * np.log2(p)
            mean_i += i * p
            mean_j += j * p

    var_i = 0.0
    var_j = 0.0
    correlation = 0.0
    homogeneity = 0.0

    for i in range(n_levels):
        for j in range(n_levels):
            p = glcm[i, j]
            var_i += p * ((i - mean_i) ** 2)
            var_j += p * ((j - mean_j) ** 2)
            homogeneity += p / (1 + abs(i - j))

    std_i = np.sqrt(var_i)
    std_j = np.sqrt(var_j)

    if std_i > 0 and std_j > 0:
        for i in range(n_levels):
            for j in range(n_levels):
                p = glcm[i, j]
                correlation += (p * (i - mean_i) * (j - mean_j)) / (std_i * std_j)

    return {
        "GLCM_Contrast": float(contrast),
        "GLCM_Correlation": float(correlation),
        "GLCM_Homogeneity": float(homogeneity),
        "GLCM_Energy": float(energy),
        "GLCM_Entropy": float(entropy),
    }


def glrl_features(raw: np.ndarray, mask: np.ndarray, width: int, height: int, n_levels: int = 12) -> Dict[str, float]:
    """Compute Gray-Level Run-Length features."""
    bb = bbox(mask, width, height)

    if bb is None:
        return {"SRE": np.nan, "LRE": np.nan, "GLN": np.nan}

    min_x, max_x, min_y, max_y = bb

    roi_vals = []
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if mask[y * width + x]:
                roi_vals.append(raw[y, x])

    if len(roi_vals) == 0:
        return {"SRE": np.nan, "LRE": np.nan, "GLN": np.nan}

    min_val = np.min(roi_vals)
    max_val = np.max(roi_vals)

    if max_val == min_val:
        return {"SRE": np.nan, "LRE": np.nan, "GLN": np.nan}

    # Quantize
    q = np.zeros((height, width), dtype=np.int16)

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if mask[y * width + x]:
                lvl = int(np.floor(((raw[y, x] - min_val) / (max_val - min_val)) * (n_levels - 1))) + 1
                lvl = max(1, min(n_levels, lvl))
                q[y, x] = lvl

    # Compute run-length matrix
    max_run = max(max_x - min_x + 1, max_y - min_y + 1)
    P = np.zeros((n_levels + 1, max_run + 1))
    dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for dx, dy in dirs:
        visited = np.zeros((height, width), dtype=np.uint8)

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if q[y, x] == 0 or visited[y, x]:
                    continue

                level = q[y, x]
                run_len = 1
                visited[y, x] = 1

                px, py = x + dx, y + dy
                while min_x <= px <= max_x and min_y <= py <= max_y and q[py, px] == level:
                    visited[py, px] = 1
                    run_len += 1
                    px += dx
                    py += dy

                P[level, min(run_len, max_run)] += 1

    Nr = 0
    for l in range(1, n_levels + 1):
        for r in range(1, max_run + 1):
            Nr += P[l, r]

    if Nr == 0:
        return {"SRE": np.nan, "LRE": np.nan, "GLN": np.nan}

    SRE = 0.0
    LRE = 0.0
    GLN = 0.0

    for l in range(1, n_levels + 1):
        row_sum = 0.0
        for r in range(1, max_run + 1):
            SRE += P[l, r] / (r * r)
            LRE += P[l, r] * (r * r)
            row_sum += P[l, r]
        GLN += row_sum * row_sum

    return {"SRE": float(SRE / Nr), "LRE": float(LRE / Nr), "GLN": float(GLN / Nr)}


def liver_context_features(raw: np.ndarray, mask: np.ndarray, width: int, height: int, tumor_mean: float) -> Dict[str, float]:
    """Compute liver-context features."""
    n = width * height

    # Estimate background threshold (bottom 10%)
    sorted_vals = np.sort(raw.flatten())
    thresh = sorted_vals[max(0, int(n * 0.1))]

    context_vals = []
    for i in range(n):
        if not mask[i] and raw.flat[i] > thresh:
            context_vals.append(raw.flat[i])

    if len(context_vals) == 0:
        liver_entropy = np.nan
        liver_mean = np.nan
    else:
        liver_entropy = shannon_entropy(np.array(context_vals))
        liver_mean = np.mean(context_vals)

    tumor_liver_contrast = abs(tumor_mean - liver_mean) / (abs(liver_mean) + 1e-6) if not np.isnan(liver_mean) else np.nan

    return {
        "LiverEntropy": float(liver_entropy),
        "TumorLiverContrast": float(tumor_liver_contrast),
    }


def extract_all_features(raw: np.ndarray, mask: np.ndarray, width: int, height: int) -> Dict[str, float]:
    """Extract all 25 radiomic features from a masked ROI."""
    roi_vals = []

    for i in range(width * height):
        if mask[i]:
            roi_vals.append(raw.flat[i])

    # First-order stats
    fo = first_order_stats(np.array(roi_vals))

    # Morphological
    shape = shape_stats(mask, width, height)

    # GLCM
    glcm = glcm_features(raw, mask, width, height)

    # Run-length
    rlm = glrl_features(raw, mask, width, height)

    # Liver context
    liver_ctx = liver_context_features(raw, mask, width, height, fo["Mean"])

    # Combine all
    all_features = {**shape, **fo, **glcm, **rlm, **liver_ctx}

    return all_features


def rasterize_mask(polygon: List[Dict[str, float]], width: int, height: int) -> np.ndarray:
    """Convert polygon to binary mask using point-in-polygon test."""
    mask = np.zeros(width * height, dtype=np.uint8)

    if len(polygon) < 3:
        return mask

    # Find bounding box
    min_x = width
    max_x = 0
    min_y = height
    max_y = 0

    for p in polygon:
        x, y = int(p["x"]), int(p["y"])
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)

    min_x = max(0, min_x)
    max_x = min(width - 1, max_x)
    min_y = max(0, min_y)
    max_y = min(height - 1, max_y)

    # Rasterize using point-in-polygon
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if point_in_polygon(x + 0.5, y + 0.5, polygon):
                mask[y * width + x] = 1

    return mask


def point_in_polygon(x: float, y: float, polygon: List[Dict[str, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    inside = False

    for i in range(len(polygon)):
        j = i - 1

        xi = polygon[i]["x"]
        yi = polygon[i]["y"]
        xj = polygon[j]["x"]
        yj = polygon[j]["y"]

        intersect = (yi > y) != (yj > y) and x < ((xj - xi) * (y - yi) / (yj - yi) + xi)
        if intersect:
            inside = not inside

    return inside
