"""Rossby Wave Breaking detection using QGPV contour analysis.

Detects, classifies, and maps RWB events from 2-D QGPV fields using
the overturning index (Barnes and Hartmann, 2012) and cut-off index,
adapted from the WaveBreaking package (Kaderli, 2023).

See README.md for method details, references, and differences from the
original WaveBreaking package.
"""

import itertools
import logging
import typing

import numpy as np
import pandas as pd
import skimage.measure

_LOG = logging.getLogger(__name__)

DEFAULT_QGPV_LEVEL: float = 0.0001


def _build_clusters(y_vals: typing.List[int]) -> typing.List[typing.List[int]]:
    if not y_vals:
        return []
    clusters: typing.List[typing.List[int]] = [[y_vals[0]]]
    for y in y_vals[1:]:
        if y - clusters[-1][-1] > 1:
            clusters.append([y])
        else:
            clusters[-1].append(y)
    return clusters


def _filament_gaps(clusters: typing.List[typing.List[int]]) -> typing.List[int]:
    inter_gaps = [
        (clusters[k + 1][0] - clusters[k][-1], k)
        for k in range(len(clusters) - 1)
    ]
    filament_indices: typing.List[int] = []
    for p in range(0, len(inter_gaps) - 1, 2):
        gap_a_size, gap_a_k = inter_gaps[p]
        gap_b_size, gap_b_k = inter_gaps[p + 1]
        if gap_a_size <= gap_b_size:
            filament_indices.append(gap_a_k)
        else:
            filament_indices.append(gap_b_k)
    return filament_indices


def _check_duplicate_contours(
    contours_list: typing.List[np.ndarray],
    *,
    nlon: int,
) -> typing.List[int]:
    temp = [np.c_[item[:, 0] % nlon, item[:, 1]] for item in contours_list]
    check = [
        (i1, i2)
        for (i1, e1), (i2, e2) in itertools.permutations(enumerate(temp), r=2)
        if set(map(tuple, e1)).issubset(set(map(tuple, e2)))
    ]
    drop: typing.List[int] = []
    lens = np.array([len(item) for item in temp])
    for indices in check:
        if lens[indices[0]] == lens[indices[1]]:
            drop.append(max(indices))
        else:
            drop.append(indices[np.argmin(lens[[indices[0], indices[1]]])])
    return list(set(drop))


def extract_contour(
    qgpv_2d: np.ndarray,
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    level: float,
    periodic_add: float = 120.0,
) -> typing.Optional[pd.DataFrame]:
    """Extract the main (widest) contour from a 2-D QGPV field."""
    nlat, nlon = qgpv_2d.shape
    dlon = abs(float(lon[1] - lon[0])) if len(lon) > 1 else 1.0
    pad_cols = int(periodic_add / dlon)

    qgpv_extended = np.concatenate(
        [qgpv_2d, qgpv_2d[:, :pad_cols]], axis=1,
    )

    try:
        contours_raw = skimage.measure.find_contours(qgpv_extended, level)
    except Exception:
        _LOG.exception("Contour extraction failed at level %s", level)
        return None

    if len(contours_raw) == 0:
        return None

    contours_index: typing.List[np.ndarray] = []
    for c in contours_raw:
        rounded = np.round(c).astype(int)
        xy = rounded[:, ::-1]
        unique = np.array(list(dict.fromkeys(map(tuple, xy))))
        if len(unique) >= 4:
            contours_index.append(unique)

    if len(contours_index) == 0:
        return None

    drop = _check_duplicate_contours(contours_index, nlon=nlon)
    contours_index = [c for i, c in enumerate(contours_index) if i not in drop]

    if len(contours_index) == 0:
        return None

    exp_lons = [len(set(c[:, 0])) * dlon for c in contours_index]
    best_idx = int(np.argmax(exp_lons))
    best = contours_index[best_idx]

    return pd.DataFrame({"x": best[:, 0], "y": best[:, 1]})


def compute_overturning_index(
    contour: typing.Optional[pd.DataFrame],
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    range_group: float = 5.0,
    min_exp: float = 5.0,
    min_fold_lat: float = 5.0,
) -> typing.List[typing.Dict[str, typing.Any]]:
    """Detect overturning events on the main jet contour."""
    if contour is None or len(contour) < 10:
        return []

    nlon = len(lon)
    dlon = abs(float(lon[1] - lon[0])) if len(lon) > 1 else 1.0
    min_fold_cells = max(2, int(min_fold_lat / dlon))

    lons_arr = np.unique(contour["x"].values)
    is_overturning = np.zeros(len(lons_arr), dtype=bool)

    for i, xi in enumerate(lons_arr):
        y_vals = sorted(set(
            int(v) for v in contour[contour["x"] == xi]["y"].values
        ))
        if len(y_vals) == 0:
            continue
        clusters = _build_clusters(y_vals)
        if len(clusters) < 3:
            continue
        max_gap = max(
            clusters[k + 1][0] - clusters[k][-1]
            for k in range(len(clusters) - 1)
        )
        if max_gap >= min_fold_cells:
            is_overturning[i] = True

    if not is_overturning.any():
        return []

    ot_lons = pd.DataFrame({"lon": lons_arr[is_overturning]})
    ot_lons["label"] = (ot_lons["lon"].diff() > range_group / dlon).cumsum()

    groups = ot_lons.groupby("label")
    df_ot = groups.agg(["min", "max"]).astype(int).reset_index(drop=True)
    df_ot.columns = ["min_lon", "max_lon"]

    if len(df_ot) == 0:
        return []

    def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) <= 1:
            return df
        temp = [
            np.array(range(row.min_lon, row.max_lon + 1)) % nlon
            for _, row in df.iterrows()
        ]
        index_combinations = list(itertools.permutations(df.index, r=2))
        check = [
            item for item in index_combinations
            if set(temp[item[0]]).issubset(set(temp[item[1]]))
        ]
        drop_idx: typing.List[int] = []
        for item in check:
            lens_pair = [len(temp[i]) for i in item]
            if lens_pair[0] == lens_pair[1]:
                drop_idx.append(max(item))
            else:
                drop_idx.append(item[np.argmin(lens_pair)])
        return df[~df.reset_index(drop=True).index.isin(drop_idx)]

    def _check_expansion(df: pd.DataFrame) -> pd.DataFrame:
        keep = []
        for idx, row in df.iterrows():
            exp_lon_cells = row.max_lon - row.min_lon
            mid_y = (
                contour[
                    contour["x"].isin(range(int(row.min_lon), int(row.max_lon) + 1))
                ]["y"].mean()
            )
            mid_lat_deg = float(lat[min(int(round(mid_y)), len(lat) - 1)])
            cos_lat = max(np.cos(np.radians(mid_lat_deg)), 0.05)
            effective_exp = exp_lon_cells * dlon * cos_lat
            keep.append(effective_exp >= min_exp)
        return df[keep]

    def _find_lat_expansion(df: pd.DataFrame) -> pd.DataFrame:
        lats_list = [
            contour[contour["x"].isin(range(row.min_lon, row.max_lon + 1))]["y"]
            for _, row in df.iterrows()
        ]
        ot_lats = pd.DataFrame(
            [(item.min(), item.max()) for item in lats_list],
            columns=["min_lat", "max_lat"],
        ).astype(int)
        return pd.concat([df.reset_index(drop=True), ot_lats], axis=1)

    routines = [_remove_duplicates, _check_expansion, _find_lat_expansion]
    step = 0
    while len(df_ot) > 0 and step <= 2:
        df_ot = routines[step](df_ot).reset_index(drop=True)
        step += 1

    if len(df_ot) == 0:
        return []

    events: typing.List[typing.Dict[str, typing.Any]] = []
    for _, row in df_ot.iterrows():
        events.append({
            "min_lon_idx": int(row.min_lon),
            "max_lon_idx": int(row.max_lon),
            "min_lat_idx": int(row.min_lat),
            "max_lat_idx": int(row.max_lat),
            "contour": contour,
        })
    return events


def classify_wave_breaking(
    event: typing.Dict[str, typing.Any],
    *,
    qgpv_2d: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    contour_level: float = DEFAULT_QGPV_LEVEL,
) -> str:
    """Classify an overturning event as AWB or CWB using mean-QGPV voting."""
    contour = event["contour"]
    min_lon_idx = event["min_lon_idx"]
    max_lon_idx = event["max_lon_idx"]
    nlon = len(lon)
    nlat = len(lat)

    awb_votes = 0
    cwb_votes = 0

    for lon_idx in range(min_lon_idx, max_lon_idx + 1):
        y_vals = sorted(set(
            int(v) for v in contour[contour["x"] == lon_idx]["y"].values
        ))
        clusters = _build_clusters(y_vals)
        if len(clusters) < 3:
            continue

        actual_x = int(lon_idx) % nlon
        for k in _filament_gaps(clusters):
            fill_lo = max(0, clusters[k][-1])
            fill_hi = min(nlat - 1, clusters[k + 1][0])
            if fill_hi <= fill_lo:
                continue
            mean_q = float(qgpv_2d[fill_lo:fill_hi + 1, actual_x].mean())
            if mean_q > contour_level:
                cwb_votes += 1
            else:
                awb_votes += 1

    if awb_votes > cwb_votes:
        return "AWB"
    return "CWB"


def get_event_properties(
    event: typing.Dict[str, typing.Any],
    *,
    event_type: str,
    lat: np.ndarray,
    lon: np.ndarray,
) -> typing.Dict[str, typing.Any]:
    """Extract geographic properties of a wave breaking event."""
    nlon = len(lon)
    nlat = len(lat)
    lon_min = float(lon[event["min_lon_idx"] % nlon])
    lon_max = float(lon[event["max_lon_idx"] % nlon])
    lat_min = float(lat[min(event["min_lat_idx"], nlat - 1)])
    lat_max = float(lat[min(event["max_lat_idx"], nlat - 1)])

    return {
        "type": event_type,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "centroid_lon": (lon_min + lon_max) / 2,
        "centroid_lat": (lat_min + lat_max) / 2,
    }


def create_rwb_mask(
    events: typing.List[typing.Dict[str, typing.Any]],
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    types: typing.List[str],
    qgpv_2d: typing.Optional[np.ndarray] = None,
    contour_level: float = DEFAULT_QGPV_LEVEL,
    min_filament_lat: float = 3.0,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Create binary AWB/CWB masks from overturning events."""
    nlon = len(lon)
    nlat = len(lat)
    dlat = abs(float(lat[1] - lat[0])) if len(lat) > 1 else 1.0
    min_filament_cells = max(2, int(round(min_filament_lat / dlat)))
    mask_awb = np.zeros((nlat, nlon), dtype=np.float32)
    mask_cwb = np.zeros((nlat, nlon), dtype=np.float32)

    for event, etype_fallback in zip(events, types):
        contour = event["contour"]
        min_lon_idx = event["min_lon_idx"]
        max_lon_idx = event["max_lon_idx"]

        for lon_idx in range(min_lon_idx, max_lon_idx + 1):
            y_vals = sorted(set(
                int(v) for v in contour[contour["x"] == lon_idx]["y"].values
            ))
            clusters = _build_clusters(y_vals)
            if len(clusters) < 3:
                continue

            actual_lon_idx = int(lon_idx) % nlon

            for k in _filament_gaps(clusters):
                fill_lo = max(0, clusters[k][-1])
                fill_hi = min(nlat - 1, clusters[k + 1][0])
                if fill_hi <= fill_lo:
                    continue
                if (fill_hi - fill_lo) < min_filament_cells:
                    continue

                if qgpv_2d is not None:
                    mean_q = float(
                        qgpv_2d[fill_lo:fill_hi + 1, actual_lon_idx].mean()
                    )
                    if mean_q > contour_level:
                        mask_cwb[fill_lo:fill_hi + 1, actual_lon_idx] = 1.0
                    else:
                        mask_awb[fill_lo:fill_hi + 1, actual_lon_idx] = 1.0
                else:
                    target = mask_awb if etype_fallback == "AWB" else mask_cwb
                    target[fill_lo:fill_hi + 1, actual_lon_idx] = 1.0

    return mask_awb, mask_cwb


def detect_cutoffs(
    qgpv_2d: np.ndarray,
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    level: float,
    periodic_add: float = 120.0,
    min_exp: float = 5.0,
    max_lon_span: int = 90,
) -> typing.Tuple[typing.List[np.ndarray], typing.List[np.ndarray]]:
    """Detect closed cut-off contours, classified by mean enclosed QGPV."""
    nlat, nlon = qgpv_2d.shape
    dlon = abs(float(lon[1] - lon[0]))
    pad_cols = int(periodic_add / dlon)
    min_exp_cells = max(2, int(round(min_exp / dlon)))

    qgpv_ext = np.concatenate([qgpv_2d, qgpv_2d[:, :pad_cols]], axis=1)

    try:
        contours_raw = skimage.measure.find_contours(qgpv_ext, level)
    except Exception:
        _LOG.exception("Cut-off contour extraction failed at level %s", level)
        return [], []

    if len(contours_raw) == 0:
        return [], []

    contours_idx: typing.List[np.ndarray] = []
    closed_flags: typing.List[bool] = []
    for c in contours_raw:
        is_closed = bool(np.allclose(c[0], c[-1]))
        rounded = np.round(c).astype(int)
        xy = rounded[:, ::-1]
        unique = np.array(list(dict.fromkeys(map(tuple, xy))))
        if len(unique) >= 4:
            contours_idx.append(unique)
            closed_flags.append(is_closed)

    if len(contours_idx) == 0:
        return [], []

    exp_lons = [len(set(c[:, 0])) for c in contours_idx]
    main_idx = int(np.argmax(exp_lons))
    max_exp = exp_lons[main_idx]

    cutoff_awb: typing.List[np.ndarray] = []
    cutoff_cwb: typing.List[np.ndarray] = []
    seen_sigs: typing.Set[frozenset] = set()

    for ci, pts in enumerate(contours_idx):
        if ci == main_idx:
            continue
        if not closed_flags[ci]:
            continue

        pts_orig = pts.copy()
        pts_orig[:, 0] = pts_orig[:, 0] % nlon
        unique_lons = len(set(pts_orig[:, 0]))

        if unique_lons > max_lon_span:
            continue
        if unique_lons * dlon >= max_exp * dlon:
            continue

        centroid_lat_idx = float(np.mean(pts_orig[:, 1]))
        centroid_lat_deg = float(
            lat[min(int(round(centroid_lat_idx)), nlat - 1)]
        )
        cos_lat = max(np.cos(np.radians(centroid_lat_deg)), 0.05)
        effective_exp = unique_lons * dlon * cos_lat
        if effective_exp < min_exp:
            continue

        sig = frozenset(map(tuple, pts_orig))
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)

        unique_x = set(pts_orig[:, 0])
        qgpv_sum = 0.0
        n_cells = 0
        for xi in unique_x:
            y_at_x = pts_orig[pts_orig[:, 0] == xi, 1]
            y_lo = max(0, int(y_at_x.min()))
            y_hi = min(nlat - 1, int(y_at_x.max()))
            actual_x = int(xi) % nlon
            qgpv_sum += float(qgpv_2d[y_lo:y_hi + 1, actual_x].sum())
            n_cells += (y_hi - y_lo + 1)

        if n_cells == 0:
            continue

        mean_qgpv = qgpv_sum / n_cells
        if mean_qgpv > level:
            cutoff_cwb.append(pts_orig)
        else:
            cutoff_awb.append(pts_orig)

    return cutoff_awb, cutoff_cwb


def fill_cutoff_masks(
    *,
    cutoff_awb: typing.List[np.ndarray],
    cutoff_cwb: typing.List[np.ndarray],
    mask_awb: np.ndarray,
    mask_cwb: np.ndarray,
) -> None:
    """Fill cut-off contour interiors into existing masks (in-place)."""
    nlat, nlon = mask_awb.shape
    for pts_list, target in [(cutoff_awb, mask_awb), (cutoff_cwb, mask_cwb)]:
        for contour_pts in pts_list:
            unique_x = set(contour_pts[:, 0])
            for xi in unique_x:
                y_at_x = contour_pts[contour_pts[:, 0] == xi, 1]
                y_lo = max(0, int(y_at_x.min()))
                y_hi = min(nlat - 1, int(y_at_x.max()))
                actual_x = int(xi) % nlon
                target[y_lo:y_hi + 1, actual_x] = 1.0


def process_single_timestep(
    qgpv_2d: np.ndarray,
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    contour_level: float = DEFAULT_QGPV_LEVEL,
) -> typing.Tuple[
    typing.List[typing.Dict[str, typing.Any]], np.ndarray, np.ndarray
]:
    """Run full RWB detection pipeline on one 2-D QGPV field.

    Returns (events_list, mask_awb, mask_cwb).
    """
    nlat, nlon = len(lat), len(lon)
    empty = np.zeros((nlat, nlon), dtype=np.float32)

    contour = extract_contour(
        qgpv_2d, lat=lat, lon=lon, level=contour_level,
    )

    events: typing.List[typing.Dict[str, typing.Any]] = []
    event_types: typing.List[str] = []

    if contour is not None:
        events = compute_overturning_index(contour, lat=lat, lon=lon)
        if len(events) > 0:
            event_types = [
                classify_wave_breaking(
                    e, qgpv_2d=qgpv_2d, lat=lat, lon=lon,
                    contour_level=contour_level,
                )
                for e in events
            ]

    events_list = [
        get_event_properties(e, event_type=t, lat=lat, lon=lon)
        for e, t in zip(events, event_types)
    ]

    if len(events) > 0:
        mask_awb, mask_cwb = create_rwb_mask(
            events, lat=lat, lon=lon, types=event_types,
            qgpv_2d=qgpv_2d, contour_level=contour_level,
        )
    else:
        mask_awb, mask_cwb = empty.copy(), empty.copy()

    try:
        co_awb, co_cwb = detect_cutoffs(
            qgpv_2d, lat=lat, lon=lon, level=contour_level,
        )
        if co_awb or co_cwb:
            fill_cutoff_masks(
                cutoff_awb=co_awb, cutoff_cwb=co_cwb,
                mask_awb=mask_awb, mask_cwb=mask_cwb,
            )
    except Exception:
        _LOG.exception("Cut-off detection failed; continuing without cut-offs")

    return events_list, mask_awb, mask_cwb
