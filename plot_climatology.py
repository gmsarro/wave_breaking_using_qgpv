"""CLI to produce a DJF RWB frequency climatology map from mask files."""

import glob
import logging
import pathlib
import typing

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.util as cutil
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import typer
import xarray as xr

import detection

matplotlib.use("Agg")

_LOG = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)


def cli(
    mask_directory: typing.Annotated[
        pathlib.Path,
        typer.Option(help="Directory containing rwb_masks_YYYY_MM.nc files"),
    ],
    output_directory: typing.Annotated[
        pathlib.Path,
        typer.Option(help="Directory to write climatology PNG"),
    ],
    qgpv_directory: typing.Annotated[
        typing.Optional[pathlib.Path],
        typer.Option(help="QGPV directory for mean contour overlay (optional)"),
    ] = None,
    smooth_sigma: typing.Annotated[
        float,
        typer.Option(help="Gaussian smoothing sigma for frequency map"),
    ] = 1.0,
) -> None:
    """Compute and plot DJF RWB frequency climatology from mask NetCDFs."""
    output_directory.mkdir(parents=True, exist_ok=True)
    height_idx = 19

    djf_months = {12, 1, 2}
    mask_files = sorted(glob.glob(str(mask_directory / "rwb_masks_*.nc")))

    djf_files: typing.List[pathlib.Path] = []
    for f in mask_files:
        base = pathlib.Path(f).stem.replace("rwb_masks_", "")
        parts = base.split("_")
        if len(parts) == 2 and int(parts[1]) in djf_months:
            djf_files.append(pathlib.Path(f))

    print("DJF Climatology from mask files")
    print("Mask directory:    %s" % mask_directory)
    print("DJF mask files:    %d" % len(djf_files))

    if len(djf_files) == 0:
        print("No DJF mask files found. Exiting.")
        raise typer.Abort()

    with xr.open_dataset(str(djf_files[0])) as ds_tmp:
        lat = ds_tmp["lat"].values
        lon = ds_tmp["lon"].values

    awb_count = np.zeros((len(lat), len(lon)), dtype=np.float64)
    cwb_count = np.zeros((len(lat), len(lon)), dtype=np.float64)
    qgpv_sum = np.zeros((len(lat), len(lon)), dtype=np.float64)
    n_timesteps = 0
    has_qgpv_mean = False

    for fi, fpath in enumerate(djf_files):
        base = fpath.stem.replace("rwb_masks_", "")
        print("[%3d/%d] %s" % (fi + 1, len(djf_files), base), flush=True)

        with xr.open_dataset(str(fpath)) as ds:
            awb_count += ds["rwb_mask_awb"].values.sum(axis=0)
            cwb_count += ds["rwb_mask_cwb"].values.sum(axis=0)
            n_timesteps += len(ds["time"])

        if qgpv_directory is not None:
            qgpv_path = qgpv_directory / ("%s_qgpv.nc" % base)
            if qgpv_path.exists():
                with xr.open_dataset(str(qgpv_path)) as ds_q:
                    lat_full = ds_q["lat"].values
                    nh_mask = lat_full >= 0
                    for t_idx in range(len(ds_q["time"])):
                        qgpv_sum += ds_q["qgpv"].isel(
                            height=height_idx, time=t_idx,
                        ).values[nh_mask, :]
                    has_qgpv_mean = True

    print("Total DJF timesteps: %d" % n_timesteps)

    awb_freq = 100.0 * awb_count / n_timesteps
    cwb_freq = 100.0 * cwb_count / n_timesteps
    total_freq = 100.0 * (awb_count + cwb_count) / n_timesteps
    total_smooth = scipy.ndimage.gaussian_filter(total_freq, sigma=smooth_sigma)

    print("Max AWB freq: %.1f%%" % awb_freq.max())
    print("Max CWB freq: %.1f%%" % cwb_freq.max())
    print("Max Total freq: %.1f%%" % total_freq.max())

    fig = plt.figure(figsize=(10, 10))
    ax: plt.Axes = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([-180, 180, 20, 90], crs=ccrs.PlateCarree())  # type: ignore[attr-defined]

    total_cyclic, lon_cyclic = cutil.add_cyclic_point(total_smooth, coord=lon)
    lon_grid, lat_grid = np.meshgrid(lon_cyclic, lat)

    main_colors = [
        "#0000FF", "#00BFFF", "#00FFFF", "#7FFF00",
        "#FFFF00", "#FF8C00", "#FF0000", "#8B0000",
    ]
    main_cmap = mcolors.LinearSegmentedColormap.from_list("rwb_main", main_colors, N=256)
    levels = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
    n_intervals = len(levels) - 1
    color_list = ["#D0D0D0"]
    for ci in range(n_intervals - 1):
        color_list.append(main_cmap(ci / (n_intervals - 2)))
    cmap = mcolors.ListedColormap(color_list)
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    cf = ax.contourf(
        lon_grid, lat_grid, total_cyclic,
        levels=levels, cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(), extend="max",
    )

    if has_qgpv_mean and n_timesteps > 0:
        qgpv_mean = qgpv_sum / n_timesteps
        qgpv_cyclic, _ = cutil.add_cyclic_point(qgpv_mean, coord=lon)
        ax.contour(
            lon_grid, lat_grid, qgpv_cyclic,
            levels=[detection.DEFAULT_QGPV_LEVEL],
            colors="black", linewidths=2,
            transform=ccrs.PlateCarree(),
        )

    ax.coastlines(resolution="50m", color="gray", linewidth=0.5)  # type: ignore[attr-defined]
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)  # type: ignore[attr-defined]

    theta = np.linspace(0, 2 * np.pi, 100)
    for lat_c in [20, 40, 60, 80]:
        ax.plot(
            np.degrees(theta), [lat_c] * len(theta),
            color="gray", linewidth=0.5, linestyle="--",
            transform=ccrs.PlateCarree(),
        )

    cbar = plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.08)
    cbar.set_label("Occurrence frequency (%)", fontsize=11)

    years_in_djf = sorted(set(
        int(pathlib.Path(f).stem.replace("rwb_masks_", "").split("_")[0])
        for f in djf_files
    ))
    year_lo, year_hi = years_in_djf[0], years_in_djf[-1]

    ax.set_title(
        "ERA5 DJF %d\u2013%d | %d timesteps\n"
        "RWB Frequency | QGPV = %.1e s$^{-1}$\n"
        "Pair-fill + cut-offs (min_exp=5\u00b0)"
        % (year_lo, year_hi, n_timesteps, detection.DEFAULT_QGPV_LEVEL),
        fontsize=12,
    )

    plt.tight_layout()
    outpath = output_directory / "rwb_climatology_final.png"
    plt.savefig(str(outpath), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: %s" % outpath)


if __name__ == "__main__":
    typer.run(cli)
