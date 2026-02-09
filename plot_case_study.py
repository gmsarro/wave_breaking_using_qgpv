"""CLI to plot a multi-panel RWB case study from QGPV data."""

import logging
import pathlib
import typing

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.util as cutil
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import typer
import xarray as xr

import detection

matplotlib.use("Agg")

_LOG = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)

HEIGHT_IDX: int = 19


def cli(
    qgpv_file: typing.Annotated[
        pathlib.Path,
        typer.Option(help="Path to a single QGPV NetCDF file"),
    ],
    output_directory: typing.Annotated[
        pathlib.Path,
        typer.Option(help="Directory to write the case study PNG"),
    ],
    time_indices: typing.Annotated[
        typing.Optional[str],
        typer.Option(
            help="Comma-separated time indices to plot (default: 6 evenly spaced)",
        ),
    ] = None,
    output_name: typing.Annotated[
        str,
        typer.Option(help="Output filename"),
    ] = "case_study_rwb.png",
    title_prefix: typing.Annotated[
        str,
        typer.Option(help="Title prefix for the figure"),
    ] = "RWB Case Study",
    marker_lon: typing.Annotated[
        typing.Optional[float],
        typer.Option(help="Longitude for a marker (e.g. storm centre)"),
    ] = None,
    marker_lat: typing.Annotated[
        typing.Optional[float],
        typer.Option(help="Latitude for a marker"),
    ] = None,
) -> None:
    """Plot a 6-panel RWB detection case study from a QGPV file."""
    output_directory.mkdir(parents=True, exist_ok=True)

    if not qgpv_file.is_file():
        print("QGPV file not found: %s" % qgpv_file)
        raise typer.Abort()

    with xr.open_dataset(str(qgpv_file)) as ds:
        lat_full = ds["lat"].values
        lon = ds["lon"].values
        nh_mask = lat_full >= 0
        lat_nh = lat_full[nh_mask]
        n_times = len(ds["time"])
        time_vals = ds["time"].values

        if time_indices is not None:
            indices = [int(x.strip()) for x in time_indices.split(",")]
        else:
            step = max(1, n_times // 6)
            indices = list(range(0, n_times, step))[:6]

        indices = [i for i in indices if 0 <= i < n_times]
        if len(indices) == 0:
            print("No valid time indices. File has %d timesteps." % n_times)
            raise typer.Abort()

        sample_data: typing.List[typing.Tuple[str, np.ndarray, np.ndarray, np.ndarray, typing.List]] = []
        for t_idx in indices:
            qgpv_nh = ds["qgpv"].isel(height=HEIGHT_IDX, time=t_idx).values[nh_mask, :]
            time_str = str(time_vals[t_idx])[:16].replace("T", " ")

            events_list, mask_awb, mask_cwb = detection.process_single_timestep(
                qgpv_nh, lat=lat_nh, lon=lon,
            )
            sample_data.append((time_str, qgpv_nh, mask_awb, mask_cwb, events_list))

    n_panels = min(6, len(sample_data))
    nrows = 2 if n_panels > 3 else 1
    ncols = min(3, n_panels)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 5.5 * nrows),
        subplot_kw={"projection": ccrs.NorthPolarStereo()},
    )
    axes_flat: np.ndarray = np.array(axes).flatten()

    lon_cyclic = np.append(lon, lon[0] + 360.0)
    lon_grid, lat_grid = np.meshgrid(lon_cyclic, lat_nh)

    def _add_cyclic(field: np.ndarray) -> np.ndarray:
        return np.concatenate([field, field[:, :1]], axis=1)

    for ax_idx in range(n_panels):
        time_str, qgpv_nh, mask_awb, mask_cwb, evts = sample_data[ax_idx]
        ax = axes_flat[ax_idx]
        ax.set_extent([-180, 180, 20, 90], crs=ccrs.PlateCarree())

        qgpv_cyc = _add_cyclic(qgpv_nh)
        qgpv_levels = np.linspace(-1e-4, 3e-4, 21)
        ax.contourf(
            lon_grid, lat_grid, qgpv_cyc,
            levels=qgpv_levels, cmap="RdBu_r",
            transform=ccrs.PlateCarree(), extend="both",
        )

        if mask_awb.any():
            awb_cyc = _add_cyclic(mask_awb)
            awb_masked = np.ma.masked_where(awb_cyc == 0, awb_cyc)
            ax.pcolormesh(
                lon_grid, lat_grid, awb_masked,
                cmap="Reds", vmin=0, vmax=1.5, alpha=0.6,
                transform=ccrs.PlateCarree(), shading="auto",
            )

        if mask_cwb.any():
            cwb_cyc = _add_cyclic(mask_cwb)
            cwb_masked = np.ma.masked_where(cwb_cyc == 0, cwb_cyc)
            ax.pcolormesh(
                lon_grid, lat_grid, cwb_masked,
                cmap="Blues", vmin=0, vmax=1.5, alpha=0.6,
                transform=ccrs.PlateCarree(), shading="auto",
            )

        ax.contour(
            lon_grid, lat_grid, qgpv_cyc,
            levels=[detection.DEFAULT_QGPV_LEVEL],
            colors="black", linewidths=2,
            transform=ccrs.PlateCarree(),
        )

        ax.coastlines(resolution="50m", color="gray", linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
        ax.gridlines(linestyle="--", alpha=0.3)

        if marker_lon is not None and marker_lat is not None:
            ax.plot(
                marker_lon, marker_lat, "g*", markersize=14,
                transform=ccrs.PlateCarree(), zorder=10,
            )

        n_awb = sum(1 for e in evts if e["type"] == "AWB")
        n_cwb = sum(1 for e in evts if e["type"] == "CWB")
        ax.set_title("%s\n%d AWB, %d CWB" % (time_str, n_awb, n_cwb), fontsize=11)

    for ax_idx in range(n_panels, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    legend_elements = [
        mpatches.Patch(facecolor="red", alpha=0.5, label="AWB (low-PV filament)"),
        mpatches.Patch(facecolor="blue", alpha=0.5, label="CWB (high-PV filament)"),
        plt.Line2D([0], [0], color="black", linewidth=2.5, label="QGPV contour"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=11)

    fig.suptitle(
        "%s\nQGPV = %.1e s$^{-1}$" % (title_prefix, detection.DEFAULT_QGPV_LEVEL),
        fontsize=13,
    )

    plt.tight_layout(rect=(0, 0.06, 1, 0.93))
    outpath = output_directory / output_name
    plt.savefig(str(outpath), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: %s" % outpath)


if __name__ == "__main__":
    typer.run(cli)
