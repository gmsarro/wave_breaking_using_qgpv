"""CLI to generate an animated GIF of RWB detection from QGPV data."""

import logging
import pathlib
import sys
import typing

import cartopy.crs
import cartopy.feature
import matplotlib
import matplotlib.animation
import matplotlib.patches
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


def _resolve_coordinates(
    *,
    ds: xr.Dataset,
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if "lat" in ds.dims:
        lat_full = ds["lat"].values
    elif "latitude" in ds.dims:
        nlat = ds.sizes["latitude"]
        lat_full = np.linspace(-90, 90, nlat)
    else:
        raise ValueError("Cannot find latitude dimension")

    if "lon" in ds.dims:
        lon = ds["lon"].values
    elif "longitude" in ds.dims:
        nlon = ds.sizes["longitude"]
        lon = np.linspace(0, 360 - 360 / nlon, nlon)
    else:
        raise ValueError("Cannot find longitude dimension")

    if "time" in ds.coords and np.issubdtype(ds["time"].dtype, np.datetime64):
        time_dim = "datetime"
    else:
        time_dim = "index"

    nh_mask = lat_full >= 0
    return lat_full[nh_mask], lon, nh_mask, time_dim


def _format_time_label(
    *,
    ds: xr.Dataset,
    t_idx: int,
    time_dim: str,
    year_month: str,
) -> str:
    if time_dim == "datetime":
        return str(ds["time"].values[t_idx])[:16].replace("T", " ")
    days_per_month = 30
    n_times = len(ds["time"]) if "time" in ds.dims else ds.sizes.get("time", 60)
    hours_per_step = (days_per_month * 24) / n_times
    total_hours = t_idx * hours_per_step
    day = int(total_hours // 24) + 1
    hour = int(total_hours % 24)
    return "%s-%02d %02d:00" % (year_month, day, hour)


def _load_ibtracs_track(
    *,
    ibtracs_file: pathlib.Path,
    storm_name: str,
    season: int,
) -> typing.Optional[typing.Tuple[np.ndarray, np.ndarray, typing.List[str]]]:
    with xr.open_dataset(str(ibtracs_file)) as ds:
        names = ds["name"].values
        seasons = ds["season"].values
        for i in range(len(names)):
            n = names[i].decode("ascii", errors="ignore").strip()
            if storm_name.upper() in n.upper() and int(seasons[i]) == season:
                lats = ds["lat"].values[i]
                lons = ds["lon"].values[i]
                times_raw = ds["iso_time"].values[i]
                valid = ~np.isnan(lats)
                track_lats = lats[valid]
                track_lons = lons[valid]
                track_times = [
                    times_raw[j].decode("ascii", errors="ignore").strip()[:16]
                    for j in np.where(valid)[0]
                ]
                return track_lats, track_lons, track_times
    return None


def _find_track_position(
    *,
    track_times: typing.List[str],
    track_lats: np.ndarray,
    track_lons: np.ndarray,
    frame_time: str,
) -> typing.Optional[typing.Tuple[float, float]]:
    target = frame_time.replace(" ", "T")
    for i, t in enumerate(track_times):
        t_cmp = t.replace(" ", "T")
        if t_cmp[:13] == target[:13]:
            return float(track_lats[i]), float(track_lons[i])
    return None


def cli(
    qgpv_file: typing.Annotated[
        pathlib.Path,
        typer.Option(help="Path to a single QGPV NetCDF file"),
    ],
    output_directory: typing.Annotated[
        pathlib.Path,
        typer.Option(help="Directory to write the animated GIF"),
    ],
    output_name: typing.Annotated[
        str,
        typer.Option(help="Output filename"),
    ] = "rwb_animation.gif",
    title_prefix: typing.Annotated[
        str,
        typer.Option(help="Title prefix for the figure"),
    ] = "RWB Detection",
    year_month: typing.Annotated[
        str,
        typer.Option(help="Year-month string for time labels, e.g. 2014-11"),
    ] = "",
    start_index: typing.Annotated[
        int,
        typer.Option(help="First time index to include"),
    ] = 0,
    end_index: typing.Annotated[
        typing.Optional[int],
        typer.Option(help="Last time index (exclusive; default: all)"),
    ] = None,
    ibtracs_file: typing.Annotated[
        typing.Optional[pathlib.Path],
        typer.Option(help="IBTrACS NetCDF file for storm track overlay"),
    ] = None,
    storm_name: typing.Annotated[
        str,
        typer.Option(help="Storm name to look up in IBTrACS"),
    ] = "",
    storm_season: typing.Annotated[
        int,
        typer.Option(help="Storm season (year) for IBTrACS lookup"),
    ] = 0,
    central_longitude: typing.Annotated[
        float,
        typer.Option(help="Central longitude for the polar stereographic projection"),
    ] = 180.0,
    fps: typing.Annotated[
        int,
        typer.Option(help="Frames per second"),
    ] = 3,
    dpi: typing.Annotated[
        int,
        typer.Option(help="Resolution in dots per inch"),
    ] = 120,
) -> None:
    """Generate an animated GIF of RWB detection from a QGPV file."""
    output_directory.mkdir(parents=True, exist_ok=True)

    if not qgpv_file.is_file():
        print("QGPV file not found: %s" % qgpv_file)
        raise typer.Abort()

    if not year_month:
        stem = qgpv_file.stem.replace("_qgpv", "")
        year_month = stem.replace("_", "-")

    track_lats: typing.Optional[np.ndarray] = None
    track_lons: typing.Optional[np.ndarray] = None
    track_times: typing.Optional[typing.List[str]] = None
    if ibtracs_file is not None and storm_name and storm_season > 0:
        result = _load_ibtracs_track(
            ibtracs_file=ibtracs_file,
            storm_name=storm_name,
            season=storm_season,
        )
        if result is not None:
            track_lats, track_lons, track_times = result
            print("Loaded %s %d track: %d points" % (storm_name, storm_season, len(track_lats)))
        else:
            print("WARNING: %s %d not found in IBTrACS" % (storm_name, storm_season))

    with xr.open_dataset(str(qgpv_file)) as ds:
        lat_nh, lon, nh_mask, time_dim = _resolve_coordinates(ds=ds)
        n_times = ds.sizes.get("time", len(ds["time"]))
        t_end = min(end_index, n_times) if end_index is not None else n_times
        indices = list(range(start_index, t_end))

        if len(indices) == 0:
            print("No valid time indices.")
            raise typer.Abort()

        print("Processing %d timesteps (%d to %d)" % (len(indices), indices[0], indices[-1]))

        frames: typing.List[typing.Tuple[str, np.ndarray, np.ndarray, np.ndarray, int, int]] = []
        for i, t_idx in enumerate(indices):
            qgpv_nh = ds["qgpv"].isel(height=HEIGHT_IDX, time=t_idx).values[nh_mask, :]
            time_str = _format_time_label(
                ds=ds, t_idx=t_idx, time_dim=time_dim, year_month=year_month,
            )

            events_list, mask_awb, mask_cwb = detection.process_single_timestep(
                qgpv_nh, lat=lat_nh, lon=lon,
            )
            n_awb = sum(1 for e in events_list if e["type"] == "AWB")
            n_cwb = sum(1 for e in events_list if e["type"] == "CWB")
            frames.append((time_str, qgpv_nh, mask_awb, mask_cwb, n_awb, n_cwb))

            sys.stdout.write("\r  [%d/%d] %s  %d AWB, %d CWB" % (
                i + 1, len(indices), time_str, n_awb, n_cwb,
            ))
            sys.stdout.flush()
        print()

    lon_grid, lat_grid = np.meshgrid(lon, lat_nh)
    qgpv_levels = np.linspace(-1e-4, 3e-4, 21)

    proj = cartopy.crs.NorthPolarStereo(central_longitude=central_longitude)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    legend_elements = [
        matplotlib.patches.Patch(facecolor="red", alpha=0.5, label="AWB (low-PV filament)"),
        matplotlib.patches.Patch(facecolor="blue", alpha=0.5, label="CWB (high-PV filament)"),
        plt.Line2D([0], [0], color="black", linewidth=2.5, label="QGPV contour"),
    ]
    if track_lats is not None:
        legend_elements.append(
            plt.Line2D([0], [0], color="lime", linewidth=2, marker="*",
                       markersize=10, label="%s track" % storm_name),
        )

    def _draw_frame(frame_idx: int) -> None:
        ax.clear()
        time_str, qgpv_nh, mask_awb, mask_cwb, n_awb, n_cwb = frames[frame_idx]

        ax.set_extent([-180, 180, 20, 90], crs=cartopy.crs.PlateCarree())

        ax.contourf(
            lon_grid, lat_grid, qgpv_nh,
            levels=qgpv_levels, cmap="RdBu_r",
            transform=cartopy.crs.PlateCarree(), extend="both",
        )

        if mask_awb.any():
            awb_masked = np.ma.masked_where(mask_awb == 0, mask_awb)
            ax.pcolormesh(
                lon_grid, lat_grid, awb_masked,
                cmap="Reds", vmin=0, vmax=1.5, alpha=0.6,
                transform=cartopy.crs.PlateCarree(), shading="auto",
            )

        if mask_cwb.any():
            cwb_masked = np.ma.masked_where(mask_cwb == 0, mask_cwb)
            ax.pcolormesh(
                lon_grid, lat_grid, cwb_masked,
                cmap="Blues", vmin=0, vmax=1.5, alpha=0.6,
                transform=cartopy.crs.PlateCarree(), shading="auto",
            )

        ax.contour(
            lon_grid, lat_grid, qgpv_nh,
            levels=[detection.DEFAULT_QGPV_LEVEL],
            colors="black", linewidths=2,
            transform=cartopy.crs.PlateCarree(),
        )

        ax.coastlines(resolution="50m", color="gray", linewidth=0.5)
        ax.add_feature(cartopy.feature.LAND, facecolor="lightgray", alpha=0.3)
        ax.gridlines(linestyle="--", alpha=0.3)

        if track_lats is not None and track_lons is not None and track_times is not None:
            ax.plot(
                track_lons, track_lats, color="lime", linewidth=1.5,
                transform=cartopy.crs.PlateCarree(), zorder=9,
            )
            pos = _find_track_position(
                track_times=track_times,
                track_lats=track_lats,
                track_lons=track_lons,
                frame_time=time_str,
            )
            if pos is not None:
                ax.plot(
                    pos[1], pos[0], "g*", markersize=16,
                    transform=cartopy.crs.PlateCarree(), zorder=10,
                )

        ax.set_title(
            "%s\nQGPV = %.1e s$^{-1}$ | %s\n%d AWB, %d CWB" % (
                title_prefix, detection.DEFAULT_QGPV_LEVEL, time_str, n_awb, n_cwb,
            ),
            fontsize=13,
        )

    _draw_frame(0)
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=10)
    plt.tight_layout(rect=(0, 0.06, 1, 0.95))

    print("Rendering %d-frame GIF at %d fps ..." % (len(frames), fps))
    anim = matplotlib.animation.FuncAnimation(
        fig, _draw_frame, frames=len(frames), interval=1000 // fps,
    )
    outpath = output_directory / output_name
    anim.save(str(outpath), writer="pillow", fps=fps, dpi=dpi)
    plt.close()
    print("Saved: %s (%.1f MB)" % (outpath, outpath.stat().st_size / 1e6))


if __name__ == "__main__":
    typer.run(cli)
