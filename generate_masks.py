"""CLI to generate per-month RWB mask NetCDF files from QGPV data."""

import datetime
import glob
import logging
import pathlib
import time
import typing

import numpy as np
import typer
import xarray as xr

import detection

_LOG = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)

HEIGHT_IDX: int = 19


def cli(
    qgpv_directory: typing.Annotated[
        pathlib.Path,
        typer.Option(help="Directory containing YYYY_MM_qgpv.nc files"),
    ],
    output_directory: typing.Annotated[
        pathlib.Path,
        typer.Option(help="Directory to write rwb_masks_YYYY_MM.nc files"),
    ],
    contour_level: typing.Annotated[
        float,
        typer.Option(help="QGPV contour level in s^-1"),
    ] = detection.DEFAULT_QGPV_LEVEL,
    overwrite: typing.Annotated[
        bool,
        typer.Option(help="Overwrite existing mask files"),
    ] = False,
) -> None:
    """Process all QGPV files and save per-month RWB mask NetCDFs."""
    output_directory.mkdir(parents=True, exist_ok=True)

    all_qgpv = sorted(glob.glob(str(qgpv_directory / "*_qgpv.nc")))
    print("RWB Mask Generation")
    print("QGPV contour level = %.1e s-1" % contour_level)
    print("Total QGPV files:  %d" % len(all_qgpv))
    print("Output directory:  %s" % output_directory)

    if len(all_qgpv) == 0:
        print("No QGPV files found. Exiting.")
        raise typer.Abort()

    to_process: typing.List[typing.Tuple[str, str, pathlib.Path]] = []
    for fpath in all_qgpv:
        base = pathlib.Path(fpath).stem.replace("_qgpv", "")
        mask_path = output_directory / ("rwb_masks_%s.nc" % base)
        if overwrite or not mask_path.exists():
            to_process.append((base, fpath, mask_path))

    print("Already processed: %d" % (len(all_qgpv) - len(to_process)))
    print("To process:        %d" % len(to_process))

    if len(to_process) == 0:
        print("Nothing to do!")
        return

    with xr.open_dataset(str(to_process[0][1])) as ds_tmp:
        lat_full = ds_tmp["lat"].values
        lon = ds_tmp["lon"].values

    nh_mask = lat_full >= 0
    lat_nh = lat_full[nh_mask]

    t_start = time.time()

    for idx, (base, fpath, mask_path) in enumerate(to_process):
        print("[%3d/%d] %s ..." % (idx + 1, len(to_process), base), end="", flush=True)

        try:
            with xr.open_dataset(str(fpath)) as ds:
                n_times = len(ds["time"])
                time_vals = ds["time"].values
                month_awb = np.zeros(
                    (n_times, len(lat_nh), len(lon)), dtype=np.float32,
                )
                month_cwb = np.zeros(
                    (n_times, len(lat_nh), len(lon)), dtype=np.float32,
                )

                for t_idx in range(n_times):
                    qgpv_nh = ds["qgpv"].isel(
                        height=HEIGHT_IDX, time=t_idx,
                    ).values[nh_mask, :]

                    _, mask_awb, mask_cwb = detection.process_single_timestep(
                        qgpv_nh,
                        lat=lat_nh,
                        lon=lon,
                        contour_level=contour_level,
                    )
                    month_awb[t_idx] = mask_awb
                    month_cwb[t_idx] = mask_cwb
        except Exception:
            _LOG.exception("Failed to process %s", base)
            continue

        ds_out = xr.Dataset(
            {
                "rwb_mask_awb": (["time", "lat", "lon"], month_awb),
                "rwb_mask_cwb": (["time", "lat", "lon"], month_cwb),
            },
            coords={"time": time_vals, "lat": lat_nh, "lon": lon},
        )
        ds_out.attrs["description"] = "RWB detection masks (NH, 10 km QGPV)"
        ds_out.attrs["qgpv_contour_level"] = contour_level
        ds_out.attrs["method"] = (
            "Overturning index (Barnes & Hartmann 2012), "
            "pair-based filament detection, mean-QGPV classification, "
            "cut-offs (closed contours, min_exp>=5 deg)"
        )
        ds_out.attrs["created"] = datetime.datetime.now().isoformat()
        ds_out["rwb_mask_awb"].attrs = {
            "long_name": "Anticyclonic wave breaking mask", "units": "1",
        }
        ds_out["rwb_mask_cwb"].attrs = {
            "long_name": "Cyclonic wave breaking mask", "units": "1",
        }
        ds_out.to_netcdf(str(mask_path))

        elapsed = time.time() - t_start
        rate = (idx + 1) / elapsed * 60
        eta = (len(to_process) - idx - 1) / rate if rate > 0 else 0
        print("  %dts | %.1fm | ~%.0fm left" % (n_times, elapsed / 60, eta), flush=True)

    total_time = time.time() - t_start
    n_masks = len(list(output_directory.glob("rwb_masks_20*_*.nc")))
    print("Done! %d files in %.1f min | %d mask files total" % (
        len(to_process), total_time / 60, n_masks,
    ))


if __name__ == "__main__":
    typer.run(cli)
