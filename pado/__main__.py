from __future__ import annotations

import json
import operator
import os.path
import sys
from pathlib import Path
from pathlib import PurePath
from typing import List
from typing import Optional

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import typer
import typer.colors
from rich.console import Console
from rich.table import Table
from typer import Argument
from typer import Option

from pado._version import version as pado_version
from pado.dataset import PadoDataset
from pado.images.ids import FilterMissing
from pado.images.ids import ImageId
from pado.images.ids import filter_image_ids
from pado.io.store import get_dataset_store_infos
from pado.settings import dataset_registry

# --- pado command line interface -------------------------------------

cli = typer.Typer(
    name="pado", epilog="#### pado dataset management ####", no_args_is_help=True
)


@cli.command("version")
def version():
    """show the pado version"""
    typer.echo(pado_version)


@cli.command("info", no_args_is_help=True)
def info(
    name: Optional[str] = Option(None),
    path: Optional[Path] = Argument(
        None, exists=True, file_okay=False, dir_okay=True, readable=True
    ),
    storage_options: str = Option(None),
):
    """return info regarding the pado dataset"""
    try:
        ds = _ds_from_name_or_path(
            name=name,
            path=path,
            storage_options=storage_options,
            mode="r",
        )
    except ValueError as err:
        typer.echo(f"ERROR: {err}", err=True)
        raise typer.Exit(1)
    else:
        out = ds.describe(output_format="plain_text")
        typer.echo(out)
        raise typer.Exit(0)


@cli.command("stores", no_args_is_help=True)
def stores(
    name: Optional[str] = Option(None),
    path: Optional[Path] = Argument(
        None, exists=True, file_okay=False, dir_okay=True, readable=True
    ),
    storage_options: str = Option(None),
):
    """return versions of all dataset providers"""
    ds = _ds_from_name_or_path(
        name=name,
        path=path,
        storage_options=storage_options,
        mode="r",
    )
    store_infos = get_dataset_store_infos(ds)

    table = Table(title="Dataset Store Version")
    table.add_column("Filename", justify="right")
    table.add_column("Type", justify="left")
    table.add_column("Store", justify="left", no_wrap=True)
    table.add_column("Provider", justify="left", no_wrap=True)
    table.add_column("Identifier", justify="left")
    table.add_column("Data", justify="left")

    for pth, si in store_infos.items():
        sv, pv = (0, 0) if si.store_version is None else si.store_version
        di, dv = ("-", 0) if si.data_version is None else si.data_version
        table.add_row(
            os.path.basename(pth),
            si.store_type.value,
            str(sv),
            str(pv),
            di,
            str(dv),
        )
    Console().print(table)


@cli.command("copy", no_args_is_help=True)
def copy(
    src: str = Option(..., metavar="from"),
    dst: str = Option(...),
    image_providers: bool = Option(False),
    metadata_providers: bool = Option(False),
    annotation_providers: bool = Option(False),
    image_prediction_providers: bool = Option(False),
    images_path: Optional[str] = Option(None),
    image_predictions_path: Optional[str] = Option(None),
    keep_individual_providers: bool = Option(True),
):
    """copy data from dataset to dataset"""
    from pado.shutil import transfer

    with dataset_registry() as registry:
        try:
            p0, so0 = registry[src]
        except KeyError:
            typer.secho(f"Name {src!r} not registered", err=True)
            raise typer.Exit(1)
        try:
            p1, so1 = registry[dst]
        except KeyError:
            typer.secho(f"Name {dst!r} not registered", err=True)
            raise typer.Exit(1)

    ds0 = PadoDataset(p0, mode="r", storage_options=so0)
    ds1 = PadoDataset(p1, mode="a", storage_options=so1)

    def progress_callback(name):
        typer.secho(f"copying: {name}")

    transfer(
        ds0,
        ds1,
        image_providers=image_providers,
        metadata_providers=metadata_providers,
        annotation_providers=annotation_providers,
        image_prediction_providers=image_prediction_providers,
        images_path=images_path,
        image_predictions_path=image_predictions_path,
        keep_individual_providers=keep_individual_providers,
        progress_callback=progress_callback,
    )


# --- pado dataset operations -----------------------------------------

cli_ops = typer.Typer(no_args_is_help=True)
cli.add_typer(cli_ops, name="ops", help="common operations")


@cli_ops.command(name="list-ids")
def ops_list_ids(
    name: Optional[str] = Option(None),
    path: Optional[Path] = Argument(
        None, exists=True, file_okay=False, dir_okay=True, readable=True
    ),
    storage_options: str = Option(None),
    as_path: bool = Option(False),
):
    """list image ids in dataset"""
    ds = _ds_from_name_or_path(
        name=name,
        path=path,
        storage_options=storage_options,
        mode="r",
    )
    if not as_path:
        for iid in ds.index:
            typer.echo(iid.to_str())
    else:
        for iid in ds.index:
            typer.echo(iid.to_path(ignore_site=True))
    raise typer.Exit(0)


@cli_ops.command(name="filter-ids")
def ops_filter_ids(
    name: Optional[str] = Option(None),
    path: Optional[Path] = Argument(
        None, exists=True, file_okay=False, dir_okay=True, readable=True
    ),
    storage_options: str = Option(None),
    image_ids: Optional[List[str]] = Option(
        None, "--image-id", "-i", help="str encoded ImageId or 'some/path/file.svs'"
    ),
    csv_file: Optional[Path] = Option(None, "--csv", help="path to csv file"),
    csv_column: Optional[List[int]] = Option(
        None, "-c", help="columns to build target ids from"
    ),
    missing: FilterMissing = Option("warn", help="what to do iid can't be matched"),
    as_path: bool = Option(False),
    output: Optional[Path] = Option(
        None, "--out", "-o", file_okay=False, dir_okay=True, help="output path"
    ),
):
    """list image ids in dataset"""

    if not image_ids and not csv_file:
        typer.echo("must provide either --image-id some/id.svs or --csv iids.csv")
        raise typer.Exit(1)

    ds = _ds_from_name_or_path(
        name=name,
        path=path,
        storage_options=storage_options,
        mode="r",
    )

    targets = []

    if csv_file:
        import csv

        # get selectors for columns
        if not csv_column:
            get_cells = operator.itemgetter(slice(None))
        elif len(csv_column) == 1:
            get_cells = lambda r, idx=csv_column[0]: (r[idx],)  # noqa: E731
        else:
            get_cells = operator.itemgetter(*csv_column)

        # collect targets
        with csv_file.open(mode="r", newline="") as f:
            # todo: might have to expose some config for `DictReader` here
            for row in csv.DictReader(f):
                cells = tuple(row.values())
                targets.append(get_cells(cells))

    # add cli provided image ids
    for t in image_ids:
        try:
            iid = ImageId.from_str(t)
        except ValueError:
            iid = PurePath(t).parts
        targets.append(iid)

    image_ids = set(ds.index)
    filtered_ids = sorted(filter_image_ids(image_ids, targets, missing=missing))

    if output is not None:
        typer.echo(f"Filtered {len(filtered_ids)} of {len(image_ids)} image ids")
        filtered_ds = PadoDataset(output, mode="x")
        filtered_ds.ingest_obj(ds.filter(filtered_ids))
        typer.echo(f"Wrote new pado dataset to path: '{output}'")
        raise typer.Exit(0)

    else:
        if not as_path:
            for iid in filtered_ids:
                typer.echo(iid.to_str())
        else:
            for iid in filtered_ids:
                typer.echo(iid.to_path(ignore_site=True))
        raise typer.Exit(0)


# --- pado dataset registry -------------------------------------------

cli_registry = typer.Typer(no_args_is_help=True)
cli.add_typer(cli_registry, name="registry")


@cli_registry.command(name="add")
def registry_add(
    name: str = Argument(...),
    location: str = Argument(...),
    storage_options: str = Option(None),
):
    """manage registries for datasets"""
    so = None
    if storage_options:
        try:
            so = json.loads(storage_options)
        except json.JSONDecodeError:
            typer.secho(
                f"provided incorrect JSON as storage_options:\n{storage_options!r}",
                err=True,
            )
            raise typer.Exit(1)
    try:
        print(location, so)
        _ = PadoDataset(location, mode="r", storage_options=so)
    except ValueError as err:
        typer.secho(f"error: {err!s}", err=True)
        typer.secho(f"PadoDataset at {location!s} is not readable", err=True)
        raise typer.Exit(1)
    with dataset_registry() as registry:
        registry[name] = {
            "urlpath": location,
            "storage_options": so,
        }
    typer.secho(f"Added {name} at {location!r} with {so!r}", color=typer.colors.GREEN)


@cli_registry.command(name="list")
def registry_list(check_readable: bool = Option(False)):
    """list configured registries"""
    with dataset_registry() as registry:
        name_urlpaths = list(registry.items())

    def readable(p) -> Optional[bool]:
        if not check_readable:
            return None
        else:
            try:
                PadoDataset(p.urlpath, mode="r", storage_options=p.storage_options)
            except (ValueError, NotADirectoryError, RuntimeError):
                return False
            else:
                return True

    entries = []
    with typer.progressbar(name_urlpaths) as _name_urlpaths:
        for name, urlpath in _name_urlpaths:
            entries.append(
                (name, urlpath.urlpath, urlpath.storage_options, readable(urlpath))
            )

    if not entries:
        typer.secho("No datasets registered", color=typer.colors.YELLOW, err=True)
    else:
        table = Table(title="Registered Datasets")
        table.add_column("Name", justify="left", no_wrap=True)
        table.add_column("Location", justify="left")
        table.add_column("Storage Options", justify="left")
        if check_readable:
            table.add_column("Readable")

        for name, up, so, read in entries:
            _so = json.dumps(so) if so else ""
            if check_readable:
                table.add_row(name, up, _so, str(read))
            else:
                table.add_row(name, up, _so)
        Console().print(table)


@cli_registry.command(name="remove")
def registry_remove(
    name: str = Argument(...),
):
    """remove a registry"""
    try:
        with dataset_registry() as registry:
            urlpath, storage_options = registry[name]
            del registry[name]
    except KeyError:
        typer.secho(f"Name {name!r} not registered", err=True)
    else:
        typer.secho(
            f"Removed {name} with "
            f"urlpath={urlpath!r} and storage_options={storage_options!r}",
            color=typer.colors.GREEN,
        )


# --- helpers ---------------------------------------------------------


def _ds_from_name_or_path(
    *,
    path: Path | None,
    storage_options: str | None,
    name: str | None,
    mode: Literal["r", "w", "a", "x"],
) -> PadoDataset:

    if name is not None and path is not None:
        typer.echo("Can't specify both name and path", err=True)
        raise typer.Exit(1)
    elif name is None and path is None:
        typer.echo("Specify --name or path", err=True)
        raise typer.Exit(1)

    if name is not None:
        with dataset_registry() as registry:
            try:
                path, so = registry[name]
            except KeyError:
                typer.secho(f"Name {name!r} not registered", err=True)
                raise typer.Exit(1)
    else:
        so = json.loads(storage_options or "{}")

    return PadoDataset(path, mode=mode, storage_options=so)


if __name__ == "__main__":  # pragma: no cover
    cli()
