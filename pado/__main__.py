from __future__ import annotations

import json
import os.path
from pathlib import Path
from typing import Optional

import typer
import typer.colors
from rich.console import Console
from rich.table import Table
from typer import Argument
from typer import Option

from pado._version import version as pado_version
from pado.dataset import PadoDataset
from pado.io.store import get_dataset_store_infos

# --- pado command line interface -------------------------------------
from pado.settings import dataset_registry

cli = typer.Typer(
    name="pado",
    epilog="#### pado dataset management ####",
)


@cli.command("version")
def version():
    """show the pado version"""
    typer.echo(pado_version)


@cli.command("info")
def info(
    name: Optional[str] = Option(...),
    path: Optional[Path] = Argument(
        None, exists=True, file_okay=False, dir_okay=True, readable=True
    ),
):
    """return info regarding the pado dataset"""
    if name is not None and path is not None:
        typer.echo("Can't specify both name and path", err=True)
        raise typer.Exit(1)
    elif name is None and path is None:
        typer.echo("Specify --name or path", err=True)
        raise typer.Exit(1)

    if name is not None:
        with dataset_registry() as registry:
            path, so = registry[name]

    out = PadoDataset(path, mode="r", storage_options=so).describe(
        output_format="plain_text"
    )
    typer.echo(out)


@cli.command("info-stores")
def info_stores(
    name: Optional[str] = Option(None),
    path: Optional[Path] = Argument(
        None, exists=True, file_okay=False, dir_okay=True, readable=True
    ),
):
    """return versions of all dataset providers"""
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

    ds = PadoDataset(path, mode="r", storage_options=so)
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


# --- pado dataset info -----------------------------------------------

cli_registry = typer.Typer()
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
                PadoDataset(p, mode="r")
            except ValueError:
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
                table.add_row(name, up, _so, read)
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


if __name__ == "__main__":  # pragma: no cover
    cli()
