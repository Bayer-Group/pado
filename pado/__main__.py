from pathlib import Path
from typing import Optional

import typer
import typer.colors
from typer import Argument
from typer import Option

from rich.console import Console
from rich.table import Table


from pado._version import version as pado_version
from pado.dataset import PadoDataset


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
    path: Optional[Path] = Argument(None, exists=True, file_okay=False, dir_okay=True, readable=True),
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
            path = registry[name]

    out = PadoDataset(path, mode="r").describe()
    typer.echo(out)


# --- pado dataset info -----------------------------------------------

cli_registry = typer.Typer()
cli.add_typer(cli_registry, name="registry")


@cli_registry.command(name="add")
def registry_add(
    name: str = Argument(...),
    location: str = Argument(...),
):
    """manage registries for datasets"""
    try:
        ds = PadoDataset(location, mode="r")
    except ValueError as err:
        typer.secho(f"error: {err!s}", err=True)
        typer.secho(f"PadoDataset at {location!s} is not readable", err=True)
        raise typer.Exit(1)
    with dataset_registry() as registry:
        registry[name] = ds.urlpath
    typer.secho(f"Added {name} at {location}", color=typer.colors.GREEN)


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
                PadoDataset(urlpath, mode="r")
            except ValueError:
                return False
            else:
                return True

    entries = []
    for name, urlpath in name_urlpaths:
        entries.append((name, urlpath, readable(urlpath)))

    if not entries:
        typer.secho("No datasets registered", color=typer.colors.YELLOW, err=True)
    else:
        table = Table(title="Registered Datasets")
        table.add_column("Name", justify="left", no_wrap=True)
        table.add_column("Location", justify="left")
        if check_readable:
            table.add_column("Readable")

        for name, urlpath, read in entries:
            if check_readable:
                table.add_row(name, urlpath, read)
            else:
                table.add_row(name, urlpath)
        Console().print(table)


@cli_registry.command(name="remove")
def registry_remove(
    name: str = Argument(...),
):
    """remove a registry"""
    try:
        with dataset_registry() as registry:
            location = registry[name]
            del registry[name]
    except KeyError:
        typer.secho(f"Name {name!r} not registered", err=True)
    else:
        typer.secho(f"Removed {name} at {location}", color=typer.colors.GREEN)


if __name__ == "__main__":  # pragma: no cover
    cli()
