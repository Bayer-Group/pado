from __future__ import annotations

import itertools
import json
import os.path
import sys
from pathlib import Path
from pathlib import PurePath
from typing import TYPE_CHECKING
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
from pado.types import FilterMissing

if TYPE_CHECKING:
    from pado.dataset import PadoDataset

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
    from pado.io.store import get_dataset_store_infos

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
    from pado.dataset import PadoDataset
    from pado.registry import dataset_registry
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
    csv_column: Optional[List[str]] = Option(
        None,
        "-c",
        help="column_names (or indices if --no-header) to build target ids from",
    ),
    no_header: bool = Option(False, help="csv file has no header"),
    missing: FilterMissing = Option("warn", help="what to do iid can't be matched"),
    as_path: bool = Option(False),
    output: Optional[Path] = Option(
        None, "--out", "-o", file_okay=False, dir_okay=True, help="output path"
    ),
):
    """filter image ids"""
    from pado.dataset import PadoDataset
    from pado.images.ids import ImageId
    from pado.images.ids import filter_image_ids
    from pado.images.ids import load_image_ids_from_csv

    if not image_ids and not csv_file:
        typer.echo("must provide either --image-id some/id.svs or --csv iids.csv")
        raise typer.Exit(1)

    if no_header and not all(x.isdigit() for x in csv_column):
        typer.secho(
            "error: must provide integer column indices when using --no-header",
            err=True,
            fg="red",
        )
        raise typer.Exit(2)

    ds = _ds_from_name_or_path(
        name=name,
        path=path,
        storage_options=storage_options,
        mode="r",
    )

    if csv_file:
        targets, headers = load_image_ids_from_csv(
            csv_file,
            csv_columns=csv_column,
            no_header=no_header,
        )
    else:
        headers = None
        targets = []

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

    else:
        if not as_path:
            for iid in filtered_ids:
                typer.echo(iid.to_str())
        else:
            for iid in filtered_ids:
                typer.echo(iid.to_path(ignore_site=True))

    if headers:
        typer.secho(f"# using csv headers: {headers!r}", fg="yellow", err=True)
    raise typer.Exit(0)


@cli_ops.command(name="remote-images")
def ops_remote_ids(
    name: Optional[str] = Option(None),
    path: Optional[Path] = Argument(
        None, exists=True, file_okay=False, dir_okay=True, readable=True
    ),
    storage_options: str = Option(None),
    as_path: bool = Option(False),
):
    """image ids with remote urlpaths"""
    from fsspec.implementations.local import LocalFileSystem

    from pado.io.files import urlpathlike_get_fs_cls
    from pado.io.files import urlpathlike_to_uri

    ds = _ds_from_name_or_path(
        name=name,
        path=path,
        storage_options=storage_options,
        mode="r",
    )

    if not as_path:

        def _echo(i, _, u):
            typer.echo(f"{i}\t{u}")

    else:

        def _echo(_, i, u):
            typer.echo(f"{i.to_path(ignore_site=True)}\t{u}")

    up = ds.images.df.urlpath
    for iid in ds.index:
        str_iid = iid.to_str()
        urlpath = up[str_iid]
        fs = urlpathlike_get_fs_cls(urlpath)
        if issubclass(fs, LocalFileSystem):
            continue
        uri = urlpathlike_to_uri(urlpath, ignore_options=True)
        _echo(str_iid, iid, uri)

    raise typer.Exit(0)


@cli_ops.command(name="local-images")
def ops_local_images(
    name: Optional[str] = Option(None),
    path: Optional[Path] = Argument(
        None, exists=True, file_okay=False, dir_okay=True, readable=True
    ),
    storage_options: str = Option(None),
    as_path: bool = Option(False),
    check_missing: bool = Option(False),
):
    """image ids with remote urlpaths"""
    from fsspec.implementations.local import LocalFileSystem

    from pado.io.files import urlpathlike_get_fs_cls
    from pado.io.files import urlpathlike_get_path
    from pado.io.files import urlpathlike_to_uri

    ds = _ds_from_name_or_path(
        name=name,
        path=path,
        storage_options=storage_options,
        mode="r",
    )

    if not as_path:

        def _echo(s, _, u, m):
            c = [s, u] if m is None else [s, u, m]
            typer.echo("\t".join(c))

    else:

        def _echo(_, i, u, m):
            s = str(i.to_path(ignore_site=True))
            c = [s, u] if m is None else [s, u, m]
            typer.echo("\t".join(c))

    up = ds.images.df.urlpath
    for iid in ds.index:
        str_iid = iid.to_str()
        urlpath = up[str_iid]
        fs = urlpathlike_get_fs_cls(urlpath)
        if not issubclass(fs, LocalFileSystem):
            continue
        uri = urlpathlike_to_uri(urlpath, ignore_options=True)
        missing = None
        if check_missing:
            pth = urlpathlike_get_path(urlpath)
            missing = "missing" if not ds._fs.exists(pth) else None

        _echo(str_iid, iid, uri, missing)

    raise typer.Exit(0)


@cli_ops.command(name="update-images")
def ops_update_images(
    name: Optional[str] = Option(None),
    path: Optional[Path] = Argument(
        None, exists=True, file_okay=False, dir_okay=True, readable=True
    ),
    storage_options: str = Option(None),
    search_urlpath: str = Option(..., help="the search path"),
    glob: Optional[str] = Option(None),
    search_storage_options: str = Option(None),
    dry_run: bool = Option(False, help="don't update the dataset"),
):
    """update image urlpaths with new locations"""
    from pado.images.providers import update_image_provider_urlpaths
    from pado.io.files import fsopen
    from pado.io.files import urlpathlike_get_path
    from pado.io.files import urlpathlike_to_fs_and_path
    from pado.io.files import urlpathlike_to_uri
    from pado.io.paths import search_dataset

    is_pattern = "*" in search_urlpath
    has_glob = glob is not None

    if has_glob and "*" not in glob:
        typer.secho("provided `--glob` does not contain wildcard '*'", fg="yellow")
        raise typer.Exit(1)

    if is_pattern and has_glob:
        typer.secho(
            "Provide wildcard in `--search-urlpath` OR provide `--glob`", fg="yellow"
        )
        raise typer.Exit(1)

    if not is_pattern and not has_glob:
        typer.secho(
            "`--search-urlpath` is not a pattern: must provide --glob (i.e. `--glob '*.svs'`)",
            fg="red",
        )
        raise typer.Exit(1)

    # split the non wildcard path and glob pattern without instantiating the fs
    pth = urlpathlike_get_path(search_urlpath)
    if glob:
        pth = os.path.join(pth, glob)
    parts = list(PurePath(pth).parts)
    base_parts = list(itertools.takewhile(lambda x: "*" not in x, parts))
    glob_parts = parts[len(base_parts) :]
    if not (base_parts and glob_parts):
        typer.secho(f"ERROR: base={base_parts}, glob={glob_parts}", err=True, fg="red")
        raise typer.Exit(2)
    idx = search_urlpath.find(glob_parts[0])
    if idx >= 0:
        search_urlpath = search_urlpath[:idx]
    glob = os.path.join(*glob_parts)

    # get the filesystem and the path
    search_so = json.loads(search_storage_options or "{}")
    fs, pth = urlpathlike_to_fs_and_path(search_urlpath, storage_options=search_so)

    # get the dataset
    ds = _ds_from_name_or_path(
        name=name,
        path=path,
        storage_options=storage_options,
        mode="r",
    )

    mode: Literal["wb", "rb"]
    if dry_run:
        mode = "rb"
    else:
        mode = "wb"
    providers = search_dataset(ds, "*.image.parquet", mode=mode)

    for ip_urlpath in providers:
        ip_uri = urlpathlike_to_uri(ip_urlpath, ignore_options=True)
        typer.secho(f"updating: {ip_uri}", fg="green")
        update_image_provider_urlpaths(
            fsopen(fs, pth),
            glob,
            provider=ip_urlpath,
            progress=True,
            inplace=not dry_run,
        )
        if dry_run:
            typer.secho("dry-run: skipping write", fg="yellow")

    raise typer.Exit(0)


# --- pado dataset registry -------------------------------------------

cli_registry = typer.Typer(no_args_is_help=True)
cli.add_typer(cli_registry, name="registry")


@cli_registry.command(name="add")
def registry_add(
    name: str = Argument(...),
    location: str = Argument(...),
    storage_options: str = Option(None),
    urlpath_is_secret: bool = Option(False, help="the urlpath itself is secret"),
    secret: List[str] = Option([], help="which storage_options are secret"),
):
    """manage registries for datasets"""
    from pado.dataset import PadoDataset
    from pado.registry import dataset_registry
    from pado.registry import set_secret

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

    if secret and not set(secret).issubset(so or {}):
        typer.secho("secret not a key in storage_options", err=True, fg="red")
        raise typer.Exit(1)

    typer.echo(f"path: {location}, storage_options: {so!r}")
    try:
        _ = PadoDataset(location, mode="r", storage_options=so)
    except (ValueError, NotADirectoryError, RuntimeError) as err:
        typer.secho(f"error: {err!s}", err=True)
        typer.secho(f"PadoDataset at {location!s} is not readable", err=True)
        raise typer.Exit(1)

    if urlpath_is_secret:
        typer.secho("scrambling: urlpath", fg="green")
        _location = set_secret(
            "urlpath", location, registry_name=None, dataset_name=name
        )
    else:
        _location = location

    if so is not None:
        _so = so.copy()
        for s in secret:
            typer.secho(f"scrambling: {s}", fg="green")
            _so[s] = set_secret(s, so[s], registry_name=None, dataset_name=name)
    else:
        _so = None

    with dataset_registry() as registry:
        registry[name] = {
            "urlpath": _location,
            "storage_options": _so,
        }
    typer.secho(f"Added {name} at {location!r} with {so!r}", color=typer.colors.GREEN)


@cli_registry.command(name="list")
def registry_list(check_readable: bool = Option(False)):
    """list configured registries"""
    from pado.registry import dataset_registry
    from pado.registry import has_secrets

    if check_readable:
        from pado.dataset import PadoDataset

        def readable(name, p) -> Optional[bool]:
            try:
                PadoDataset(p.urlpath, mode="r", storage_options=p.storage_options)
            except (ValueError, NotADirectoryError, RuntimeError) as err:
                typer.secho(
                    f"[{name}] -> {err!r})",
                    fg="yellow",
                    err=True,
                )
                return False
            else:
                return True

    else:

        def readable(*_) -> Optional[bool]:
            return None

    with dataset_registry() as registry:
        name_urlpaths = list(registry.items())

    entries = []
    with typer.progressbar(name_urlpaths) as _name_urlpaths:
        for name, urlpath in _name_urlpaths:
            _has_secrets = has_secrets(urlpath)
            can_read = not _has_secrets and readable(name, urlpath)

            entries.append(
                (
                    name,
                    urlpath.urlpath,
                    urlpath.storage_options,
                    can_read,
                    _has_secrets,
                )
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
        table.add_column("Secrets")

        for name, up, so, read, sec in entries:
            _so = json.dumps(so) if so else ""
            if check_readable:
                table.add_row(name, up, _so, str(read), str(sec))
            else:
                table.add_row(name, up, _so, str(sec))
        Console().print(table)


@cli_registry.command(name="remove")
def registry_remove(
    name: str = Argument(...),
):
    """remove a registry"""
    from pado.registry import dataset_registry

    try:
        with dataset_registry() as registry:
            urlpath, storage_options = registry[name]
            del registry[name]
    except KeyError:
        typer.secho(f"Name {name!r} not registered", err=True)
        raise typer.Exit(1)
    else:
        typer.secho(
            f"Removed {name} with "
            f"urlpath={urlpath!r} and storage_options={storage_options!r}",
            color=typer.colors.GREEN,
        )


@cli_registry.command(name="input-secrets")
def registry_input_secrets():
    """enter secrets from registry"""
    from pado.registry import dataset_registry
    from pado.registry import list_secrets
    from pado.registry import set_secret

    with dataset_registry() as registry:
        for dataset_name, up_so in registry.items():
            secrets = list_secrets(up_so)
            if not secrets:
                typer.secho(f"{dataset_name} has no missing secrets", fg="green")
                continue

            for secret in secrets:
                typer.secho(f"{dataset_name} requires {secret}", fg="yellow")
                value = typer.prompt(
                    "set value (empty is skip)", default="", show_default=False
                )
                if not value.strip():
                    typer.echo("skipped.")
                    continue
                set_secret(secret, value.strip())
                typer.secho(f"{secret} = {value}")
    typer.echo("done")


# --- config ----------------------------------------------------------

cli_config = typer.Typer(no_args_is_help=True)
cli.add_typer(cli_config, name="config")


@cli_config.command(name="show")
def show():
    """display the current config"""
    from pado.settings import settings

    Console().print_json(data=settings.to_dict())


# --- helpers ---------------------------------------------------------


def _ds_from_name_or_path(
    *,
    path: Path | None,
    storage_options: str | None,
    name: str | None,
    mode: Literal["r", "w", "a", "x"],
) -> PadoDataset:
    from pado.dataset import PadoDataset
    from pado.registry import dataset_registry

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

    try:
        return PadoDataset(path, mode=mode, storage_options=so)
    except RuntimeError as err:
        typer.secho(f"{str(err)}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":  # pragma: no cover
    cli()
