import warnings

from v_cwt import file_to_cwt_array_disk
import torch
from tqdm.auto import tqdm
import numpy as np

from pathlib import Path
import click


@click.group()
def cli():
    pass


@cli.command()
@click.argument("in_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("out_dir_prefix", type=click.Path(file_okay=False))
@click.option("--batches", type=int)
def batch_data(in_dir, out_dir_prefix, batches: int = 20):
    out_dir_prefix = Path(out_dir_prefix)
    in_dir = Path(in_dir)

    csv_files = list(in_dir.glob("*.csv"))

    for i, f in tqdm(enumerate(csv_files)):
        b = i % batches
        out_dir = out_dir_prefix / f"{b}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Path('the-link-you-want-to-create').symlink_to('the-original-file')
        (out_dir / f"{f.name}").symlink_to(f.absolute())


@cli.command()
@click.argument("in_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("out_dir", type=click.Path(file_okay=False))
@click.option("--force", default=False, is_flag=True)
@click.option("--dry", default=False, is_flag=True)
@click.option("--verbose/--silent", default=True, is_flag=True)
def cwt_dir(in_dir, out_dir, dry=False, verbose=True, force=False):
    return cwt_dir_base(in_dir, out_dir, dry=dry, verbose=verbose, force=force)


def cwt_dir_base(in_dir, out_dir, dry=False, verbose=True, force=False):
    if verbose:
        print(
            f"in_dir: {in_dir}, out_dir: {out_dir},"
            f" dry: {dry}, verbose: {verbose}, force: {force}"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_dir = Path(in_dir)
    csv_files = list(in_dir.glob("*.csv"))

    if verbose:
        my_iter = tqdm(csv_files)
    else:
        my_iter = csv_files

    for f in my_iter:
        out_file = out_dir / f"{f.stem}.pt"

        if out_file.is_file() or force:
            warnings.warn(f"skipping file {str(out_file)} because it already exists!!")
            continue

        file_to_cwt_array_disk(
            f, out_file, reshape_size=(512, 512), verbose=verbose, dry=dry
        )


def test_cwt_dir():
    cwt_dir_base("./sample_data", "./sample_data", dry=True)


if __name__ == "__main__":
    cli()
