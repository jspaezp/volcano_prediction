from v_cwt import file_to_cwt_array
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
def cwt_dir(in_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_dir = Path(in_dir)
    csv_files = list(in_dir.glob("*.csv"))

    for f in tqdm(csv_files):
        out_file = out_dir / f"{f.stem}.pt"

        if out_file.is_file():
            print(f"skipping file {str(out_file)} because it already exists!!")
            continue

        a = file_to_cwt_array(f, reshape_size=(512, 512))
        npy_arr2 = np.stack([np.rollaxis(a, 2, 0)])
        tensor = torch.from_numpy(npy_arr2)
        torch.save(tensor, out_file)


if __name__ == "__main__":
    cli()
