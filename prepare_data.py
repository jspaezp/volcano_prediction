from v_cwt import file_to_cwt_array
import torch
from tqdm import tqdm

from pathlib import Path
import click


@click.command()
@click.argument("in_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("out_dir", type=click.Path(file_okay=False))
def cli(in_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_dir = Path(in_dir)
    csv_files = list(in_dir.glob("*.csv"))

    for f in tqdm(csv_files):
        out_file = out_dir / f"{f.stem}.pt"
        assert not out_file.is_file()
        a = file_to_cwt_array(f, reshape_size=(224, 224))
        torch.save(a, out_file)


if __name__ == "__main__":
    cli()
