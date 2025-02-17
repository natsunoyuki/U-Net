from pathlib import Path
import natsort


def list_files(directory_path: Path):
    files = list(directory_path.iterdir())
    for f in files:
        if f.name.startswith(".") is True:
            files.remove(f)
    return natsort.natsorted(files)
