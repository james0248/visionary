import io
from pathlib import Path

import grain.python as grain
import numpy as np


def build_raw_index(shards_dir: str) -> dict[tuple[str, int, str], bytes]:
    """Map (repo, episode, camera) -> mp4 bytes from the packed video shards.

    These are the same trimmed streams the latents were encoded from, so a
    latent record's start_index indexes directly into this video.
    """
    paths = sorted(str(p) for p in Path(shards_dir).glob("*.arecord"))
    if not paths:
        raise FileNotFoundError(f"No .arecord files in {shards_dir}")
    source = grain.ArrayRecordDataSource(paths)
    index: dict[tuple[str, int, str], bytes] = {}
    for i in range(len(source)):
        with np.load(io.BytesIO(source[i])) as data:
            key = (str(data["repo"]), int(data["episode"]), str(data["camera"]))
            index[key] = data["video"].tobytes()
    return index
