import argparse
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np

ENCODE_ARGS = [
    # yuv420p needs even dimensions; a no-op unless a source has an odd width or height
    "-vf",
    "scale=trunc(iw/2)*2:trunc(ih/2)*2",
    "-c:v",
    "libx264",
    "-preset",
    "fast",
    "-crf",
    "16",
    "-maxrate",
    "4M",
    "-bufsize",
    "8M",
    "-g",
    "25",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
    "-an",
]
WRIST_HINTS = ("wrist", "gripper", "hand", "claw", "ego", "arm_cam")
BLANK_BYTES_PER_FRAME = 300
MAX_LOGGED_ISSUES = 50
SHARD_MAX_BYTES = 2_000_000_000


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, **kw)


def gcs_rsync(src, dst):
    run(["gcloud", "storage", "rsync", "--recursive", str(src), str(dst)])


def gcs_exists(uri):
    return subprocess.run(["gcloud", "storage", "ls", uri], capture_output=True).returncode == 0


def gcs_mark_done(prefix):
    subprocess.run(
        ["gcloud", "storage", "cp", "-", f"{prefix}/_DONE"],
        input=b"done",
        check=True,
    )


def transcode(src, dst, extra_in=()):
    run(["ffmpeg", "-loglevel", "error", "-y", *extra_in, "-i", str(src), *ENCODE_ARGS, str(dst)])
    out = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_packets",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_read_packets",
            "-of",
            "json",
            str(dst),
        ],
        text=True,
    )
    stream = json.loads(out.stdout)["streams"][0]
    return int(stream["nb_read_packets"]), int(stream["width"]), int(stream["height"])


def looks_blank(path, n_frames):
    # LeRobot v3 pads cameras a session lacked with all-black video, valid timestamps and all
    if n_frames < 1 or Path(path).stat().st_size / n_frames >= BLANK_BYTES_PER_FRAME:
        return False
    import cv2

    cap = cv2.VideoCapture(str(path))
    read_any = False
    for fraction in (0.1, 0.5, 0.9):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n_frames * fraction))
        ok, frame = cap.read()
        if not ok:
            continue
        read_any = True
        if frame.max() > 4 or frame.std() > 1.0:
            cap.release()
            return False
    cap.release()
    return read_any


def camera_name_hint(name):
    lower = name.lower()
    return "wrist" if any(h in lower for h in WRIST_HINTS) else "fixed"


class ShardWriter:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_idx = -1
        self.tar = None
        self.index = {}
        self.shards = []
        self._roll()

    def _roll(self):
        self._close_current()
        self.shard_idx += 1
        self.path = self.out_dir / f"shard-{self.shard_idx:04d}.tar"
        self.tar = tarfile.open(self.path, "w")
        self.index = {}

    def _close_current(self):
        if self.tar is not None:
            self.tar.close()
            index_path = self.path.with_suffix(".index.json")
            index_path.write_text(json.dumps(self.index))
            self.shards.append(self.path.name)

    def add_entry(self, name, files):
        if self.path.exists() and self.path.stat().st_size > SHARD_MAX_BYTES:
            self._roll()
        entry = {}
        for fname, payload in files.items():
            data = payload if isinstance(payload, bytes) else Path(payload).read_bytes()
            info = tarfile.TarInfo(f"{name}/{fname}")
            info.size = len(data)
            import io as _io

            start = self.tar.offset
            self.tar.addfile(info, _io.BytesIO(data))
            padded = ((len(data) + 511) // 512) * 512
            header_len = self.tar.offset - start - padded
            entry[fname] = {"offset": start + header_len, "size": len(data)}
        self.index[name] = entry

    def close(self):
        self._close_current()
        self.tar = None
        return self.shards


def frames_npz_bytes(actions, state):
    import io as _io

    buf = _io.BytesIO()
    np.savez_compressed(
        buf,
        actions=np.asarray(actions, np.float32),
        state=np.asarray(state, np.float32),
    )
    return buf.getvalue()


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def read_parquet(path, columns=None):
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=columns)


def stack_column(table, name):
    column = table.column(name).combine_chunks()
    try:
        values = column.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
        return values.reshape(len(column), -1)
    except (AttributeError, ValueError):
        return np.asarray(column.to_pylist(), np.float32)


def so101_annotations(ann_dir, source):
    path = Path(ann_dir) / source.replace("__", "/") / "tasks_annotated.parquet"
    if not path.exists():
        return {}
    table = read_parquet(path)
    frame = table.to_pydict()
    episode_indices = (
        table.column("episode_index").to_pylist()
        if "episode_index" in table.column_names
        else list(range(len(frame["task"])))
    )
    return dict(zip(episode_indices, frame["task"]))


def process_so101_repo(source, raw_dir, out_dir, ann_dir, report):
    raw = Path(raw_dir)
    info = json.loads((raw / "meta" / "info.json").read_text())
    fps = int(round(info.get("fps", 30)))
    episodes = load_jsonl(raw / "meta" / "episodes.jsonl")
    annotations = so101_annotations(ann_dir, source)

    video_files = sorted(raw.glob("videos/chunk-*/observation.images.*/episode_*.mp4"))
    cameras = sorted({p.parent.name.removeprefix("observation.images.") for p in video_files})
    issues = []
    if len(video_files) != len(episodes) * len(cameras):
        issues.append(
            f"video_count={len(video_files)} episodes={len(episodes)} cameras={len(cameras)}"
        )

    writer = ShardWriter(out_dir)
    tmp = Path(out_dir) / "_tmp.mp4"
    written = 0
    blank = 0
    action_dims = set()
    for ep in episodes:
        ep_idx = ep["episode_index"]
        parquets = list(raw.glob(f"data/chunk-*/episode_{ep_idx:06d}.parquet"))
        if not parquets:
            issues.append(f"ep{ep_idx}: missing parquet")
            continue
        table = read_parquet(parquets[0])
        actions = stack_column(table, "action")
        state = stack_column(table, "observation.state")
        action_dims.add(actions.shape[1])
        language = annotations.get(ep_idx) or (ep.get("tasks") or [""])[0]
        npz = frames_npz_bytes(actions, state)
        for cam_idx, cam in enumerate(cameras):
            vids = list(
                raw.glob(f"videos/chunk-*/observation.images.{cam}/episode_{ep_idx:06d}.mp4")
            )
            if not vids:
                issues.append(f"ep{ep_idx}/{cam}: missing video")
                continue
            try:
                n_frames, width, height = transcode(vids[0], tmp)
            except subprocess.CalledProcessError as exc:
                issues.append(f"ep{ep_idx}/{cam}: transcode failed: {exc.stderr[-200:]}")
                continue
            if looks_blank(tmp, n_frames):
                blank += 1
                continue
            if abs(n_frames - len(actions)) > 2:
                issues.append(f"ep{ep_idx}/{cam}: frames={n_frames} rows={len(actions)}")
            meta = {
                "dataset": "so101",
                "source": source,
                "episode": ep_idx,
                "camera": {"index": cam_idx, "name_hint": camera_name_hint(cam), "orig_name": cam},
                "language": language,
                "fps": fps,
                "frames": n_frames,
                "width": width,
                "height": height,
                "parquet_rows": len(actions),
            }
            writer.add_entry(
                f"ep{ep_idx:06d}_cam{cam_idx:02d}",
                {"meta.json": json.dumps(meta).encode(), "video.mp4": tmp, "frames.npz": npz},
            )
            written += 1
    tmp.unlink(missing_ok=True)
    shards = writer.close()
    manifest = {
        "dataset": "so101",
        "source": source,
        "fps": fps,
        "episodes": len(episodes),
        "cameras": [
            {"index": i, "orig_name": c, "name_hint": camera_name_hint(c)}
            for i, c in enumerate(cameras)
        ],
        "entries": written,
        "blank_skipped": blank,
        "action_dims": sorted(action_dims),
        "annotated_language": bool(annotations),
        "shards": shards,
        "issue_count": len(issues),
        "issues": issues[:MAX_LOGGED_ISSUES],
    }
    (Path(out_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2))
    report.append(manifest)


def bridge_episode_table(raw):
    # stays columnar: materializing all 50k rows as dicts costs 2.5 GB per worker
    import pyarrow as pa

    return pa.concat_tables(
        [read_parquet(f) for f in sorted(raw.glob("meta/episodes/chunk-*/*.parquet"))]
    )


def bridge_camera_keys(names):
    return sorted(
        k.removeprefix("videos/").removesuffix("/chunk_index")
        for k in names
        if k.startswith("videos/") and k.endswith("/chunk_index")
    )


def bridge_units(table):
    units = []
    for cam in bridge_camera_keys(table.column_names):
        chunks = table.column(f"videos/{cam}/chunk_index").to_numpy()
        files = table.column(f"videos/{cam}/file_index").to_numpy()
        pairs = np.unique(np.stack([chunks, files], axis=1), axis=0)
        units.extend((cam, int(c), int(f)) for c, f in pairs)
    return sorted(units)


def process_bridge_video(cam, vchunk, vfile, raw_dir, out_dir, episode_table, data_cache, report):
    import pyarrow as pa

    raw = Path(raw_dir)
    info = json.loads((raw / "meta" / "info.json").read_text())
    fps = int(round(info.get("fps", 5)))
    source = f"{cam.removeprefix('observation.images.')}_c{vchunk:03d}_f{vfile:03d}"
    mask = (episode_table.column(f"videos/{cam}/chunk_index").to_numpy() == vchunk) & (
        episode_table.column(f"videos/{cam}/file_index").to_numpy() == vfile
    )
    episodes = episode_table.filter(pa.array(mask)).to_pylist()
    if not episodes:
        return
    src = raw / "videos" / cam / f"chunk-{vchunk:03d}" / f"file-{vfile:03d}.mp4"
    writer = ShardWriter(out_dir)
    tmp = Path(out_dir) / "_tmp.mp4"
    issues = []
    written = 0
    blank = 0
    cam_index = bridge_camera_keys(episode_table.column_names).index(cam)
    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        key = (ep["data/chunk_index"], ep["data/file_index"])
        if key not in data_cache:
            table = read_parquet(
                raw / "data" / f"chunk-{key[0]:03d}" / f"file-{key[1]:03d}.parquet"
            )
            data_cache.clear()
            data_cache[key] = (
                np.asarray(table.column("episode_index").to_numpy()),
                stack_column(table, "action"),
                stack_column(table, "observation.state"),
            )
        ep_column, all_actions, all_state = data_cache[key]
        mask = ep_column == ep_idx
        actions = all_actions[mask]
        state = all_state[mask]
        t0 = ep[f"videos/{cam}/from_timestamp"]
        t1 = ep[f"videos/{cam}/to_timestamp"]
        try:
            n_frames, width, height = transcode(
                src, tmp, extra_in=["-ss", str(t0), "-t", str(t1 - t0)]
            )
        except subprocess.CalledProcessError as exc:
            issues.append(f"ep{ep_idx}: transcode failed: {exc.stderr[-200:]}")
            continue
        if looks_blank(tmp, n_frames):
            blank += 1
            continue
        if abs(n_frames - len(actions)) > 2:
            issues.append(f"ep{ep_idx}: frames={n_frames} rows={len(actions)}")
        meta = {
            "dataset": "bridge",
            "source": source,
            "episode": ep_idx,
            "camera": {
                "index": cam_index,
                "name_hint": camera_name_hint(cam),
                "orig_name": cam.removeprefix("observation.images."),
            },
            "language": (ep.get("tasks") or [""])[0],
            "fps": fps,
            "frames": n_frames,
            "width": width,
            "height": height,
            "parquet_rows": int(mask.sum()),
        }
        writer.add_entry(
            f"ep{ep_idx:06d}_cam{cam_index:02d}",
            {"meta.json": json.dumps(meta).encode(), "video.mp4": tmp, "frames.npz": frames_npz_bytes(actions, state)},
        )
        written += 1
    tmp.unlink(missing_ok=True)
    shards = writer.close()
    manifest = {
        "dataset": "bridge",
        "source": source,
        "fps": fps,
        "camera": {"index": cam_index, "orig_name": cam.removeprefix("observation.images.")},
        "episodes": len(episodes),
        "entries": written,
        "blank_skipped": blank,
        "shards": shards,
        "issue_count": len(issues),
        "issues": issues[:MAX_LOGGED_ISSUES],
    }
    (Path(out_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2))
    report.append(manifest)


def process_soar_split(tfrecord_paths, split_name, out_dir, fps, report):
    import tensorflow as tf

    writer = ShardWriter(out_dir)
    tmp = Path(out_dir) / "_tmp.mp4"
    issues = []
    written = 0
    blank = 0
    dataset = tf.data.TFRecordDataset([str(p) for p in tfrecord_paths])
    for ep_idx, record in enumerate(dataset):
        try:
            feats = tf.train.Example.FromString(record.numpy()).features.feature
            num_steps = len(feats["steps/is_first"].int64_list.value)
            flat_actions = np.asarray(feats["steps/action"].float_list.value, np.float32)
            actions = flat_actions.reshape(num_steps, -1)
            flat_state = np.asarray(feats["steps/observation/state"].float_list.value, np.float32)
            state = flat_state.reshape(num_steps, -1)
            language = ""
            lang_values = feats["steps/language_instruction"].bytes_list.value
            if lang_values:
                language = lang_values[0].decode()
            success = bool(feats["episode_metadata/success"].int64_list.value[0])
            image_keys = sorted(
                k
                for k in feats
                if k.startswith("steps/observation/image") and feats[k].bytes_list.value
            )
            npz = frames_npz_bytes(actions, state)
            for cam_idx, key in enumerate(image_keys):
                jpegs = feats[key].bytes_list.value
                if len(jpegs) != num_steps:
                    issues.append(f"ep{ep_idx}/{key}: jpegs={len(jpegs)} steps={num_steps}")
                    continue
                frames_dir = Path(out_dir) / "_frames"
                shutil.rmtree(frames_dir, ignore_errors=True)
                frames_dir.mkdir()
                for i, jpeg in enumerate(jpegs):
                    (frames_dir / f"{i:06d}.jpg").write_bytes(jpeg)
                n_frames, width, height = transcode(
                    frames_dir / "%06d.jpg", tmp, extra_in=["-framerate", str(fps)]
                )
                shutil.rmtree(frames_dir, ignore_errors=True)
                if looks_blank(tmp, n_frames):
                    blank += 1
                    continue
                meta = {
                    "dataset": "soar",
                    "source": split_name,
                    "episode": ep_idx,
                    "camera": {
                        "index": cam_idx,
                        "name_hint": camera_name_hint(key),
                        "orig_name": key.split("/")[-1],
                    },
                    "language": language,
                    "fps": fps,
                    "frames": n_frames,
                    "width": width,
                    "height": height,
                    "parquet_rows": num_steps,
                    "success": success,
                }
                writer.add_entry(
                    f"ep{ep_idx:06d}_cam{cam_idx:02d}",
                    {"meta.json": json.dumps(meta).encode(), "video.mp4": tmp, "frames.npz": npz},
                )
                written += 1
        except Exception as exc:
            issues.append(f"ep{ep_idx}: {type(exc).__name__}: {exc}")
    tmp.unlink(missing_ok=True)
    shards = writer.close()
    manifest = {
        "dataset": "soar",
        "source": split_name,
        "fps": fps,
        "entries": written,
        "blank_skipped": blank,
        "shards": shards,
        "issue_count": len(issues),
        "issues": issues[:MAX_LOGGED_ISSUES],
    }
    (Path(out_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2))
    report.append(manifest)


def so101_main(args, report):
    repos = sorted(json.loads(Path(args.manifest).read_text()))
    todo = [r for i, r in enumerate(repos) if i % args.num_shards == args.shard]
    for repo in todo:
        source = repo.replace("/", "__")
        dst_prefix = f"{args.out}/so101/{source}"
        if gcs_exists(f"{dst_prefix}/_DONE"):
            print(f"skip(done) {repo}", flush=True)
            continue
        raw_dir = Path(args.workdir) / "raw" / source
        out_dir = Path(args.workdir) / "out" / source
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)
        raw_dir.mkdir(parents=True)
        out_dir.mkdir(parents=True)
        try:
            gcs_rsync(f"{args.raw}/so101/{repo}", raw_dir)
            process_so101_repo(source, raw_dir, out_dir, args.annotations, report)
            gcs_rsync(out_dir, dst_prefix)
            gcs_mark_done(dst_prefix)
            print(f"done {repo}", flush=True)
        except Exception as exc:
            print(f"FAILED {repo}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            shutil.rmtree(raw_dir, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)


def bridge_main(args, report):
    cache = Path(args.workdir) / "bridge_cache"
    if not (cache / "meta").exists():
        cache.mkdir(parents=True, exist_ok=True)
        gcs_rsync(f"{args.raw}/bridge_v2/meta", cache / "meta")
        gcs_rsync(f"{args.raw}/bridge_v2/data", cache / "data")
    episode_table = bridge_episode_table(cache)
    units = bridge_units(episode_table)
    todo = [u for i, u in enumerate(units) if i % args.num_shards == args.shard]
    print(f"bridge: {len(units)} units, {len(todo)} on this shard", flush=True)
    data_cache = {}
    for position, (cam, vchunk, vfile) in enumerate(todo, 1):
        short = cam.removeprefix("observation.images.")
        source = f"{short}_c{vchunk:03d}_f{vfile:03d}"
        dst_prefix = f"{args.out}/bridge/{source}"
        if gcs_exists(f"{dst_prefix}/_DONE"):
            print(f"[{position}/{len(todo)}] skip {source}", flush=True)
            continue
        work = Path(args.workdir) / "bridge_unit"
        shutil.rmtree(work, ignore_errors=True)
        video_dir = work / "videos" / cam / f"chunk-{vchunk:03d}"
        video_dir.mkdir(parents=True)
        try:
            for name in ("meta", "data"):
                (work / name).symlink_to(cache / name)
            run([
                "gcloud", "storage", "cp",
                f"{args.raw}/bridge_v2/videos/{cam}/chunk-{vchunk:03d}/file-{vfile:03d}.mp4",
                str(video_dir / f"file-{vfile:03d}.mp4"),
            ])
            out_dir = work / "out"
            out_dir.mkdir(parents=True)
            process_bridge_video(
                cam, vchunk, vfile, work, out_dir, episode_table, data_cache, report
            )
            gcs_rsync(out_dir, dst_prefix)
            gcs_mark_done(dst_prefix)
            print(f"[{position}/{len(todo)}] done {source}", flush=True)
        except Exception as exc:
            print(f"[{position}/{len(todo)}] FAILED {source}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)


def soar_main(args, report):
    listing = subprocess.run(
        ["gcloud", "storage", "ls", f"{args.raw}/soar/1.0.0/"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    groups = {}
    for uri in listing:
        name = Path(uri).name
        if ".tfrecord-" not in name:
            continue
        split = name.split(".tfrecord-")[0].replace("soar_dataset-", "")
        groups.setdefault(split, []).append(uri)
    items = []
    for split, uris in sorted(groups.items()):
        for i in range(0, len(uris), 8):
            items.append((f"{split}_{i // 8:04d}", sorted(uris)[i : i + 8]))
    todo = [it for i, it in enumerate(items) if i % args.num_shards == args.shard]
    for source, uris in todo:
        dst_prefix = f"{args.out}/soar/{source}"
        if gcs_exists(f"{dst_prefix}/_DONE"):
            print(f"skip(done) {source}", flush=True)
            continue
        work = Path(args.workdir) / "soar_work" / source
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        try:
            local = []
            for uri in uris:
                dst = work / Path(uri).name
                run(["gcloud", "storage", "cp", uri, str(dst)])
                local.append(dst)
            out_dir = work / "out"
            out_dir.mkdir()
            process_soar_split(local, source, out_dir, args.soar_fps, report)
            gcs_rsync(out_dir, dst_prefix)
            gcs_mark_done(dst_prefix)
            print(f"done {source}", flush=True)
        except Exception as exc:
            print(f"FAILED {source}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["so101", "bridge", "soar"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--raw", default="gs://visionary-uc1/raw")
    parser.add_argument("--out", default="gs://visionary-uc1/so101/data/processed/v1")
    parser.add_argument("--workdir", default="/mnt/work")
    parser.add_argument("--manifest", default="artifacts/so101/download_pool.json")
    parser.add_argument("--annotations", default="/mnt/work/language_annotations")
    parser.add_argument("--soar_fps", type=int, default=5)
    args = parser.parse_args()

    report = []
    if args.dataset == "so101":
        so101_main(args, report)
    elif args.dataset == "bridge":
        bridge_main(args, report)
    else:
        soar_main(args, report)

    report_path = Path(args.workdir) / f"report_{args.dataset}_{args.shard}.json"
    report_path.write_text(json.dumps(report, indent=2))
    total_issues = sum(len(m.get("issues", [])) for m in report)
    print(
        f"ARCHIVE_COMPLETE dataset={args.dataset} shard={args.shard} "
        f"sources={len(report)} issues={total_issues}",
        flush=True,
    )


if __name__ == "__main__":
    main()
