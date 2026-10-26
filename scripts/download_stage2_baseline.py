#!/usr/bin/env python3
"""Resume and verify the pinned official VPP2 Stage-2 checkpoint download."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


PROJECT = Path(__file__).resolve().parents[1]
RUN = PROJECT / "runs/baseline_setup_20261015"
RELATIVE = "checkpoints_video/vpp2-video-stage2-17f.pth"


def save(status):
    path = RUN / "stage2_download_status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download():
    metadata = json.loads((RUN / "hf_model_info.json").read_text())
    entry = next(x for x in metadata["siblings"] if x["rfilename"] == RELATIVE)
    expected = entry["lfs"]["sha256"]
    final = PROJECT / "weights" / RELATIVE
    partial = final.with_suffix(final.suffix + ".part")
    final.parent.mkdir(parents=True, exist_ok=True)
    status = dict(
        repo_id="Haodong082399/VPP2", revision=metadata["sha"],
        relative_path=RELATIVE, path=str(final), partial_path=str(partial),
        expected_size=entry["size"], expected_sha256=expected,
        worker_pid=os.getpid(), started_unix=time.time(), status="starting",
    )
    save(status)
    try:
        if final.exists():
            if final.stat().st_size != entry["size"] or digest(final) != expected:
                raise ValueError("Existing final checkpoint failed verification; leaving it unchanged")
        else:
            url = f"https://huggingface.co/Haodong082399/VPP2/resolve/{metadata['sha']}/{RELATIVE}"
            command = [
                "aria2c", "--continue=true", "--auto-file-renaming=false",
                "--allow-overwrite=false", "--file-allocation=none",
                "--max-connection-per-server=8", "--split=8", "--min-split-size=64M",
                "--max-tries=12", "--retry-wait=5", "--connect-timeout=30", "--timeout=60",
                "--summary-interval=30", "--console-log-level=warn", "--enable-color=false",
                "--show-console-readout=false", "--download-result=full",
                "--dir=" + str(final.parent), "--out=" + partial.name, url,
            ]
            env = os.environ.copy()
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
                if env.get(key):
                    env[key.lower()] = env[key]
            with (RUN / "stage2_download.log").open("ab", buffering=0) as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, env=env)
                status.update(status="downloading", download_pid=child.pid)
                save(status)
                code = child.wait()
            if code != 0:
                raise RuntimeError(f"aria2c exited with code {code}; see stage2_download.log")
            status["status"] = "verifying_sha256"
            save(status)
            if partial.stat().st_size != entry["size"] or digest(partial) != expected:
                raise ValueError("Downloaded checkpoint failed official size/SHA256 verification")
            partial.replace(final)
        status.update(status="complete", sha256_verified=True, completed_unix=time.time())
        save(status)
    except Exception as error:
        status.update(status="failed", error=str(error), failed_unix=time.time())
        save(status)
        raise


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "stage2_download.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        download()


if __name__ == "__main__":
    main()
