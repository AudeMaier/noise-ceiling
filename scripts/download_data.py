"""Download the data archives from OSF and unpack them into data/.

Every archive is checked against data_checksums.txt before it is unpacked.
Archives already present and matching their checksum are skipped, so the script
is safe to re-run after an interrupted download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

OSF_NODE = "2btpr"
API_URL = "https://api.osf.io/v2/nodes/{node}/files/osfstorage/"
CHECKSUM_FILE = "data_checksums.txt"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
DOWNLOAD_DIR = os.path.join(DATA_DIR, "_archives")


def with_token(url, view_only):
    """Add the view-only token to a URL, which OSF needs for a private project."""
    if not view_only:
        return url
    separator = "&" if "?" in url else "?"
    return url + separator + urlencode({"view_only": view_only})


def list_remote_files(node, view_only=None):
    """Every file in the project's OSF Storage, as (name, download url, size)."""
    url = with_token(API_URL.format(node=node), view_only)
    files = []

    while url:
        try:
            with urlopen(url) as response:
                payload = json.load(response)
        except HTTPError as error:
            hint = ("  the project is private: pass --view-only TOKEN, the value "
                    "after view_only= in the share link"
                    if error.code in (401, 403, 404) else "")
            raise SystemExit(f"OSF returned {error.code} for {url}\n{hint}")
        except URLError as error:
            raise SystemExit(f"could not reach OSF: {error.reason}")

        for item in payload["data"]:
            attributes = item["attributes"]
            if attributes.get("kind") != "file":
                continue
            link = item["links"]["download"]
            files.append((attributes["name"], with_token(link, view_only),
                          attributes.get("size")))

        url = payload.get("links", {}).get("next")
        if url:
            url = with_token(url, view_only)

    return files


def download(url, path, size=None):
    """Stream a URL to disk, reporting progress."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    name = os.path.basename(path)
    done = 0

    with urlopen(url) as response, open(path, "wb") as out:
        total = size or int(response.headers.get("Content-Length") or 0)
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total:
                sys.stdout.write(f"\r  {name}  {100 * done / total:5.1f}%  "
                                 f"({done / 1e6:.0f}/{total / 1e6:.0f} MB)")
            else:
                sys.stdout.write(f"\r  {name}  {done / 1e6:.0f} MB")
            sys.stdout.flush()

    sys.stdout.write("\n")


def sha256(path, block_size=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_checksums(path):
    """Parse `sha256sum` output into {filename: digest}."""
    checksums = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, _, name = line.partition(" ")
            checksums[os.path.basename(name.strip().lstrip("*"))] = digest
    return checksums


def safe_extract(archive, destination):
    """Unpack a zip, refusing any member that would escape `destination`."""
    destination = os.path.abspath(destination)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            target = os.path.abspath(os.path.join(destination, member))
            if not target.startswith(destination + os.sep) and target != destination:
                raise SystemExit(f"{archive} contains an unsafe path: {member}")
        zf.extractall(destination)


def main(node, view_only, only, keep_archives):
    print(f"listing files in OSF project {node}")
    remote = list_remote_files(node, view_only)
    if not remote:
        raise SystemExit("no files found in the project's OSF Storage")

    by_name = {name: (url, size) for name, url, size in remote}

    # The checksums are committed to the repository, so verifying against them
    # confirms the archive is the one this code expects, not merely that the
    # download was not corrupted. Fall back to the copy on OSF only if the
    # local one is missing.
    local_checksums = os.path.join(ROOT, CHECKSUM_FILE)
    if os.path.exists(local_checksums):
        checksum_path = local_checksums
    elif CHECKSUM_FILE in by_name:
        print(f"{CHECKSUM_FILE} not found in the repository, using the copy on OSF")
        checksum_path = os.path.join(DOWNLOAD_DIR, CHECKSUM_FILE)
        download(by_name[CHECKSUM_FILE][0], checksum_path)
    else:
        raise SystemExit(f"{CHECKSUM_FILE} is in neither the repository nor the "
                         f"OSF project")
    checksums = read_checksums(checksum_path)

    archives = [name for name in sorted(by_name)
                if name.endswith(".zip") and (not only or name in only)]
    if only:
        missing = set(only) - set(by_name)
        if missing:
            raise SystemExit(f"not in the project: {', '.join(sorted(missing))}")

    for name in archives:
        url, size = by_name[name]
        path = os.path.join(DOWNLOAD_DIR, name)
        expected = checksums.get(name)

        if os.path.exists(path) and expected and sha256(path) == expected:
            print(f"  {name}  already downloaded")
        else:
            download(url, path, size)

        if expected is None:
            print(f"  {name}  WARNING: no checksum listed, not verified")
        elif sha256(path) != expected:
            raise SystemExit(f"  {name}  checksum mismatch, delete it and retry")
        else:
            print(f"  {name}  checksum ok")

        safe_extract(path, DATA_DIR)
        print(f"  {name}  unpacked into {os.path.relpath(DATA_DIR, ROOT)}/")

        if not keep_archives:
            os.remove(path)

    print("\ndone")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--node", default=OSF_NODE, help="OSF project id")
    parser.add_argument("--view-only", default=os.environ.get("OSF_VIEW_ONLY"),
                        help="view-only token, if the project is still private")
    parser.add_argument("--only", nargs="+", metavar="ARCHIVE",
                        help="download only these archives")
    parser.add_argument("--keep-archives", action="store_true",
                        help="keep the zip files after unpacking")
    args = parser.parse_args()

    main(args.node, args.view_only, args.only, args.keep_archives)