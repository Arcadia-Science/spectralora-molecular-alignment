import argparse
import shutil
import subprocess
import tarfile
import urllib.request
from io import BytesIO
from pathlib import Path


def download_sdist(version: str) -> bytes:
    url = f"https://files.pythonhosted.org/packages/source/d/dm-tree/dm-tree-{version}.tar.gz"
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def patch_setup(setup_path: Path) -> None:
    text = setup_path.read_text()
    if "-DCMAKE_POLICY_VERSION_MINIMUM=3.5" in text:
        return

    lines = text.splitlines()
    out_lines = []
    inserted = False
    for line in lines:
        out_lines.append(line)
        if not inserted and "cmake_args = [" in line:
            indent = line.split("cmake_args")[0]
            out_lines.append(f"{indent}    '-DCMAKE_POLICY_VERSION_MINIMUM=3.5',")
            inserted = True

    if not inserted:
        raise RuntimeError("Failed to find cmake_args list in setup.py")

    setup_path.write_text("\n".join(out_lines) + "\n")


def install_package(path: Path) -> None:
    subprocess.check_call(["python", "-m", "pip", "install", str(path)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch dm-tree for CMake >= 3.5 policy requirements.")
    parser.add_argument("--version", default="0.1.8", help="dm-tree version to patch.")
    parser.add_argument("--dest", type=Path, default=None, help="Destination directory for extracted source.")
    parser.add_argument("--force", action="store_true", help="Overwrite destination if it exists.")
    parser.add_argument("--install", action="store_true", help="Install patched package after patching.")
    args = parser.parse_args()

    dest = args.dest or Path("/tmp") / f"dm-tree-{args.version}"
    if dest.exists():
        if args.force:
            shutil.rmtree(dest)
        else:
            raise SystemExit(f"Destination already exists: {dest}")

    data = download_sdist(args.version)
    with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as tf:
        tf.extractall(path=dest.parent)

    setup_path = dest / "setup.py"
    if not setup_path.exists():
        raise SystemExit(f"setup.py not found at {setup_path}")

    patch_setup(setup_path)
    print(f"Patched {setup_path}")

    if args.install:
        install_package(dest)


if __name__ == "__main__":
    main()
