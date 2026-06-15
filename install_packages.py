from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata


REQUIRED_PACKAGES = [
    "numpy",
    "opencv-python",
    "pillow",
    "torch>=1.7",
    "torchvision",
    "torchsummary",
    "albumentations",
    "matplotlib",
    "seaborn",
    "scikit-image",
    "scikit-learn",
    "pandas",
    "tqdm",
    "xlsxwriter",
    "pyinstaller",
    "PySide6",
]

# Some install names do not match the metadata package name exactly.
DIST_NAME_OVERRIDES = {
    "opencv-python": "opencv-python",
    "pillow": "Pillow",
    "scikit-image": "scikit-image",
    "scikit-learn": "scikit-learn",
    "PySide6": "PySide6",
}


@dataclass
class InstallResult:
    package: str
    status: str
    detail: str = ""


def base_requirement_name(requirement: str) -> str:
    for separator in ("<=", ">=", "==", "~=", "!=", "<", ">", "["):
        if separator in requirement:
            return requirement.split(separator, 1)[0].strip()
    return requirement.strip()


def installed_distribution_name(requirement: str) -> str:
    base_name = base_requirement_name(requirement)
    return DIST_NAME_OVERRIDES.get(base_name, base_name)


def is_installed(requirement: str) -> bool:
    try:
        metadata.version(installed_distribution_name(requirement))
        return True
    except metadata.PackageNotFoundError:
        return False


def run_pip(command: list[str], *, dry_run: bool) -> None:
    rendered = " ".join(command)
    print(f"$ {rendered}")
    if dry_run:
        return
    subprocess.check_call(command)


def install_one(requirement: str, *, upgrade: bool, dry_run: bool) -> InstallResult:
    if is_installed(requirement) and not upgrade:
        return InstallResult(requirement, "already-installed")

    command = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        command.append("--upgrade")
    command.append(requirement)

    try:
        run_pip(command, dry_run=dry_run)
        return InstallResult(requirement, "installed")
    except subprocess.CalledProcessError as exc:
        return InstallResult(requirement, "failed", f"pip exited with status {exc.returncode}")


def maybe_upgrade_pip(*, dry_run: bool) -> InstallResult:
    command = [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
    try:
        run_pip(command, dry_run=dry_run)
        return InstallResult("pip", "installed")
    except subprocess.CalledProcessError as exc:
        return InstallResult("pip", "failed", f"pip exited with status {exc.returncode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install Python dependencies for segment_app.")
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="force reinstall/upgrade packages even if they are already present",
    )
    parser.add_argument(
        "--upgrade-pip",
        action="store_true",
        help="upgrade pip before installing the project dependencies",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print pip commands without executing them",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Python executable: {sys.executable}")
    print(f"Package count: {len(REQUIRED_PACKAGES)}")

    results: list[InstallResult] = []

    if args.upgrade_pip:
        print("\nUpgrading pip")
        results.append(maybe_upgrade_pip(dry_run=args.dry_run))

    print("\nInstalling dependencies")
    for requirement in REQUIRED_PACKAGES:
        result = install_one(requirement, upgrade=args.upgrade, dry_run=args.dry_run)
        results.append(result)
        detail_suffix = f" ({result.detail})" if result.detail else ""
        print(f"- {requirement}: {result.status}{detail_suffix}")

    failures = [result for result in results if result.status == "failed"]
    installed = [result for result in results if result.status == "installed"]
    skipped = [result for result in results if result.status == "already-installed"]

    print("\nSummary")
    print(f"- Installed: {len(installed)}")
    print(f"- Already installed: {len(skipped)}")
    print(f"- Failed: {len(failures)}")

    if failures:
        print("\nFailures")
        for result in failures:
            detail_suffix = f": {result.detail}" if result.detail else ""
            print(f"- {result.package}{detail_suffix}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
