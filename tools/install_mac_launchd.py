from __future__ import annotations

import argparse
import getpass
import plistlib
import subprocess
from pathlib import Path


def build_plist(
    *,
    label: str,
    repo: Path,
    python_executable: str,
    config: str,
    env_file: str,
    log_dir: Path,
) -> dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    return {
        "Label": label,
        "ProgramArguments": [
            python_executable,
            str(repo / "run_upstox_paper.py"),
            "--config",
            config,
            "--env-file",
            env_file,
        ],
        "WorkingDirectory": str(repo),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProcessType": "Background",
        "StandardOutPath": str(log_dir / "paper.log"),
        "StandardErrorPath": str(log_dir / "paper.error.log"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Install the NIFTY paper engine as a macOS LaunchAgent")
    ap.add_argument("--repo", required=True, help="Absolute path to the research repo")
    ap.add_argument("--python", default="python3", help="Python executable")
    ap.add_argument("--config", default="config_dte6_frozen_75_25.yaml")
    ap.add_argument("--env-file", default=".env.paper")
    ap.add_argument("--label", default="com.sai.nifty-strangle-paper")
    args = ap.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    log_dir = repo / "results" / "paper" / "logs"
    plist = build_plist(
        label=args.label,
        repo=repo,
        python_executable=args.python,
        config=args.config,
        env_file=args.env_file,
        log_dir=log_dir,
    )

    launch_agents = Path.home() / "Library" / "LaunchAgents"
    launch_agents.mkdir(parents=True, exist_ok=True)
    plist_path = launch_agents / f"{args.label}.plist"
    with plist_path.open("wb") as fh:
        plistlib.dump(plist, fh, sort_keys=False)

    uid = getpass.getuser()
    domain = f"gui/{subprocess.check_output(['id', '-u'], text=True).strip()}"
    subprocess.run(
        ["launchctl", "bootout", domain, str(plist_path)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(["launchctl", "bootstrap", domain, str(plist_path)], check=True)
    print(f"Installed: {plist_path}")
    print(f"Logs: {log_dir}")


if __name__ == "__main__":
    main()
