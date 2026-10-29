"""Preflight checks for deterministic headless SimSpark collection."""

from pathlib import Path
import re
import sys


def setting(path, variable):
    text = path.read_text(encoding="utf-8")
    match = re.search(
        rf"^\s*\${re.escape(variable)}\s*=\s*(true|false)\s*$",
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"Could not find ${variable} in {path}")
    return match.group(1) == "true"


def main():
    spark_config = Path.home() / ".simspark" / "spark.rb"
    server_roots = (Path("/usr/local/share/rcssserver3d"), Path("/usr/share/rcssserver3d"))
    server_config = next(
        (root / "rcssserver3d.rb" for root in server_roots if (root / "rcssserver3d.rb").is_file()),
        None,
    )

    if not spark_config.is_file():
        raise FileNotFoundError(f"SimSpark configuration not found: {spark_config}")
    sync_enabled = setting(spark_config, "agentSyncMode")
    valid = True
    if not sync_enabled:
        print(
            f"[Error] SimSpark synchronous mode is disabled in {spark_config}.\n"
            "Set '$agentSyncMode = true'.",
            file=sys.stderr,
        )
        valid = False

    if server_config is None:
        raise FileNotFoundError("Could not locate rcssserver3d.rb")
    if setting(server_config, "enableRealTimeMode"):
        print(
            f"[Error] Real-time mode is enabled in {server_config}.\n"
            "Set '$enableRealTimeMode = false'.",
            file=sys.stderr,
        )
        valid = False

    if not valid:
        print("Restart rcssserver3d after changing either setting.", file=sys.stderr)
        return 1

    print("[Preflight] SimSpark synchronous mode is enabled and real-time mode is disabled.")
    print("[Preflight] Restart rcssserver3d after any configuration change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
