import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from datetime import datetime, timezone


DEFAULT_SCENE = Path("/usr/share/rcssserver3d/rsg/agent/nao/soccer.rsg")
SLIP_PATTERN = re.compile(
    r"(?m)^(?P<prefix>\s*\(setContactSlip\s+)"
    r"(?P<first>[0-9.eE+-]+)(?:(?P<middle>\s+)"
    r"(?P<second>[0-9.eE+-]+))?(?P<suffix>\)\s*)$"
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_slip(path):
    text = path.read_text(encoding="utf-8")
    matches = list(SLIP_PATTERN.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one setContactSlip expression in {path}; found {len(matches)}"
        )
    first = float(matches[0].group("first"))
    second = float(matches[0].group("second") or matches[0].group("first"))
    if not math.isfinite(first) or not math.isfinite(second) or first < 0 or second < 0:
        raise RuntimeError('Slip must be finite and non-negative')
    if abs(first - second) > 1e-12:
        raise RuntimeError(f"Anisotropic field slip is not supported: {first}, {second}")
    return text, matches[0], first


def status(path, expected=None):
    _, match, value = read_slip(path)
    result = {
        "scene_path": str(path.resolve()),
        "field_contact_slip": value,
        "scene_setter_method": "single_argument_v1" if match.group('second') is None else "legacy_two_argument",
        "sha256": digest(path),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if expected is not None:
        result["expected_field_contact_slip"] = expected
        # apply() serializes at 12 significant digits; verify that exact encoding
        # as well as the original tight numeric comparison. This is not a broad
        # relative tolerance and does not change scene bytes or solver inputs.
        result["matches_expected"] = (abs(value - expected) <= 1e-12 or
                                      value == float(f'{expected:.12g}'))
    return result


def atomic_write(path, text):
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        shutil.copymode(path, temporary)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def apply(path, value, backup_path, setter_method='scalar'):
    if not math.isfinite(value) or value < 0:
        raise ValueError("Field contact slip must be finite and non-negative")
    if setter_method not in ('scalar','legacy'):
        raise ValueError('Unknown slip setter method')
    text, match, current = read_slip(path)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
    else:
        read_slip(backup_path)
    values = f'{value:.12g}' if setter_method=='scalar' else f'{value:.12g} {value:.12g}'
    replacement = f'{match.group("prefix")}{values}{match.group("suffix")}'
    updated = text[:match.start()] + replacement + text[match.end():]
    if updated != text:
        atomic_write(path, updated)
    result = status(path, value)
    result.update({
        "action": "apply",
        "previous_field_contact_slip": current,
        "backup_path": str(backup_path.resolve()),
        "restart_required": True,
    })
    return result


def restore(path, backup_path):
    if not backup_path.is_file():
        raise FileNotFoundError(f"Nominal backup not found: {backup_path}")
    _, _, backup_value = read_slip(backup_path)
    shutil.copy2(backup_path, path)
    result = status(path, backup_value)
    result.update({
        "action": "restore",
        "backup_path": str(backup_path.resolve()),
        "restart_required": True,
    })
    return result


def emit(result, output):
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Back up, configure, verify, and restore SimSpark field contact slip"
    )
    parser.add_argument("action", choices=("status", "apply", "restore"))
    parser.add_argument("value", type=float, nargs="?")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--backup", type=Path, default=None)
    parser.add_argument("--expect", type=float, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    scene = args.scene.resolve()
    backup = args.backup or scene.with_name(scene.name + ".walker_phase2_nominal.bak")
    if args.action == "apply":
        if args.value is None:
            parser.error("apply requires a slip value")
        result = apply(scene, args.value, backup)
    elif args.action == "restore":
        if args.value is not None:
            parser.error("restore does not accept a value")
        result = restore(scene, backup)
    else:
        if args.value is not None:
            parser.error("status uses --expect rather than a positional value")
        result = status(scene, args.expect)
        if args.expect is not None and not result["matches_expected"]:
            emit(result, args.output)
            raise SystemExit(2)
    emit(result, args.output)
