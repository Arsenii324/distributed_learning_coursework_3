"""Export embedded outputs from an .ipynb into a folder.

This repo's notebooks mostly store plots as embedded PNG outputs inside the notebook.
For reproducible artifacts (outside the notebook JSON), this script extracts:
- image/* outputs as individual files (png/jpeg/svg)
- text outputs into a single outputs.txt
- a manifest.json describing what was exported

Usage:
  python -m research.code.distopt.tools.export_ipynb_artifacts \
    --ipynb research/code/distopt/default_experiments.ipynb \
    --outdir research/artifacts/2026-04-01/notebooks/default_experiments

Note: This does *not* execute the notebook; it's purely an exporter.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import nbformat


_IMAGE_MIME_TO_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/svg+xml": "svg",
    "image/webp": "webp",
}


def _ensure_text(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, str):
        return s
    if isinstance(s, list):
        return "".join(str(x) for x in s)
    return str(s)


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def export_ipynb_outputs(ipynb_path: Path, outdir: Path) -> dict[str, Any]:
    nb = nbformat.read(str(ipynb_path), as_version=4)

    outdir.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, Any]] = []
    text_chunks: list[str] = []

    for cell_index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        outputs = cell.get("outputs", []) or []
        if not outputs:
            continue

        for output_index, out in enumerate(outputs):
            output_type = out.get("output_type")

            if output_type == "stream":
                name = out.get("name", "stream")
                text = _ensure_text(out.get("text"))
                if text:
                    text_chunks.append(f"\n--- cell {cell_index} stream({name}) out {output_index} ---\n")
                    text_chunks.append(text)
                continue

            if output_type in {"execute_result", "display_data"}:
                data = out.get("data", {}) or {}
                # Prefer image exports.
                for mime, ext in _IMAGE_MIME_TO_EXT.items():
                    if mime not in data:
                        continue

                    blob = data[mime]
                    if ext == "svg":
                        svg_text = _ensure_text(blob)
                        filename = f"cell{cell_index:03d}_out{output_index:02d}.{ext}"
                        filepath = outdir / filename
                        filepath.write_text(svg_text, encoding="utf-8")
                    else:
                        # In nbformat this is base64-encoded string (or list of strings).
                        b64 = _ensure_text(blob).strip()
                        raw = base64.b64decode(b64)
                        filename = f"cell{cell_index:03d}_out{output_index:02d}.{ext}"
                        filepath = outdir / filename
                        _write_bytes(filepath, raw)

                    exported.append(
                        {
                            "cell_index": cell_index,
                            "output_index": output_index,
                            "mime": mime,
                            "file": str((outdir / filename).resolve()),
                        }
                    )

                # Also capture plain text results, if present.
                if "text/plain" in data:
                    text = _ensure_text(data.get("text/plain"))
                    if text:
                        text_chunks.append(
                            f"\n--- cell {cell_index} text/plain out {output_index} ---\n"
                        )
                        text_chunks.append(text)
                continue

            if output_type == "error":
                traceback = out.get("traceback") or []
                tb_text = "\n".join(traceback)
                text_chunks.append(f"\n--- cell {cell_index} ERROR out {output_index} ---\n")
                text_chunks.append(tb_text)
                continue

            # Fallback: serialize unknown output.
            text_chunks.append(
                f"\n--- cell {cell_index} {output_type} out {output_index} (raw) ---\n"
            )
            text_chunks.append(json.dumps(out, ensure_ascii=False, indent=2))

    outputs_txt = outdir / "outputs.txt"
    if text_chunks:
        outputs_txt.write_text("".join(text_chunks), encoding="utf-8")

    manifest = {
        "ipynb": str(ipynb_path.resolve()),
        "outdir": str(outdir.resolve()),
        "exported_images": exported,
        "outputs_txt": str(outputs_txt.resolve()) if outputs_txt.exists() else None,
        "counts": {
            "images": len(exported),
            "text_chunks": len(text_chunks),
        },
    }
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipynb", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    ipynb_path = Path(args.ipynb)
    outdir = Path(args.outdir)

    if not ipynb_path.exists():
        raise SystemExit(f"Not found: {ipynb_path}")

    export_ipynb_outputs(ipynb_path=ipynb_path, outdir=outdir)


if __name__ == "__main__":
    main()
