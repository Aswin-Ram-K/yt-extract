#!/usr/bin/env python3
"""MCP server for yt-extract — exposes video extraction as MCP tools."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import time
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yt-extract-mcp")

YT_EXTRACT_CMD = shutil.which("yt-extract") or "yt-extract"


def _run_extract(
    url: str,
    output_dir: str | None = None,
    frame_budget: int | None = None,
    ocr: bool = False,
    ocr_lang: str = "eng",
    compact: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run yt-extract CLI and return structured result."""
    args = [YT_EXTRACT_CMD, url, "--json-output"]
    if output_dir:
        args.extend(["--output-dir", output_dir])
    if frame_budget:
        args.extend(["--frame-budget", str(frame_budget)])
    if ocr:
        args.append("--ocr")
    if ocr_lang:
        args.extend(["--ocr-lang", ocr_lang])
    if compact:
        args.append("--compact")
    if dry_run:
        args.append("--dry-run")

    logger.info("Running: %s", " ".join(args))
    start = time.time()
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=600,
        )
        elapsed = round(time.time() - start, 2)

        # Try to parse JSON output
        manifest = {}
        report = ""
        if result.stdout.strip():
            try:
                manifest = json.loads(result.stdout.strip())
            except json.JSONDecodeError:
                manifest = {"raw_output": result.stdout.strip()}
                report = result.stdout.strip()

        report = report or result.stderr.strip().split("\n")[-1] if result.stderr.strip() else ""

        return {
            "status": "ok" if result.returncode == 0 else "error",
            "exit_code": result.returncode,
            "manifest": manifest,
            "report": report,
            "elapsed_seconds": elapsed,
            "stderr": result.stderr.strip() if result.stderr else "",
            "output_dir": output_dir,
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error": "Timed out after 600s",
            "elapsed_seconds": round(time.time() - start, 2),
            "output_dir": output_dir,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "elapsed_seconds": round(time.time() - start, 2),
            "output_dir": output_dir,
        }


# ── MCP Server ─────────────────────────────────────────────────────────────

app = Server("yt-extract")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="yt-extract-extract",
            description=(
                "Extract frames, audio features, and metadata from a YouTube video. "
                "Downloads video+audio, detects scene cuts, extracts frames, computes "
                "librosa audio features, and writes a structured manifest.json. "
                "Use this for deep video analysis by agents."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "YouTube video URL to extract from",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory (default: yt-extract-<video_id>/)",
                    },
                    "frame_budget": {
                        "type": "integer",
                        "description": "Max frames to extract (default: 200)",
                    },
                    "ocr": {
                        "type": "boolean",
                        "description": "Enable OCR on scene-cut frames",
                    },
                    "ocr_lang": {
                        "type": "string",
                        "description": "OCR language (default: eng)",
                    },
                    "compact": {
                        "type": "boolean",
                        "description": "Omit full feature arrays from manifest",
                    },
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="yt-extract-dry-run",
            description=(
                "Get video metadata without downloading. Returns title, duration, "
                "chapters, resolution, view count, and thumbnails. Use this to check "
                "video availability before full extraction."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "YouTube video URL",
                    },
                },
                "required": ["url"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    if name == "yt-extract-extract":
        result = _run_extract(
            url=arguments.get("url", ""),
            output_dir=arguments.get("output_dir"),
            frame_budget=arguments.get("frame_budget", 200),
            ocr=arguments.get("ocr", False),
            ocr_lang=arguments.get("ocr_lang", "eng"),
            compact=arguments.get("compact", False),
        )
    elif name == "yt-extract-dry-run":
        result = _run_extract(
            url=arguments.get("url", ""),
            dry_run=True,
        )
    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


async def main() -> None:
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
