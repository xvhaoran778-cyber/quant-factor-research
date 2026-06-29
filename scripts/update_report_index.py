#!/usr/bin/env python3
"""更新 reports/INDEX.md — 扫描目录自动生成策略报告列表"""

from __future__ import annotations

import os
from datetime import date

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


def main():
    lines = ["# 策略报告索引 — 自动生成\n", f"\n_更新于 {date.today()}_\n"]
    for f in sorted(os.listdir(REPORTS_DIR)):
        if f.endswith(".md") and f != "INDEX.md":
            lines.append(f"- [{f}]({f})")
        elif f.endswith(".csv"):
            lines.append(f"- [{f}]({f})")
    lines.append("\n")
    path = os.path.join(REPORTS_DIR, "INDEX.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"Updated {path} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
