"""Tạo báo cáo Markdown ngắn từ evaluation artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def format_number(value: float | None) -> str:
    """Định dạng metric nhất quán; giữ rõ trường hợp chưa xác định."""
    return "N/A" if value is None else f"{value:.3f}"


def build_markdown(evaluation: dict) -> str:
    """Chuyển kết quả machine-readable thành bảng Markdown dễ đọc."""
    rows = ["| Mô hình | MAE backtest | Độ lệch chuẩn | RMSE backtest |", "|---|---:|---:|---:|"]
    for name, report in evaluation["backtest"].items():
        rows.append(
            f"| {name} | {format_number(report['mae_mean'])} | "
            f"{format_number(report['mae_std'])} | {format_number(report['rmse_mean'])} |"
        )

    test = evaluation["champion_test"]
    gate = evaluation["quality_gate"]
    return "\n".join(
        [
            "# Báo cáo đánh giá mô hình",
            "",
            *rows,
            "",
            "## Test cuối",
            "",
            f"- MAE: {format_number(test['mae'])}",
            f"- RMSE: {format_number(test['rmse'])}",
            f"- Macro-F1: {format_number(test['macro_f1'])}",
            f"- Recall lớp Xấu: {format_number(test['high_pm25_recall'])}",
            f"- Quality gate: **{gate['status']}**",
            "",
            "> Chỉ công bố báo cáo này khi evaluation được tạo từ dữ liệu thật có nguồn rõ ràng.",
        ]
    )


def main() -> None:
    """Đọc evaluation JSON và ghi báo cáo Markdown."""
    parser = argparse.ArgumentParser(description="Tạo báo cáo Markdown từ evaluation artifact")
    parser.add_argument("--input", default="artifacts/evaluation.json")
    parser.add_argument("--output", default="reports/evaluation_summary.md")
    args = parser.parse_args()

    evaluation = json.loads(Path(args.input).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(evaluation), encoding="utf-8")
    print(f"Đã tạo báo cáo: {output}")


if __name__ == "__main__":
    main()
