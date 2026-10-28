"""Read an Inspect eval log and summarise it per category and per sample.

This is descriptive only: counts and rates, no inference. Its job is to make the
run-to-run picture visible, in particular which samples flip between epochs.
"""

import argparse
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.scorer import CORRECT

SCORER = "answer_line"


@dataclass
class SampleRecord:
    sample_id: str
    category: str
    passes: list[bool] = field(default_factory=list)  # one entry per epoch
    output_errors: int = 0

    @property
    def flaky(self) -> bool:
        return 0 < sum(self.passes) < len(self.passes)


def per_sample(log: EvalLog) -> dict[str, SampleRecord]:
    """Group epochs by sample id. Fails loudly on anything unscored."""
    if log.status != "success":
        raise ValueError(f"log status is {log.status!r}, not 'success'")
    records: dict[str, SampleRecord] = {}
    for s in log.samples or []:
        if s.scores is None or SCORER not in s.scores:
            raise ValueError(f"sample {s.id} epoch {s.epoch} has no {SCORER} score")
        metadata = s.metadata or {}
        key = str(s.id)
        rec = records.setdefault(key, SampleRecord(key, str(metadata.get("category", "?"))))
        rec.passes.append(s.scores[SCORER].value == CORRECT)
        rec.output_errors += "model_output_error" in metadata
    return records


def render(log: EvalLog, records: dict[str, SampleRecord]) -> str:
    epochs = log.eval.config.epochs or 1
    lines = [
        f"Log: {log.location}",
        f"Model: {log.eval.model}   samples: {len(records)}   epochs: {epochs}",
        "",
        f"{'category':<10} {'passed':>9} {'rate':>6} {'flaky':>6} {'out-err':>8}",
    ]
    by_cat: dict[str, list[SampleRecord]] = defaultdict(list)
    for rec in records.values():
        by_cat[rec.category].append(rec)
    for cat in sorted(by_cat):
        recs = by_cat[cat]
        passed = sum(sum(r.passes) for r in recs)
        total = sum(len(r.passes) for r in recs)
        flaky = sum(r.flaky for r in recs)
        errors = sum(r.output_errors for r in recs)
        lines.append(
            f"{cat:<10} {passed:>4}/{total:<4} {passed / total:>6.3f} {flaky:>6} {errors:>8}"
        )
    passed = sum(sum(r.passes) for r in records.values())
    total = sum(len(r.passes) for r in records.values())
    lines.append(f"{'ALL':<10} {passed:>4}/{total:<4} {passed / total:>6.3f}")

    def listing(title: str, recs: list[SampleRecord]) -> None:
        lines.append("")
        lines.append(f"{title}: {len(recs)}")
        lines.extend(f"  {r.sample_id:<22} {sum(r.passes)}/{len(r.passes)}" for r in recs)

    listing(
        "Flaky samples (pass in some epochs, not all)", [r for r in records.values() if r.flaky]
    )
    listing("Always failing", [r for r in records.values() if not any(r.passes)])
    return "\n".join(lines)


def latest_log(log_dir: Path) -> Path:
    logs = sorted(log_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise FileNotFoundError(f"no .eval files in {log_dir}")
    return logs[-1]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", nargs="?", help="path to a .eval file (default: newest in --log-dir)")
    p.add_argument("--log-dir", default="logs")
    args = p.parse_args(argv)
    path = Path(args.log) if args.log else latest_log(Path(args.log_dir))
    log = read_eval_log(path)
    print(render(log, per_sample(log)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
