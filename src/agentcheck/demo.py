"""Write offline demo logs (mockllm, no API key, no cost) for trying `agentcheck compare`.

    python -m agentcheck.demo OUT_DIR --name baseline
    python -m agentcheck.demo OUT_DIR --name candidate --wrong 12

The scripted model answers every v3 question right except the first `--wrong` sample ids
(sorted). The log is written to OUT_DIR/NAME.eval. Used by the GitHub Action self-test.
"""

import argparse
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, ModelUsage, get_model

from agentcheck.tasks.inventory import build_dataset, inventory_lookup

MOCK = "mockllm/model"


def write_log(path: Path, wrong: int, epochs: int = 1) -> Path:
    samples = build_dataset()
    targets = {str(s.input): str(s.target) for s in samples}
    ids = {str(s.input): str(s.id) for s in samples}
    broken = set(sorted(ids.values())[:wrong])

    def model(input: list[object], *args: object) -> ModelOutput:
        question = next(q for q in targets if q in str(input))
        answer = "WRONG" if ids[question] in broken else targets[question]
        out = ModelOutput.from_content(MOCK, f"ANSWER: {answer}")
        out.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        return out

    with tempfile.TemporaryDirectory() as tmp:
        [log] = inspect_eval(
            inventory_lookup(),
            model=get_model(MOCK, custom_outputs=model),
            epochs=epochs,
            log_dir=tmp,
            display="none",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(log.location, path)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m agentcheck.demo", description=__doc__)
    p.add_argument("out_dir")
    p.add_argument("--name", required=True, help="file name without .eval")
    p.add_argument("--wrong", type=int, default=0, help="samples answered wrong (default 0)")
    p.add_argument("--epochs", type=int, default=1)
    args = p.parse_args(argv)
    if not 0 <= args.wrong <= len(build_dataset()):
        print("error: --wrong out of range", file=sys.stderr)
        return 2
    print(write_log(Path(args.out_dir) / f"{args.name}.eval", args.wrong, args.epochs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
