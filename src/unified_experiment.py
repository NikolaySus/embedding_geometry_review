"""CLI for the independent unified protocol; legacy experiment.py is unchanged."""

import argparse
import json

from embedding_geometry.common import load_config
from embedding_geometry.unified import plan, run, evaluate_saved
from embedding_geometry.unified_report import report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unified_v1.json")
    parser.add_argument("command", choices=["plan", "smoke", "run", "evaluate", "report"])
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--regimes", nargs="+")
    parser.add_argument("--smoke-results", action="store_true", help="Report technical smoke observations only")
    args = parser.parse_args()
    if args.command in {"plan", "report"} and any((args.models, args.seeds, args.regimes)):
        parser.error("Model/seed/regime filters apply only to smoke, run and evaluate")
    if args.smoke_results and args.command != "report":
        parser.error("--smoke-results applies only to report")
    config = load_config(args.config)
    if args.command == "plan":
        result = plan(config)
    elif args.command == "report":
        result = report(config, smoke=args.smoke_results)
    else:
        options = {k: getattr(args, k) for k in ("models", "seeds", "regimes")}
        result = evaluate_saved(config, **options) if args.command == "evaluate" else run(config, smoke=args.command == "smoke", **options)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
