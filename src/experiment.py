from __future__ import annotations

import argparse
import json

from embedding_geometry.common import check_disk, load_config
from embedding_geometry.data import audit_evaluation_overlap, audit_processed, prepare_all
from embedding_geometry.orchestration import evaluate_phase, run_all, train_phase
from embedding_geometry.evaluation import augment_clean_sts_summaries
from embedding_geometry.reporting import build_report
from embedding_geometry.mixtures import build_mixture_report, run_mixtures


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Controlled English text-embedding geometry experiment")
    result.add_argument("--config", default="configs/pilot_v1.json")
    subparsers = result.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-data")
    subparsers.add_parser("audit")
    subparsers.add_parser("audit-evaluation")
    run = subparsers.add_parser("run")
    run.add_argument("--phase", choices=("smoke", "seed1", "seeds3", "all"), default="all")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--suite", choices=("gate", "full"), required=True)
    evaluate.add_argument("--seeds", nargs="*", type=int)
    subparsers.add_parser("evaluate-clean-sts")
    subparsers.add_parser("report")
    subparsers.add_parser("check")
    mixtures = subparsers.add_parser("mixtures")
    mixtures.add_argument("--phase", choices=("smoke", "full", "report"), default="full")
    return result


def main() -> None:
    args = parser().parse_args()
    config = load_config(args.config)
    if args.command == "prepare-data":
        check_disk(config["storage"]["minimum_free_gb"])
        print(json.dumps(prepare_all(config, config["seeds"][0]), ensure_ascii=False, indent=2))
    elif args.command == "audit":
        report = audit_processed(config)
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "exact_cross_source_overlaps": report["exact_cross_source_overlaps"],
                    "near_duplicate_candidates": len(report["near_duplicate_candidates"]),
                    "report": "reports/pilot_v1_data_audit.json",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "audit-evaluation":
        report = audit_evaluation_overlap(config)
        print(json.dumps({"decision": report["decision"], "tasks_checked": report["tasks_checked"], "tasks_with_overlap": report["tasks_with_overlap"], "report": "reports/pilot_v1_evaluation_overlap_audit.json"}, indent=2))
    elif args.command == "run":
        if args.phase == "all":
            run_all(config)
        else:
            train_phase(config, args.phase)
    elif args.command == "evaluate":
        evaluate_phase(config, args.suite, args.seeds)
    elif args.command == "evaluate-clean-sts":
        print(json.dumps({"summaries_augmented": augment_clean_sts_summaries(config)}, indent=2))
    elif args.command == "report":
        print(build_report(config))
    elif args.command == "mixtures":
        if args.phase == "report":
            print(build_mixture_report(config))
        else:
            run_mixtures(config, smoke=args.phase == "smoke")
    else:
        check_disk(config["storage"]["minimum_free_gb"])
        print("Configuration and disk-space checks passed")


if __name__ == "__main__":
    main()
