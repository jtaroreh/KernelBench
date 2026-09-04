#!/usr/bin/env python3
"""Periodic heartbeat reminder script for agents and subagents."""

import argparse
import datetime
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Periodically logs or sends encouraging reminders to agents and subagents."
    )
    parser.add_argument(
        "--message",
        type=str,
        default="you got this! keep going",
        help="Reminder message to emit.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=120.0,
        help="Interval in seconds between periodic reminders.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Number of iterations to run (0 for infinite).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to log file where reminders are appended.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run exactly one reminder and exit immediately.",
    )
    parser.add_argument(
        "--target-subagent",
        type=str,
        default=None,
        help="Optional subagent identifier or role to mention in log.",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Include timestamp in output.",
    )
    return parser.parse_args()


def emit_reminder(message: str, target: str | None, log_file: str | None, timestamp: bool = False) -> None:
    if timestamp:
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        target_str = f" [{target}]" if target else ""
        log_line = f"[{now}]{target_str} {message}\n"
    else:
        target_str = f"[{target}] " if target else ""
        log_line = f"{target_str}{message}\n"
    sys.stdout.write(log_line)
    sys.stdout.flush()
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_line)


def main() -> None:
    args = parse_args()
    if args.once:
        emit_reminder(args.message, args.target_subagent, args.log_file, timestamp=args.timestamp)
        return

    count = 0
    try:
        while True:
            emit_reminder(args.message, args.target_subagent, args.log_file, timestamp=args.timestamp)
            count += 1
            if args.iterations > 0 and count >= args.iterations:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        sys.stdout.write("\nReminder script stopped by user.\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
