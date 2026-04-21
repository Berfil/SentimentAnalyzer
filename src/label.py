"""
Interactive label correction tool for native speaker review.

Loads scraped_comments.csv (with model predictions), shows each comment,
and lets the reviewer confirm or correct the label. Progress is saved
after every entry so quitting mid-session loses nothing.

Usage:
    python src/label.py --input data/scraped_comments.csv
                        --output data/labelled_comments.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

VALID_KEYS = {"p": "positive", "n": "negative", "u": "neutral", "s": "skip"}


def clear_line():
    print()


def print_header(current: int, total: int, remaining: int):
    print("\n" + "=" * 55)
    print(f"  Comment {current}/{total}   |   {remaining} remaining")
    print("=" * 55)


def prompt_label(comment: str, model_label: str, score: float) -> str | None:
    print(f"\n{comment}\n")
    print(f"  Model says: {model_label.upper()} (confidence {score:.2f})")
    print()
    print("  [p] positive   [n] negative   [u] neutral")
    print("  [enter] agree with model      [s] skip     [q] quit")
    print()

    while True:
        raw = input("  Your label: ").strip().lower()

        if raw == "q":
            return None
        if raw == "s":
            return "skip"
        if raw == "":
            return model_label
        if raw in VALID_KEYS:
            return VALID_KEYS[raw]

        print("  Invalid input. Use p / n / u / enter / s / q")


def run(input_path: str, output_path: str):
    df = pd.read_csv(input_path, encoding="utf-8-sig")

    required = {"comment", "sentiment", "score"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        print(f"Error: input CSV is missing columns: {missing}")
        print("Run src/analyse.py first to generate predictions.")
        sys.exit(1)

    # load existing output to resume where we left off
    out_path = Path(output_path)
    if out_path.exists():
        done_df = pd.read_csv(out_path, encoding="utf-8-sig")
        done_comments = set(done_df["comment"].tolist())
        print(f"Resuming — {len(done_comments)} comments already labelled.")
    else:
        done_df = pd.DataFrame(columns=list(df.columns) + ["human_label"])
        done_comments = set()

    pending = df[~df["comment"].isin(done_comments)].reset_index(drop=True)
    total = len(df)
    labelled_this_session = []

    print(f"\n{len(pending)} comments to review.")
    print("Press enter to agree with the model, or type a correction.")

    for i, row in pending.iterrows():
        current_num = len(done_comments) + i + 1
        print_header(current_num, total, len(pending) - i)

        result = prompt_label(
            comment=row["comment"],
            model_label=row["sentiment"],
            score=row["score"],
        )

        if result is None:  # quit
            print("\nSaving progress and quitting...")
            break

        if result == "skip":
            continue

        entry = row.to_dict()
        entry["human_label"] = result
        labelled_this_session.append(entry)

        # save incrementally after each label
        session_df = pd.DataFrame(labelled_this_session)
        combined = pd.concat([done_df, session_df], ignore_index=True)
        combined.to_csv(out_path, index=False, encoding="utf-8-sig")

    total_labelled = len(done_df) + len(labelled_this_session)
    print(f"\nDone. {len(labelled_this_session)} labelled this session.")
    print(f"Total labelled: {total_labelled} / {total}")
    print(f"Saved to {output_path}")

    if total_labelled > 0:
        combined = pd.concat(
            [done_df, pd.DataFrame(labelled_this_session)], ignore_index=True
        )
        dist = combined["human_label"].value_counts()
        print("\nLabel distribution so far:")
        for label, count in dist.items():
            print(f"  {label:<12}: {count}")

        corrections = combined[combined["human_label"] != combined["sentiment"]]
        print(f"\nModel corrections: {len(corrections)} / {total_labelled} "
              f"({len(corrections)/total_labelled*100:.1f}% error rate)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/scraped_comments.csv")
    parser.add_argument("--output", default="data/labelled_comments.csv")
    args = parser.parse_args()
    run(args.input, args.output)
