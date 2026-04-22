
import os
import csv
import time
import threading
import argparse
from datetime import datetime

import numpy as np

from flexsensors import FlexSensors
from IMUc import IMUc

#enhanced collecting script 

# ---------- left hand IMU ---------
class LeftIMU:

    def __init__(self):
        self.last = (0.0, 0.0, 0.0)
        print("Left IMU: (zeros)")

    def read(self):
        return self.last

    def close(self):
        pass


# ---------- keyboard trigger ----------
class EnterTrigger:

    def __init__(self):
        self.pressed = threading.Event()
        self._thread = None

    def arm(self):
        self.pressed.clear()
        self._thread = threading.Thread(target=self._wait, daemon=True)
        self._thread.start()

    def _wait(self):
        try:
            input()
        except EOFError:
            pass
        self.pressed.set()


# ============================================================ 

def record_one_sample(flex, right_imu, left_imu, rate_hz, trigger):
    #samples at 50hz until trigger fires. Returns numpy array of shape (T, 11)
    period = 1.0 / rate_hz
    buffer = []

    last_right = (0.0, 0.0, 0.0)
    last_left = (0.0, 0.0, 0.0)

    next_tick = time.perf_counter()
    t0 = next_tick

    while not trigger.pressed.is_set():
        flex_vals = flex.read()

        ry, rp, rr = right_imu.read()
        if ry is not None:
            last_right = (ry, rp, rr)

        ly, lp, lr = left_imu.read()
        if ly is not None:
            last_left = (ly, lp, lr)

        row = [
            flex_vals[0], flex_vals[1], flex_vals[2], flex_vals[3], flex_vals[4],
            last_right[0], last_right[1], last_right[2],
            last_left[0], last_left[1], last_left[2],
        ]
        buffer.append(row)

        next_tick += period
        sleep_for = next_tick - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.perf_counter()

    duration = time.perf_counter() - t0
    arr = np.array(buffer, dtype=np.float32)
    return arr, duration


def next_sample_index(word_dir, word, session):
    if not os.path.isdir(word_dir):
        return 1
    existing = [f for f in os.listdir(word_dir)
                if f.startswith(f"{word}_{session}_") and f.endswith(".npy")]
    return len(existing) + 1


def append_manifest(manifest_path, row):
    new_file = not os.path.exists(manifest_path)
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["filepath", "word", "session", "sample_idx",
                        "n_timesteps", "duration_s", "rate_hz", "timestamp"])
        w.writerow(row)


def load_words(csv_path):
    #Reads words from the first column of a CSV. Skips empty rows.
    #If the first row looks like a header (word), it is skipped
    words = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row:
                continue
            val = row[0].strip()
            if not val:
                continue
            if i == 0 and val.lower() in ("word", "words", "label", "class"):
                continue
            words.append(val)
    return words


def prompt_choice(prompt, valid):
    #asks until the user enters one of the valid chars
    while True:
        ans = input(prompt).strip().lower()
        if ans in valid:
            return ans


def record_word(word, flex, right_imu, left_imu, args, raw_dir, manifest_path,
                repeat_idx=None, total_repeats=None):
    
    #Records one sample of a given word
    word_dir = os.path.join(raw_dir, word)
    os.makedirs(word_dir, exist_ok=True)
    idx = next_sample_index(word_dir, word, args.session)

    suffix = f" (rep {repeat_idx}/{total_repeats})" if repeat_idx else ""
    print(f"\n  Now the word is '{word}'{suffix}. Sample #{idx}.")
    print("  Be ready to start recording. Press Enter to START.")

    start_trigger = EnterTrigger()
    start_trigger.arm()
    start_trigger.pressed.wait()

    print("  RECORDING... Press Enter to STOP.")
    stop_trigger = EnterTrigger()
    stop_trigger.arm()

    arr, duration = record_one_sample(flex, right_imu, left_imu, args.rate, stop_trigger)

    filename = f"{word}_{args.session}_{idx:03d}.npy"
    filepath = os.path.join(word_dir, filename)

    expected = int(duration * args.rate)
    drift = arr.shape[0] - expected
    print(f"  Captured: shape={arr.shape}  duration={duration:.2f}s  drift={drift:+d} samples")

    choice = prompt_choice(
        "  [Enter]=accept, r=redo, s=skip, q=quit: ",
        {"", "r", "s", "q"}
    )

    if choice == "":
        np.save(filepath, arr)
        append_manifest(manifest_path, [
            filepath, word, args.session, idx,
            arr.shape[0], f"{duration:.3f}", args.rate,
            datetime.now().isoformat(timespec="seconds"),
        ])
        print(f"  Saved {filepath}")
        return "ok"
    elif choice == "r":
        print("  Discarded. Redoing this word.")
        return "redo"
    elif choice == "s":
        print("  Discarded. Skipping this word.")
        return "skip"
    else:
        print("  Discarded. Quitting.")
        return "quit"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--words", required=True, help="Path to CSV with one word per row.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--session", default=datetime.now().strftime("s%Y%m%d"))
    parser.add_argument("--rate", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1,
                        help="How many times to record each word in a row before moving on.")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Skip ahead to the Nth word in the list (0-indexed).")
    args = parser.parse_args()

    raw_dir = os.path.join(args.data_dir, "raw")
    manifest_path = os.path.join(args.data_dir, "manifest.csv")
    os.makedirs(raw_dir, exist_ok=True)

    words = load_words(args.words)
    if not words:
        print(f"No words found in {args.words}")
        return
    if args.start_from >= len(words):
        print(f"--start-from {args.start_from} is past the end of the list ({len(words)} words)")
        return

    print(f"Loaded {len(words)} words from {args.words}")
    print(f"Session: {args.session}  |  Rate: {args.rate} Hz  |  Repeats per word: {args.repeats}")
    if args.start_from:
        print(f"Starting from word index {args.start_from}: '{words[args.start_from]}'")

    print("\nInitializing sensors...")
    flex = FlexSensors()
    try:
        right_imu = IMUc()
    except Exception as e:
        print("Right IMU init failed:", e)
        return
    left_imu = LeftIMU()

    print("\nPress Enter to begin the session...")
    input()

    quit_requested = False
    try:
        for word_i, word in enumerate(words[args.start_from:], start=args.start_from):
            if quit_requested:
                break
            print(f"\n=== Word {word_i + 1}/{len(words)}: '{word}' ===")

            rep = 1
            while rep <= args.repeats:
                result = record_word(
                    word, flex, right_imu, left_imu, args,
                    raw_dir, manifest_path,
                    repeat_idx=rep if args.repeats > 1 else None,
                    total_repeats=args.repeats if args.repeats > 1 else None,
                )
                if result == "ok":
                    rep += 1
                elif result == "redo":
                    continue
                elif result == "skip":
                    break
                elif result == "quit":
                    quit_requested = True
                    break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        right_imu.close()
        left_imu.close()
        print("\nDone.")


if __name__ == "__main__":
    main()
