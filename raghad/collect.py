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


def RecordSample(flex, right_imu, left_imu, rate_hz, trigger, max_steps=None):
    #samples at rate_hz until triggered or until max_steps timesteps are collected.
    #returns numpy array of shape (T, 11) and the duration in seconds.
    period = 1.0 / rate_hz
    buffer = []

    last_right = (0.0, 0.0, 0.0)
    last_left = (0.0, 0.0, 0.0)

    next_tick = time.perf_counter() # a high-res timer, good for measuring intervals (for short durations)
    t0 = next_tick #start time of the recording

    while not trigger.pressed.is_set(): #if enter isn't pressed, keep recording
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

        #auto-stop once we hit the timestep cap
        if max_steps is not None and len(buffer) >= max_steps:
            break

        next_tick += period
        sleep_for = next_tick - time.perf_counter() #how much time left to the next tick to start recording the next.
        if sleep_for > 0: #if there's still time, means the loop is quick, so we wait till it finish to then go next tick
            time.sleep(sleep_for)
        else:
            next_tick = time.perf_counter() #otherwise dont accumulate delay, just reset the next tick to nowwwww

    duration = time.perf_counter() - t0
    arr = np.array(buffer, dtype=np.float32)
    return arr, duration


def NextSampleIndex(word_dir, word, session):
    if not os.path.isdir(word_dir):
        return 1
    existing = [f for f in os.listdir(word_dir)
                if f.startswith(f"{word}_{session}_") and f.endswith(".npy")]
    return len(existing) + 1


def AppendtoLog(LogPath, row):
    new_file = not os.path.exists(LogPath)
    with open(LogPath, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["filepath", "word", "session", "sample_idx",
                        "n_timesteps", "duration_s", "rate_hz", "timestamp"])
        w.writerow(row)


def PromptWords():
    #ask the user to type words interactively. one word per line.
    #empty line ends the entry phase.
    print("\nEnter the words you want to record, one per line.")
    print("Press Enter on an empty line when you are done.")
    words = []
    while True:
        try:
            val = input(f"  Word {len(words) + 1}: ").strip()
        except EOFError:
            break
        if not val:
            if not words:
                print("  You must enter at least one word.")
                continue
            break
        words.append(val)
    return words


def PromptOption(prompt, valid):
    #asks until the user enters one of the valid chars
    while True:
        ans = input(prompt).strip().lower()
        if ans in valid:
            return ans


def RecordWords(word, flex, right_imu, left_imu, args, raw_dir, LogPath,
                repeat_idx=None, total_repeats=None):

    #records one sample of a given word
    word_dir = os.path.join(raw_dir, word)
    os.makedirs(word_dir, exist_ok=True)
    idx = NextSampleIndex(word_dir, word, args.session)

    suffix = f" (rep {repeat_idx}/{total_repeats})" if repeat_idx else ""
    print(f"\n  Now the word is '{word}'{suffix}. Sample #{idx}.")
    print("  Be ready to start recording. Press Enter to START.")

    StartTrigger = EnterTrigger() # thread
    StartTrigger.arm() #flag is false + thread started
    StartTrigger.pressed.wait() #wait for Enter to be pressed -> here the trigger flag becomes true

    #countdown
    for i in range(3, 0, -1):
        print(f"  Starting in {i}...", end="\r", flush=True) #flush y3ni output the print immediatly with no buffering
        #3shan ykon real time counting ....
        time.sleep(1)
    print("                    ", end="\r")

    #start here!!!!

    cap = args.max_steps if args.max_steps > 0 else None
    if cap is not None:
        print(f"  RECORDING... Press Enter to STOP (auto-stops at {cap} timesteps).")
    else:
        print("  RECORDING... Press Enter to STOP.")

    stop_trigger = EnterTrigger()
    stop_trigger.arm()

    arr, duration = RecordSample(flex, right_imu, left_imu, args.rate,
                                 stop_trigger, max_steps=cap)

    filename = f"{word}_{args.session}_{idx:03d}.npy"
    filepath = os.path.join(word_dir, filename)

    expected = int(duration * args.rate) #expected T -> how many samples we expected to get based on the duration and the rate
    drift = arr.shape[0] - expected
    print(f"  Captured: shape={arr.shape}  duration={duration:.2f}s  drift={drift:+d} samples")

    #print the full numpy array for validation
    print("  Recorded data:")
    with np.printoptions(threshold=np.inf, suppress=True, precision=2, linewidth=200):
        print(arr)

    choice = PromptOption(
        "  [Enter]=accept, r=redo, s=skip, q=quit: ",
        {"", "r", "s", "q"}
    )

    if choice == "": #enter
        np.save(filepath, arr)
        AppendtoLog(LogPath, [
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
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--session", default=datetime.now().strftime("s%Y%m%d"))
    parser.add_argument("--rate", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1,
                        help="How many times to record each word in a row before moving on.")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Skip ahead to the Nth word in the list (0-indexed).")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Auto-stop recording after this many timesteps. Set to 0 to disable.")
    args = parser.parse_args()

    raw_dir = os.path.join(args.data_dir, "raw")
    LogPath = os.path.join(args.data_dir, "Logs.csv")
    os.makedirs(raw_dir, exist_ok=True)

    words = PromptWords()
    #error handling
    if not words:
        print("No words entered.")
        return
    if args.start_from >= len(words):
        print(f"--start-from {args.start_from} is past the end of the list ({len(words)} words)")
        return

    print(f"\nLoaded {len(words)} word(s): {', '.join(words)}")
    print(f"Session: {args.session}  |  Rate: {args.rate} Hz  |  Repeats per word: {args.repeats}")
    if args.max_steps > 0:
        print(f"Max timesteps per recording: {args.max_steps}")
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

    QuitReq = False
    try:
        for word_i, word in enumerate(words[args.start_from:], start=args.start_from):
            if QuitReq:
                break
            print(f"\n=== Word {word_i + 1}/{len(words)}: '{word}' ===")

            rep = 1
            while rep <= args.repeats:
                result = RecordWords(
                    word, flex, right_imu, left_imu, args,
                    raw_dir, LogPath,
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
                    QuitReq = True
                    break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        right_imu.close()
        left_imu.close()
        print("\nDone.")


if __name__ == "__main__":
    main()
