# collect.py - Full Reference Documentation

---

## 1. Purpose

`collect.py` records raw sensor data for a closed-vocabulary sign-language
recognition project. Each recording is one gesture for one word, stored
as a NumPy array and indexed in a manifest CSV. The script is the data
collection layer: it does no preprocessing, no normalization, no feature
engineering. Everything downstream (training, inference) reads from the
files this script produces.

The script is designed to be run directly on the Raspberry Pi 4 that is
connected to the sensors.

---

## 2. What one recording contains

Each recording is saved as a 2D NumPy array of shape `(T, 11)` where:

- `T` is the number of time steps, which varies per recording depending
  on how long you held Enter-to-Enter. At 50 Hz, a 2-second gesture
  yields `T = 100`.
- `11` is the fixed number of features per time step, in this exact
  column order:

```
index 0  : flex_thumb        (raw ADC value, ~0 to 32767)
index 1  : flex_pointer
index 2  : flex_middle
index 3  : flex_ring
index 4  : flex_pinky
index 5  : right_yaw         (degrees, from GY-951 UART IMU)
index 6  : right_pitch
index 7  : right_roll
index 8  : left_yaw          (degrees, currently zeros )
index 9  : left_pitch
index 10 : left_roll
```

Data type is `float32`. No normalization is applied. No calibration
offsets are applied. What you see is what the sensors reported.

---

## 3. Output file layout

Starting from whatever directory you pass to `--data-dir` (default
`data`), the script creates:

```
data/
  manifest.csv
  raw/
    <word_1>/
      <word_1>_<session>_001.npy
      <word_1>_<session>_002.npy
      ...
    <word_2>/
      <word_2>_<session>_001.npy
      ...
```

### 3.1 The .npy files

Each file contains one `(T, 11)` float32 array. Load with:

```python
import numpy as np
arr = np.load("data/raw/hello/hello_s20260421_001.npy")
print(arr.shape)   # (T, 11)
```

### 3.2 The manifest.csv

One row per saved recording. Columns:

| Column        | Meaning                                                      |
|---------------|--------------------------------------------------------------|
| filepath      | Relative path to the .npy file                               |
| word          | The word label (class)                                       |
| session       | Session identifier, e.g. `s20260421`                         |
| sample_idx    | Sample number within this word for this session              |
| n_timesteps   | Number of rows in the array (T)                              |
| duration_s    | Wall-clock duration of the recording in seconds              |
| rate_hz       | Sampling rate that was requested                             |
| timestamp     | ISO timestamp of when the sample was saved                   |

The manifest is what your training pipeline will read to build a
dataset. You can filter it to split by session, by word, by duration,
or whatever else you need.

---

## 4. Command-line usage

### 4.1 Minimum invocation

```bash
python collect.py --words words.csv
```

### 4.2 All flags

| Flag            | Default                      | Meaning                                              |
|-----------------|------------------------------|------------------------------------------------------|
| `--words`       | (required)                   | Path to the CSV file containing the word list        |
| `--data-dir`    | `data`                       | Root directory for output                            |
| `--session`     | `s` + today's date (YYYYMMDD)| Session identifier baked into filenames and manifest |
| `--rate`        | `50`                         | Master sampling rate in Hz                           |
| `--repeats`     | `1`                          | Record each word this many times in a row            |
| `--start-from`  | `0`                          | Skip to the Nth word in the list (0-indexed)         |

### 4.3 Common patterns

Record each word once through the whole list:
```bash
python collect.py --words words.csv
```

Record each word five times in a row before moving to the next:
```bash
python collect.py --words words.csv --repeats 5
```

Resume a session that got interrupted at word 40:
```bash
python collect.py --words words.csv --start-from 40
```

Use a custom session name (useful if you want to tag sessions by
subject or conditions, not just by date):
```bash
python collect.py --words words.csv --session subject1_day2
```

---

## 5. The words CSV

The script reads words from the first column of the CSV you pass with
`--words`. Rules:

- One word per row. Only the first column is read; other columns are
  ignored.
- Empty rows are skipped.
- If the first row is a header (exactly one of `word`, `words`, `label`,
  `class`, case-insensitive), it is skipped automatically.
- UTF-8 encoding is required. Arabic, English, or mixed all work.

Example `words.csv`:

```
word
hello
thank_you
please
yes
no
```

A few practical tips:

- Avoid spaces, slashes, or punctuation in the word label because the
  label becomes part of a directory name and filename. Use underscores.
- Keep labels consistent across sessions. If you record as `thank_you`
  one day and `thankyou` another, they will be treated as two different
  classes.
- For Arabic words, the script handles Unicode, but some filesystems
  or git configurations handle non-ASCII filenames awkwardly. If that
  becomes a problem, use ASCII romanizations or numeric IDs (`w001`,
  `w002`, etc.) and keep a separate lookup table.

---

## 6. Interactive workflow

After startup, the script walks through the word list one word at a
time. For each word:

1. The script prints:
   ```
   === Word N/total: '<word>' ===
     Now the word is '<word>'. Sample #<idx>.
     Be ready to start recording. Press Enter to START.
   ```
2. You press Enter. The script prints `RECORDING... Press Enter to STOP.`
3. You perform the gesture and press Enter again.
4. The script prints the captured shape, duration, and timing drift.
5. You choose what to do next:
   - Press Enter alone: accept the recording, save it, move on.
   - Type `r` + Enter: discard this take and redo the same word.
   - Type `s` + Enter: discard this take and skip to the next word.
   - Type `q` + Enter: discard this take and quit the whole session.

The redo option is important: nothing is saved to disk until you
accept, so `r` costs you nothing but a few seconds.

### 6.1 Behavior with --repeats

With `--repeats N`, the script asks you to record the same word N
times before moving on. Each rep is numbered, and the prompt shows
`(rep 2/5)` for example. If you hit `r`, you redo the current rep.
If you hit `s`, you skip the remaining reps and move to the next
word.

---

## 7. Recommended collection protocol

You want variety in your dataset. The single biggest determinant of
whether your model generalizes is whether your training data actually
spans the range of conditions the model will see at inference time.

### 7.1 Session structure

Prefer many sessions across different days over one mega-session on
one day. Concretely:

- Target 5 to 10 samples per word per session.
- Run 4 to 6 sessions spread across different days.
- Each session, physically take off and re-don the glove at least
  once partway through. This introduces placement variation that
  the model needs to learn to ignore.

This matters because sensors drift. Flex sensors especially show
slow baseline shift over minutes, and their response to a given
bend angle depends a lot on where exactly the strip sits on your
finger. If all 50 samples of a word come from one continuous
wearing of the glove, the model can memorize that specific
placement and still fail when you put the glove on differently
tomorrow.

### 7.2 Gesture timing

Try to keep gesture duration reasonably consistent across takes of
the same word. If some takes are 1 second and others are 4 seconds,
the model has to learn to handle large duration variance, which
eats capacity. Aim for a natural, comfortable pace and be consistent.

The timestamps between Enter presses are yours to control, so
practice a couple of takes per word before committing to real data.

### 7.3 Pilot before scaling

Do not jump straight to collecting all 120 words. First run a pilot:

- 5 words.
- 10 samples each.
- Full pipeline: startup, record, accept, next word, exit.

Then open a few of the saved .npy files and verify:
- Flex values change visibly when you bend fingers during the take.
- Right IMU YPR values actually update (not all zeros or flatlined).
- Shape is roughly `(expected_T, 11)` where `expected_T = rate * duration`.
- `drift` printed by the script stays within a few samples.

Only after this passes should you commit time to the full dataset.

---

## 8. How the script works internally

### 8.1 Module structure

The script is organized into:

- `LeftIMUStub`: placeholder class for the ESP32 left-hand IMU.
  Returns zeros. You will replace this with a real receiver.
- `EnterTrigger`: helper that waits for an Enter press in a background
  thread and sets an event flag. Enables the main loop to poll for
  "was Enter pressed yet?" without blocking.
- `record_one_sample`: the core sampling loop. Reads all three sensors
  on each tick, accumulates rows, returns a NumPy array when stopped.
- `next_sample_index`, `append_manifest`, `load_words`, `prompt_choice`:
  small file/IO helpers.
- `record_word`: records one sample of a given word, prompts the user
  for what to do with it, saves if accepted.
- `main`: parses args, initializes sensors, walks through the word list,
  calls `record_word` repeatedly.

### 8.2 The sampling loop

This is the heart of the script. Pseudocode:

```
period = 1 / rate_hz
next_tick = now()
while not stop_pressed:
    flex_vals = flex.read()                  # always fresh
    right_ypr = right_imu.read()             # may return Nones
    if right_ypr is fresh:
        last_right = right_ypr
    left_ypr = left_imu.read()               # stub for now
    if left_ypr is fresh:
        last_left = left_ypr
    append row [flex_vals..., last_right..., last_left...]
    next_tick += period
    sleep until next_tick
```

Two subtle but important details:

**Zero-order hold.** The flex sensors return fresh values every read.
The IMU does not: it returns a new reading only when a new line has
arrived on the serial port. So we hold the last valid IMU reading
and reuse it on ticks where nothing new arrived. This gives a clean
fixed-rate output even though the IMU stream is bursty. The same
logic will apply when the left IMU is real (packets arriving over
WiFi or ESP-NOW will not be perfectly synced to our tick).

**Absolute-time scheduling.** Instead of `sleep(period)` each iteration
(which accumulates drift because each iteration also takes some
time), we track the absolute time of the next tick and sleep to that
deadline. If a tick falls behind, we reset the deadline to `now()`
rather than trying to catch up with back-to-back iterations. This
keeps the effective rate close to the target even when individual
reads take longer than expected.

### 8.3 The drift value

After each recording, the script prints:
```
Captured: shape=(T, 11)  duration=2.03s  drift=+0 samples
```

`drift` is `actual_T - expected_T` where `expected_T = duration * rate`.
A drift of +/- 0 to 2 is normal and fine. If drift is growing large
and positive, the loop is running too fast (unlikely). If it is
growing large and negative, something is blocking the loop: most
likely a slow serial read on the IMU. If you see this, investigate
before collecting at scale, because it means your effective rate
is below the nominal 50 Hz.

### 8.4 The Enter trigger

Reading from stdin blocks by default, so we can't just call `input()`
inside the sampling loop. Instead, `EnterTrigger` starts a daemon
thread that blocks on `input()`, and when Enter is pressed, the
thread sets a `threading.Event`. The main loop polls
`event.is_set()` every tick. This is simple and avoids platform-
specific raw-terminal input handling.

One consequence: anything you type *before* pressing Enter during
a recording is consumed by that `input()` call but discarded. So
if you accidentally type characters mid-recording, they disappear
when you hit Enter. This is harmless.

---

## 9. Integrating the real left-hand IMU

When the ESP32 link is ready, replace `LeftIMUStub` with a class that
matches this interface:

```python
class LeftIMU:
    def __init__(self):
        # open the socket / BLE / ESP-NOW connection
        pass

    def read(self):
        # non-blocking
        # return (yaw, pitch, roll) for the most recent packet
        # return (None, None, None) if nothing new since last call
        pass

    def close(self):
        # tear down the connection
        pass
```

Then in `main()`, change `left_imu = LeftIMUStub()` to
`left_imu = LeftIMU()`. Nothing else in the script changes.

A few things to plan for:

- Your ESP32 should send packets at roughly 50 Hz or faster. If it
  sends much slower, the zero-order hold will produce repeated
  rows, which reduces effective left-hand information.
- Each packet should include a timestamp from the ESP32 side. Log
  it somewhere (even if you don't use it at training time) so you
  can debug synchronization later if needed.
- WiFi adds variable latency. Typical values are 5 to 30 ms on a
  quiet local network, but can spike. The zero-order hold handles
  this gracefully, but you should monitor packet loss.

---

## 10. Common pitfalls and how to avoid them

**Inconsistent word labels across sessions.** If you record as
`thank_you` one day and `thankyou` another, the manifest shows two
classes with half the data each. Fix by keeping one canonical
`words.csv` and always using `--words words.csv`. Never edit labels
between sessions.

**Sensor placement drift during long sessions.** Flex sensors can
slip on fingers over time. Check placement every 10 to 20 minutes.
If a strip has moved significantly, note the session and consider
excluding it.

**Forgetting the glove is on upside-down or sensors are swapped.**
Always run a quick sanity check at the start of each session: bend
each finger in isolation and watch the printed values to confirm
the correct sensor is reacting. The script currently does not have
a built-in sanity mode, but you can just run the `__main__` of
`flexsensors.py` directly.

**Recording too briefly.** If `T` is very small (say under 20), the
model has very little to work with. Try to keep gestures at least
1 second long. If a word is naturally fast, record it slightly
slower during collection than you would at inference time; you can
make inference robust later but during collection you want clean
signal.

**Mid-session crashes losing data.** Data is written to disk
immediately on accept, and the manifest is appended per sample, so
a crash loses only the in-flight recording (which is still in RAM
waiting for your accept/redo decision). Just restart with
`--start-from N` where N is where you left off.

**Git accidentally tracking binary .npy files.** If you version the
code alongside the data, add `data/` to `.gitignore`. The raw
dataset at 120 words times 30 samples can be hundreds of MB.

---

## 11. Quick reference: what to do, in order

1. Write your `words.csv` with one word per row.
2. Connect all sensors to the Pi and verify they read sensible values.
3. Run a pilot: `python collect.py --words words.csv` with 2 or 3 words
   only (edit the CSV temporarily, or use `--start-from` as a poor
   man's filter).
4. Open some saved .npy files in Python and inspect them.
5. Once satisfied, run the full session with your real words CSV and
   `--repeats 5` or similar.
6. Record 4 to 6 sessions across different days.
7. Back up `data/` somewhere safe before starting training.

---

## 12. Feature column reference card

Print this and tape it to your monitor:

```
.npy column layout, shape (T, 11):

 0  flex_thumb
 1  flex_pointer
 2  flex_middle
 3  flex_ring
 4  flex_pinky
 5  right_yaw     (deg)
 6  right_pitch   (deg)
 7  right_roll    (deg)
 8  left_yaw      (deg, currently stub)
 9  left_pitch    (deg, currently stub)
10  left_roll     (deg, currently stub)

dtype: float32
units: flex = raw ADC (0 to ~32767), IMU = degrees
sample rate: set by --rate (default 50 Hz)
```
