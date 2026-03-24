# EyeGazer Integration

A Python app that connects to the [Beam Eye Tracker](https://beam.eyeware.tech/) SDK,
receives real-time gaze coordinates, captures a 400×400 px screen region centred on
where you're looking, and logs each frame to the console.

---

## Project Structure

```
EyeGazerIntergration/
├── main.py            # Entry point – run this
├── gaze_tracker.py    # Beam API wrapper & gaze polling
├── screen_capture.py  # Screen region capture (mss + Pillow)
├── logger.py          # In-memory rolling log
└── requirements.txt   # pip-installable dependencies
```

---

## Prerequisites

### 1 — Beam Eye Tracker app

Download and install **Beam Eye Tracker** from:  
<https://beam.eyeware.tech/>

The app must be running (and tracking) before you launch this script.

### 2 — Beam Python SDK (`eyeware-beam`)

The Python bindings are **not on PyPI** — they ship with the Beam SDK.

After installing Beam, locate the Python wheel inside the SDK folder.  
It is typically found at a path like:

```
C:\Program Files\Eyeware\Beam Eye Tracker\SDK\python\eyeware_beam-*.whl
```

Install it with:

```powershell
pip install "C:\Program Files\Eyeware\Beam Eye Tracker\SDK\python\eyeware_beam-<version>-cp3xx-cp3xx-win_amd64.whl"
```

> Adjust the path and filename to match your installation.

### 3 — Other Python dependencies

```powershell
pip install -r requirements.txt
```

---

## Configuration

Open `gaze_tracker.py` and update the screen resolution constants if needed:

```python
SCREEN_WIDTH  = 1920   # your monitor width  in pixels
SCREEN_HEIGHT = 1080   # your monitor height in pixels
```

---

## Running

```powershell
python main.py
```

Press **Ctrl+C** to stop.

### Sample output

```
[main] Initialising Beam Eye Tracker API…
[main] Waiting for tracking data  (Ctrl+C to quit)…
------------------------------------------------------------
[10:25:01.042] gaze=( 960,  540) conf=HIGH         capture=400x400px
[10:25:01.084] gaze=( 962,  538) conf=HIGH         capture=400x400px
[10:25:01.126] gaze=( 958,  542) conf=MEDIUM       capture=400x400px
…
```

---

## Future Work

- **AI image analysis** — pass each `entry.image` (PIL Image) to a vision model
  (e.g. OpenAI GPT-4o, Google Gemini) to describe what the user is looking at.
- **PyQt / PySide UI** — overlay window showing the live gaze point and
  captured region in real time.
