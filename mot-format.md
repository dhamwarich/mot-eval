# 📄 `mot_format.md`

# MOT Tracking Output Format (For All Team Members)

This document defines the **required output format** for all tracking models in our project.
Every teammate must convert their tracking output into this format *before* running the evaluator:

```
python evaluate.py <your_output_file> --dataset <dataset-name>
```

You may @this file in your AI IDE (Cursor, Windsurf, Copilot, etc.) to generate conversion code.

---

# 🎯 Overview

Your tracker must output a **single text file** (`.txt`, `.mot`, `.out`, anything) containing **one row per detection per frame**, following the **MOTChallenge-style format**:

```
frame,track_id,x,y,width,height,confidence,-1,-1,-1
```

The evaluator reads this automatically.

---

# 📌 Required Line Format

Each detection must be written exactly as:

```
<frame>,<id>,<x>,<y>,<w>,<h>,<confidence>,-1,-1,-1
```

### Field meanings:

| Field        | Type     | Description                                                                      |
| ------------ | -------- | -------------------------------------------------------------------------------- |
| `frame`      | int      | Frame number (starting at **1**, incrementing each frame)                        |
| `track_id`   | int      | Tracker’s ID for that person. Can be ANY number as long as consistent over time. |
| `x`          | float    | Bounding box **top-left X** in pixels                                            |
| `y`          | float    | Bounding box **top-left Y** in pixels                                            |
| `width`      | float    | Bounding box width (pixels)                                                      |
| `height`     | float    | Bounding box height (pixels)                                                     |
| `confidence` | float    | Detection confidence (0–1). Use 1.0 if your tracker doesn’t provide scores.      |
| `-1,-1,-1`   | constant | Required placeholders for MOT format compatibility                               |

---

# 🧠 Important Rules

### **1. Frames must start at 1**

Correct:

```
1,3,120,80,150,300,0.91,-1,-1,-1
2,3,118,81,150,300,0.90,-1,-1,-1
```

Incorrect:

```
0,...
5,...
100,...
```

---

### **2. ID numbers DO NOT need to match ground truth**

You can output anything:

```
frame 1: Predicted ID 35
frame 2: Predicted ID 35
...
```

Evaluator handles matching automatically using IoU.

---

### **3. IDs MUST remain consistent for the same person**

Good:

```
frame 1: ID 3
frame 2: ID 3
frame 3: ID 3
```

Bad:

```
frame 1: ID 3
frame 2: ID 99   <-- counts as identity switch
frame 3: ID 3
```

---

### **4. Coordinates must be absolute pixel values**

Correct:

```
x=120
y=80
w=150
h=300
```

Wrong:

* normalized 0–1
* x1,y1,x2,y2
* center/width/height

If your model outputs xyxy → convert:

```python
x = x1
y = y1
w = x2 - x1
h = y2 - y1
```

---

### **5. Sort rows by frame, then ID**

Example sorted order:

```
1,1,...
1,2,...
1,4,...
2,1,...
2,2,...
```

Not required for correctness but highly recommended.

---

### **6. One file per tracking run**

Example:

```
test_output.mot
mall_prediction.txt
corridor_sort_output.txt
```

Name does **not** matter.
Evaluator accepts ANY filename.

---

# 📚 Complete Example MOT File

```
1,3,120,80,150,300,0.97,-1,-1,-1
1,1,500,120,140,280,0.92,-1,-1,-1
2,3,122,82,150,299,0.95,-1,-1,-1
2,1,505,121,140,280,0.93,-1,-1,-1
3,3,123,82,150,298,0.94,-1,-1,-1
```

---

# 🧰 Conversion Snippets for AI IDEs

### Convert `xyxy` → MOT:

```python
lines = []
for frame_id, det in enumerate(detections, start=1):
    for (x1, y1, x2, y2, track_id, score) in det:
        w = x2 - x1
        h = y2 - y1
        line = f"{frame_id},{track_id},{x1},{y1},{w},{h},{score},-1,-1,-1"
        lines.append(line)
```

### Convert YOLO format (`cls, x_center, y_center, w, h`):

```python
x = x_center - w/2
y = y_center - h/2
```

Then write the MOT line.

---

# ✔️ Summary Checklist

Before calling:

```
python evaluate.py output.mot --dataset datasetX
```

Make sure:

* [ ] File uses MOT line format
* [ ] Pixel coordinates (not normalized)
* [ ] Frame numbers start at 1
* [ ] Same person keeps same ID
* [ ] File is `.txt` or `.mot` (any extension ok)

---