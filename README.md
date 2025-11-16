# 🏃‍♂️ MOT Evaluator (Quick Start)

This tool evaluates your tracking output (in MOT format) against our ground-truth datasets.

## 1️⃣ Run your tracker

Generate a MOT file (any filename is OK).

## 2️⃣ Evaluate

```
python evaluate.py <your_output_file> --dataset <dataset_name>
```

Example:

```
python evaluate.py results/my_output.mot --dataset dataset1
```

## 3️⃣ Output

You’ll see:

* MOTA
* IDF1
* ID switches
* FP / FN
* GT count

## 📄 MOT Format (required)

Each line:

```
frame,id,x,y,w,h,confidence,-1,-1,-1
```

Rules:

* frame starts at **1**
* x,y,w,h = **pixel** bbox
* id = any number, but **consistent per person**
* confidence = 0–1 (or just use 1)

For full details, see `mot_format.md`.