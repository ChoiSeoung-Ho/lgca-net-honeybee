#!/usr/bin/env python3
"""Copy (or symlink) the images listed in data/image_lists/ out of an extracted AI-Hub
archive into the task folders expected by the pipeline.

    python code/select_images.py --aihub-root /path/to/71667 --out code/data [--link]
"""
import argparse, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LISTS = os.path.join(HERE, "..", "data", "image_lists")
TASKS = {"you_chalk_brood": ("normal", "abnormal"), "you_foulbrood": ("normal", "abnormal")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aihub-root", required=True, help="root of the extracted AI-Hub dataset")
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--link", action="store_true", help="symlink instead of copy")
    a = ap.parse_args()
    index = {}
    for root, _, files in os.walk(a.aihub_root):
        for f in files:
            if f.lower().endswith(".jpg"):
                index.setdefault(f, os.path.join(root, f))
    missing, done = [], 0
    for task, classes in TASKS.items():
        for cls in classes:
            lst = os.path.join(LISTS, f"{task}_{cls}.txt")
            dst = os.path.join(a.out, task, cls); os.makedirs(dst, exist_ok=True)
            for name in (l.strip() for l in open(lst, encoding="utf-8") if l.strip()):
                src = index.get(name)
                if src is None:
                    missing.append(name); continue
                tgt = os.path.join(dst, name)
                if os.path.exists(tgt):
                    continue
                (os.symlink if a.link else shutil.copy2)(os.path.abspath(src), tgt); done += 1
    print(f"placed {done} files; {len(missing)} missing")
    for m in missing[:20]:
        print("  missing:", m)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
