#!/usr/bin/env python3
"""Collect SAME test outputs into a clean, renamed folder.

src/same/test.py writes files named

    pair<idx>__<Src>__TO__<Tgt>__OUT.bvh   (and, after rendering, ...__OUT.mp4)

into a result dir. This copies each into

    <result_dir>/out/<ext>/<SourceAnimal>__<SourceAction>__TO__<TargetAnimal>__<TargetAction>.<ext>

i.e. it drops the `pair<idx>__` prefix and the `__OUT` suffix. .bvh goes to
out/bvh/, .mp4 to out/mp4/. Non-destructive by default (copies; --move to move).
Idempotent: renamed files no longer end in __OUT so they are not re-collected.

Usage:
    python collect_outputs.py --result_dir result/260803_cfg_VT_fold0/test
    python collect_outputs.py --result_dir result/260803_cfg_VT_fold0/test --ext bvh
    python collect_outputs.py --result_dir <dir> --move        # move, don't copy
"""
import argparse
import glob
import os
import re
import shutil

PREFIX_RE = re.compile(r"^pair\d+__")   # pair000123__


def template_name(out_basename: str, ext: str) -> str:
    """pair<idx>__<Src>__TO__<Tgt>__OUT.<ext> -> <Src>__TO__<Tgt>.<ext>"""
    name = PREFIX_RE.sub("", out_basename)          # drop pair<idx>__
    suffix = f"__OUT.{ext}"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    else:                                            # be lenient about the suffix
        name = os.path.splitext(name)[0]
        if name.endswith("__OUT"):
            name = name[: -len("__OUT")]
    return f"{name}.{ext}"


def collect(result_dir: str, exts, move: bool = False, overwrite: bool = True) -> int:
    result_dir = os.path.abspath(result_dir)
    total = 0
    for ext in exts:
        dst_dir = os.path.join(result_dir, "out", ext)
        srcs = sorted(glob.glob(os.path.join(result_dir, "**", f"*__OUT.{ext}"),
                                recursive=True))
        if not srcs:
            print(f"[{ext}] no *__OUT.{ext} found under {result_dir}")
            continue
        os.makedirs(dst_dir, exist_ok=True)
        seen, n = {}, 0
        for s in srcs:
            new = template_name(os.path.basename(s), ext)
            if new in seen:
                print(f"[{ext}] WARN collision: {new} "
                      f"({os.path.basename(s)} vs {seen[new]}) -> overwriting")
            seen[new] = os.path.basename(s)
            dst = os.path.join(dst_dir, new)
            if os.path.abspath(s) == os.path.abspath(dst):
                continue                                   # already correctly named
            if os.path.exists(dst) and not overwrite:
                continue
            in_dst = os.path.dirname(os.path.abspath(s)) == dst_dir
            # rename in place if the old-named file already sits in the target;
            # otherwise copy (keep the original) unless --move was requested
            (shutil.move if (move or in_dst) else shutil.copy2)(s, dst)
            n += 1
        # tidy the target: drop any leftover old-named __OUT files there
        for old in glob.glob(os.path.join(dst_dir, f"*__OUT.{ext}")):
            os.remove(old)
        print(f"[{ext}] {n} files -> {dst_dir}")
        total += n
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result_dir", required=True,
                    help="test dir containing *__OUT.bvh / *__OUT.mp4")
    ap.add_argument("--ext", nargs="+", default=["bvh", "mp4"],
                    help="extensions to collect (default: bvh mp4)")
    ap.add_argument("--move", action="store_true",
                    help="move instead of copy (saves disk)")
    args = ap.parse_args()
    collect(args.result_dir, args.ext, move=args.move)


if __name__ == "__main__":
    main()
