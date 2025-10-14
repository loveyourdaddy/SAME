#!/usr/bin/env python3
# Unordered pairs (default: each pair once, no self-pairs): python make_cross_animal_pairs.py -i samples.txt -o pairs_all.txt
# Ordered pairs (both directions): python make_cross_animal_pairs.py -i pair_origin.txt -o pairs_new.txt --ordered
# Include self-pairs (rare): python make_cross_animal_pairs.py -i samples.txt -o pairs_with_self.txt --allow-self


import argparse, os, sys
from collections import defaultdict
from itertools import combinations, permutations

def parse_path(p):
    """
    'Deer/Deer_WalkBack.npz' -> ('Deer', 'WalkBack', 'npz')
    Robust to slight format drift: falls back to splitting at the first underscore.
    """
    p = p.strip().replace('\\', '/')
    parts = p.split('/')
    if len(parts) < 2:
        return None
    animal = parts[-2]
    fname = parts[-1]
    root, ext = os.path.splitext(fname)  # ('Deer_WalkBack', '.npz')
    ext = ext.lstrip('.')
    if root.startswith(animal + "_"):
        action = root[len(animal) + 1:]
    else:
        action = root.split('_', 1)[-1] if '_' in root else root
    return animal, action, ext

def build_path(animal, action, ext):
    return f"{animal}/{animal}_{action}.{ext}"

def load_left_column_lines(infile):
    lines = []
    with open(infile, 'r', encoding='utf-8') as f:
        for ln, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            left = raw.rstrip('\n').split('\t')[0].strip()
            parsed = parse_path(left)
            if not parsed:
                sys.stderr.write(f"[WARN] Line {ln}: can't parse '{left}', skipping.\n")
                continue
            animal, action, ext = parsed
            lines.append((animal, action, ext))
    return lines

def group_by_action(lines):
    """
    action -> { animal -> ext (first seen) }
    We only need one exemplar per (animal, action); deduplicate multiples.
    """
    action_map = defaultdict(dict)
    for animal, action, ext in lines:
        action_map[action].setdefault(animal, ext)
    return action_map

def make_pairs(action_map, ordered=False, allow_self=False):
    """
    Yield lines "left_path\tright_path" for all pairs that share the action.
    - unordered: each unordered pair once (A-B)
    - ordered: both directions (A->B and B->A)
    - allow_self: include A->A pairs (rarely needed)
    """
    for action, animal_to_ext in action_map.items():
        animals = sorted(animal_to_ext.keys())
        if not animals:
            continue

        if allow_self:
            iter_pairs = ((a, b) for a in animals for b in animals) if ordered \
                         else ((a, a) for a in animals)
        else:
            if ordered:
                iter_pairs = permutations(animals, 2)
            else:
                iter_pairs = combinations(animals, 2)

        for a, b in iter_pairs:
            left = build_path(a, action, animal_to_ext[a])
            right = build_path(b, action, animal_to_ext[b])
            yield f"{left}\t{right}"

def main():
    ap = argparse.ArgumentParser(
        description="For every action, pair ALL animals that have that action."
    )
    ap.add_argument("-i", "--input", required=True, help="Input .txt (tab-separated; only left column is used).")
    ap.add_argument("-o", "--output", required=True, help="Output .txt (tab-separated pairs).")
    ap.add_argument("--ordered", action="store_true",
                    help="Output ordered pairs (A->B and B->A). Default: unordered (A-B once).")
    ap.add_argument("--allow-self", action="store_true",
                    help="Include self-pairs (A->A) per action. Default: False.")
    args = ap.parse_args()

    lines = load_left_column_lines(args.input)
    if not lines:
        sys.stderr.write("[ERROR] No valid entries parsed from input.\n")
        sys.exit(1)

    action_map = group_by_action(lines)
    pairs = list(make_pairs(action_map, ordered=args.ordered, allow_self=args.allow_self))

    with open(args.output, 'w', encoding='utf-8') as w:
        w.write("\n".join(pairs) + ("\n" if pairs else ""))

if __name__ == "__main__":
    main()
