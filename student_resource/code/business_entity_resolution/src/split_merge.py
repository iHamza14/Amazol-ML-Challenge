"""
Run inference as two concurrent processes on disjoint sets of countries, then merge their outputs.

Every country is processed independently by inference.py (its own S2/S3 pool, Blocker, pruner threshold,
decision thresholds and France calibration; S2/S3 records never cross countries), so the merged files are
identical to a single run. Two processes use the CPU better: one retrieves (16 threads) while the other is in
its main-process-bound feature phase.

    python -B split_merge.py split --test-dir /workspace/dataset/test --out-a /root/part_a --out-b /root/part_b --countries-b india
    # run inference.py twice (--test-dir /root/part_a --output-dir /workspace/out_a, and part_b / out_b)
    python -B split_merge.py merge --test-dir /workspace/dataset/test --parts /workspace/out_a /workspace/out_b --out /workspace/er_project/output

split copies test_source1.tsv line by line (byte-exact rows) and symlinks the S2/S3 files; merge writes both
output files in the original test order and checks that every test S1 entity appears exactly once.
"""
import os
import argparse


def split(test_dir, out_a, out_b, countries_b):
    cb = {c.strip().lower() for c in countries_b.split(',') if c.strip()}
    with open(os.path.join(test_dir, 'test_source1.tsv'), encoding='utf-8', newline='') as f:
        lines = f.readlines()
    header, rows = lines[0], lines[1:]
    assert header.rstrip('\r\n').split('\t')[-1].strip().lower() == 'country', header
    part_a, part_b = [header], [header]
    counts = {}
    for ln in rows:
        c = ln.rstrip('\r\n').split('\t')[-1].strip().lower()
        counts[c] = counts.get(c, 0) + 1
        (part_b if c in cb else part_a).append(ln)
    assert len(part_a) + len(part_b) - 2 == len(rows)
    src = os.path.abspath(test_dir)
    for d, part in ((out_a, part_a), (out_b, part_b)):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'test_source1.tsv'), 'w', encoding='utf-8', newline='') as f:
            f.writelines(part)
        for fn in ('test_source2.tsv', 'test_source3.tsv'):
            dst = os.path.join(d, fn)
            if os.path.lexists(dst):
                os.remove(dst)
            try:
                os.symlink(os.path.join(src, fn), dst)
            except OSError:
                import shutil
                shutil.copyfile(os.path.join(src, fn), dst)
        print(f'{d}: {len(part) - 1:,} S1 rows')
    print(f'countries in the test set: {counts}; part B = {sorted(cb)}')


def merge(test_dir, parts, out_dir):
    with open(os.path.join(test_dir, 'test_source1.tsv'), encoding='utf-8', newline='') as f:
        next(f)
        order = [ln.split('\t', 1)[0] for ln in f]
    os.makedirs(out_dir, exist_ok=True)
    for fn in ('matching_results.tsv', 'candidate_pairs.tsv'):
        rows, header = {}, None
        for p in parts:
            with open(os.path.join(p, fn), encoding='utf-8', newline='') as f:
                h = f.readline()
                header = header or h
                assert h == header, (fn, h, header)
                for ln in f:
                    if not ln.strip():
                        continue
                    sid = ln.split('\t', 1)[0]
                    assert sid not in rows, f'{fn}: duplicate S1 id {sid}'
                    rows[sid] = ln if ln.endswith('\n') else ln + '\n'
        missing = [s for s in order if s not in rows]
        assert not missing, f'{fn}: {len(missing)} test S1 ids missing, e.g. {missing[:3]}'
        assert len(rows) == len(order), (fn, len(rows), len(order))
        with open(os.path.join(out_dir, fn), 'w', encoding='utf-8', newline='') as f:
            f.write(header)
            f.writelines(rows[s] for s in order)
        n_links = sum(1 for s in order if rows[s].rstrip('\n').split('\t', 1)[1])
        print(f'{fn}: {len(order):,} rows merged ({n_links:,} non-empty) -> {os.path.join(out_dir, fn)}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('split')
    s.add_argument('--test-dir', required=True)
    s.add_argument('--out-a', required=True)
    s.add_argument('--out-b', required=True)
    s.add_argument('--countries-b', default='india')
    m = sub.add_parser('merge')
    m.add_argument('--test-dir', required=True)
    m.add_argument('--parts', nargs='+', required=True)
    m.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.cmd == 'split':
        split(a.test_dir, a.out_a, a.out_b, a.countries_b)
    else:
        merge(a.test_dir, a.parts, a.out)


if __name__ == '__main__':
    main()
