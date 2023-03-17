import argparse
import csv
import glob
import os

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="nb, pb, b の数を調べるツール")
    p.add_argument("dir")
    args = p.parse_args()
    dir = args.dir
    len_nb = 0
    len_pb = 0
    len_b = 0
    files = glob.glob(os.path.join(dir, "*.csv"))
    for file in files:
        with open(file) as fh:
            rows = csv.reader(fh, delimiter=",")
            for row in rows:
                speaker = row[0]
                text = row[1]
                if speaker == "U":
                    continue
                nb = row[2]
                pb = row[3]
                b = row[4]
                if nb > pb and nb > b:
                    len_nb += 1
                elif pb > b:
                    len_pb += 1
                else:
                    len_b += 1

    print(f"nb = {len_nb}, pb = {len_pb}, b = {len_b}")
