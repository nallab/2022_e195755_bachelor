import argparse
import csv
import glob
import logging
import os

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s- %(name)s - %(levelname)s - %(message)s",
    )
    p = argparse.ArgumentParser(description="tool to select nb dialog")
    p.add_argument("source_data_dir")

    args = p.parse_args()
    data_dir = args.source_data_dir
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    for file in files:
        with open(file) as fh:
            rows = csv.reader(fh, delimiter=",")
            is_head = True
            is_break = False
            count = 0
            for row in rows:
                if is_head:
                    is_head = False
                    continue
                speaker = row[0]
                if speaker == "U":
                    continue
                nb = row[2]
                pb = row[3]
                b = row[4]
                if pb + b > nb:
                    # logger.debug("break")
                    is_break = True
                    break
                count += 1
            # logger.info(count)
            if count >= 7:
                # logger.info(file)
                print(f"cp {file} hoge/")
