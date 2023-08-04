import re
import os
import glob
import json
import argparse

from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from data.github.preprocessing.src.code_tokenizer import (
    tokenize_python
)


def process_bimodal_instance(ex):
    try:
        _tokens = tokenize_python(ex['code'], keep_comments=True)
        tokenized_code = ' '.join(_tokens)
    except:
        return None

    return tokenized_code


def prepare(args):
    pool = Pool(min(cpu_count(), args.workers))
    src_dir = os.path.join(args.source_dir)
    with open(f'{args.target_dir}/{args.split}.functions.tok', 'w', encoding='utf-8') as writer:
        for file in glob.glob(f"{src_dir}/python_{args.split}_*.jsonl"):
            filename, _ = os.path.splitext(os.path.basename(file))
            with open(file) as f:
                data = [json.loads(line.strip()) for line in f]

            results = []
            with tqdm(total=len(data), desc=f"{filename}") as pbar:
                for out in pool.imap(process_bimodal_instance, data, 1000):
                    pbar.update()
                    results.append(out)

            for tokenized_code in results:
                if tokenized_code:
                    try:
                        writer.write(tokenized_code + '\n')
                    except:
                        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir", help='Source directory',
    )
    parser.add_argument(
        "--target_dir", help="Output directory to save tokenized functions",
    )
    parser.add_argument(
        "--split", type=str, default='train', help='Dataset split',
    )
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    prepare(args)
