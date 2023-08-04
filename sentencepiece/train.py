import argparse
import sentencepiece as spm


def main(args):
    GITHUB_DIR = args.git_data_dir
    STACKOVERFLOW_DIR = args.so_data_dir
    FILES = []

    num_split = 4
    for i in range(num_split):
        FILES.extend(
            (
                f'{GITHUB_DIR}/java/train.{i}.functions_class.tok',
                f'{GITHUB_DIR}/java/train.{i}.functions_standalone.tok',
                f'{GITHUB_DIR}/python/train.{i}.functions_class.tok',
                f'{GITHUB_DIR}/python/train.{i}.functions_standalone.tok',
            )
        )
    FILES += [
        f'{GITHUB_DIR}/java/valid.functions_class.tok',
        f'{GITHUB_DIR}/java/valid.functions_standalone.tok',
        f'{GITHUB_DIR}/java/test.functions_class.tok',
        f'{GITHUB_DIR}/java/test.functions_standalone.tok',
        f'{GITHUB_DIR}/python/valid.functions_class.tok',
        f'{GITHUB_DIR}/python/valid.functions_standalone.tok',
        f'{GITHUB_DIR}/python/test.functions_class.tok',
        f'{GITHUB_DIR}/python/test.functions_standalone.tok',
    ]

    FILES += [
        f'{STACKOVERFLOW_DIR}/train.0.description.txt',
        f'{STACKOVERFLOW_DIR}/train.1.description.txt',
        f'{STACKOVERFLOW_DIR}/train.2.description.txt',
        f'{STACKOVERFLOW_DIR}/train.3.description.txt',
        f'{STACKOVERFLOW_DIR}/train.4.description.txt',
        f'{STACKOVERFLOW_DIR}/train.5.description.txt',
        f'{STACKOVERFLOW_DIR}/train.6.description.txt',
        f'{STACKOVERFLOW_DIR}/train.7.description.txt',
        f'{STACKOVERFLOW_DIR}/valid.description.txt',
        f'{STACKOVERFLOW_DIR}/test.description.txt',
    ]

    # we may consider adding --user_defined_symbols=INDENT,DEDENT,NEW_LINE
    spm.SentencePieceTrainer.train(
        f"--input={','.join(FILES)} --vocab_size=50000 --model_prefix=sentencepiece.bpe --character_coverage=1.0 --model_type=bpe"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--git_data_dir", type=str, help='Github data directory')
    parser.add_argument("--so_data_dir", type=str, help='Stack Overflow data directory')
    args = parser.parse_args()
    main(args)
