import hashlib
import os


def exact_dedup(input_files: list[os.PathLike], output_directory: os.PathLike):
    hash_func = hashlib.md5

    line_counts = {}

    for file in input_files:
        with open(file, 'r') as f:
            for line in f:
                line_hash = hash_func(line.encode()).hexdigest()
                line_counts[line_hash] = line_counts.get(line_hash, 0) + 1

    for file in input_files:
        with open(file, 'r') as f:
            output_lines = []
            for line in f:
                line_hash = hash_func(line.encode()).hexdigest()
                
                if line_counts[line_hash] == 1:
                    output_lines.append(line)

            with open(Path(output_directory) / Path(file).name, 'w') as f_out:
                f_out.write(''.join(output_lines))

