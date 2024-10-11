# see https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering
import subprocess

command = [
    'python',
    'run_qa.py',
    '--model_name_or_path',
    '',
    '--dataset_name',
    '',
    '--do_eval',
    '--max_seq_length',
    '',
    '--doc_stride',
    '',
    '--output_dir',
    ''
]

subprocess.run(command)