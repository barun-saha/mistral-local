import argparse
import os
import requests
import shutil

from tqdm.auto import tqdm


# https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF
QUANT_METHOD = [
    'Q2_K', 'Q3_K_S', 'Q3_K_M', 'Q3_K_L', 'Q4_0',
    'Q4_K_S', 'Q4_K_M', 'Q5_0', 'Q5_K_S', 'Q5_K_M',
    'Q6_K', 'Q8_0'
]
MODEL_PATH = 'models'


if not os.path.isdir(MODEL_PATH):
    os.mkdir(MODEL_PATH)

parser = argparse.ArgumentParser()
parser.add_argument(
    'quant_method',
    help='quantization method for Mistral 7B-instruct v0.1',
    type=str,
    choices=QUANT_METHOD
)
parser.add_argument(
    '-v',
    '--verbose',
    help='increase the verbosity',
    action='store_true'  # If the option is specified, assign the value True to args.verbose
)

args = parser.parse_args()
quant_method = args.quant_method

file_name = f'mistral-7b-instruct-v0.1.{quant_method}.gguf'
model_url = f'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/{file_name}'
is_file_existing = os.path.isfile(os.path.join(MODEL_PATH, file_name))

if args.verbose:
    print(f'The quantization method is: {quant_method}')

    if is_file_existing:
        print('The specified file already exists...skipping download')
    else:
        print(f'Downloading model: {model_url}')


if not is_file_existing:
    with requests.get(model_url, stream=True) as request:
        total_length = int(request.headers.get('Content-Length'))

        with tqdm.wrapattr(request.raw, 'read', total=total_length, desc='') as raw:
            with open(os.path.join(MODEL_PATH, file_name), 'wb') as output:
                shutil.copyfileobj(raw, output)

    if args.verbose:
        print('Download complete')
