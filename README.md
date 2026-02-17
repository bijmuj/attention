# Attention

Learning to write attention mechanisms in python and (later) CUDA C.

### Notes

- Implementations are being tested against `torch.nn.functional.scaled_dot_product_attention` using the default backend for accuracy.
- Accuracy tests are in `src/python/scratch.ipynb` and are done using mean absolute error (MAE)  over randomly generated test data of size (100, 4, 2048, 768).
- As of Jan 2026, CUDA profiling through pytorch is broken in version 2.9.0 with CUDA 13. [This issue has more info.](https://github.com/pytorch/kineto/pull/1115)

## Running 

- [Install](https://docs.astral.sh/uv/#highlights) `uv` package manager. 
- Clone and navigate to this repository.
- Install pytorch separately:
    ```
    $ pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
    ```
- Install dependencies:
    ```
    $ uv sync
    ```
- Run profiler:
    ```
    $ uv run src/python/profile_mha.py --attn_impl matmul --save_trace --output_dir ./traces --output_file trace.json
    ```
    - This will generate a trace.json file in ./traces/. 
    - To visualize the trace file, open a chromium based browser, navigate to `chrome://tracing` and load the json file.

- For further information on commandline args:
    ```
    $ uv run src/python/profile_mha.py --help
    ```

## Benchmarks

All benchmarks are run using hyperparameters from `./src/config.yml` and timed in python using `./src/python/profile_mha.py`.

Test Specs:
- OS: Fedora 42
- CPU: AMD 7600x
- GPU: RTX 4070
- Driver: 580.119.02 (RPM Fusion)
- CUDA: 12.9
- torch: 2.8.0

| Language | Algorithm | Implementation | Datatype | CPU Time |  CUDA Time | 
| --- | --- | --- | --- | --- | 
| Python | Multi-Head Attention | forward_einsum from ./src/python/attention.py | float32 | 1.522s | 1.474s | 
| Python | Multi-Head Attention | forward_matmul from ./src/python/attention.py | float32 | 1.521s | 1.472s | 
| Python | Multi-Head Attention | torch SDP default backend | float32 | 0.941s | 0.901s | 
| Python | Multi-Head Attention | torch SDP MATH backend | 1.861s | float32 | 1.797s | 
| Python | Multi-Head Attention | torch SDP FLASH_ATTENTION backend | float32 | NA | NA | 
| Python | Multi-Head Attention | torch SDP EFFICIENT_ATTENTION backend | float32 | 0.947s | 0.910s |

## TODO's

- Get torch.compile to work.
- Try writing a kernel for MHA in triton.
- Write alternate kernels in python and C (MQA, GQA, MLA, etc). 