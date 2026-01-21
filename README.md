# Attention

Learning to write attention mechanisms in python and (later) CUDA C.

## Benchmarks

All benchmarks are run using hyperparameters from `./src/config.yml` and timed in python using `./src/python/profile_mha.py`.

Test Specs:
- OS: Fedora 42
- CPU: AMD 7600x
- GPU: RTX 4070
- Driver: 580.119.02 (RPM Fusion)
- CUDA: 12.9
- torch: 2.8.0

| Language | Algorithm | Implementation | CPU Time |  CUDA Time | 
| --- | --- | --- | --- | --- | 
| Python | Multi-Head Attention | forward_einsum from ./src/python/attention.py | 1.522s | 1.474s | 
| Python | Multi-Head Attention | forward_matmul from ./src/python/attention.py | 1.521s | 1.472s | 
| Python | Multi-Head Attention | torch.nn.functional.scaled_dot_product_attention | 0.941s | 0.901s | 

### Note

- Implementations are being tested against `torch.nn.functional.scaled_dot_product_attention` for accuracy.
- Accuracy tests are in `src/python/scratch.ipynb` and are done using mean absolute error (MAE)  over randomly generated test data of size (100, 4, 2048, 768).

## Running 

- [Install](https://docs.astral.sh/uv/#highlights) `uv` package manager. 
- Clone and navigate to this repository.
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

## TODO's

- Get torch.compile to work.
- Try writing a kernel for MHA in triton.
- Write alternate kernels in python and C (MQA, GQA, MLA, etc). 