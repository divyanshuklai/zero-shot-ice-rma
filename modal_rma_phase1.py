import modal
import os
import sys

# Define the image and include the local codebase directly
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git", "libgl1-mesa-glx", "libosmesa6", "mesa-utils", "patch")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir(
        ".", 
        remote_path="/root", 
        copy=True, 
        ignore=["evals", "runs", ".git", "__pycache__", ".venv", "test_logs"]
    )
)

app = modal.App("rma-phase1")
volume = modal.Volume.from_name("rl-project", create_if_missing=True)

@app.function(
    image=image,
    gpu="L4",
    timeout=3600 * 24,
    volumes={"/rl-project": volume},
    env={
        "PYTHONUNBUFFERED": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "PYTHONPATH": "/root",
    },
)
def run_phase1(
    n_iterations: int, 
    batch_size_total: int, 
    model_name: str = None,
    lr: float = 5e-4,
    dr: bool = True
):
    import subprocess
    os.chdir("/root")
    
    log_dir = "/rl-project/rma"
    os.makedirs(log_dir, exist_ok=True)
    
    args = [
        "python3", "rma_phase1.py",
        "--n-iterations", str(n_iterations),
        "--batch-size-total", str(batch_size_total),
        "--n-envs", "8192",
        "--n-minibatches", "16",
        "--num-evals", "21",
        "--log-dir", log_dir,
        "--learning-rate", str(lr),
        "--dr" if dr else "--no-dr",
    ]
    if model_name:
        args.extend(["--model-name", model_name])
    
    print(f"Starting RMA Phase 1 training...")
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    for line in process.stdout:
        print(line, end="")
    
    process.wait()
    volume.commit()
    
    if process.returncode != 0:
        sys.exit(process.returncode)

@app.local_entrypoint()
def main(n_iterations: int = 5000, batch_size_total: int = 262144, model_name: str = "modal_rma_phase1"):
    run_phase1.remote(
        n_iterations=n_iterations,
        batch_size_total=batch_size_total,
        model_name=model_name
    )
