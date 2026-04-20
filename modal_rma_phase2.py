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

app = modal.App("rma-phase2")
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
def run_phase2(
    teacher_model_name: str,
    n_iterations: int, 
    batch_size_total: int, 
    model_name: str = None,
    lr: float = 5e-4,
    dr: bool = True
):
    import subprocess
    os.chdir("/root")
    
    log_dir = "/rl-project/rma"
    teacher_model_path = f"{log_dir}/models/{teacher_model_name}"
    
    args = [
        "python3", "rma_phase2.py",
        "--teacher-model-path", teacher_model_path,
        "--n-iterations", str(n_iterations),
        "--batch-size-total", str(batch_size_total),
        "--n-envs", "4096",
        "--log-dir", log_dir,
        "--learning-rate", str(lr),
        "--dr" if dr else "--no-dr",
    ]
    if model_name:
        args.extend(["--model-name", model_name])
    
    print(f"Starting RMA Phase 2 training with teacher: {teacher_model_name}")
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    for line in process.stdout:
        print(line, end="")
    
    process.wait()
    volume.commit()
    
    if process.returncode != 0:
        sys.exit(process.returncode)

@app.local_entrypoint()
def main(
    teacher_model: str, 
    n_iterations: int = 2000, 
    batch_size_total: int = 80000, 
    model_name: str = None
):
    """
    To run: modal run modal_rma_phase2.py --teacher-model <name_of_p1_model>
    """
    run_phase2.remote(
        teacher_model_name=teacher_model,
        n_iterations=n_iterations,
        batch_size_total=batch_size_total,
        model_name=model_name
    )
