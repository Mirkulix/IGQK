"""
IGQK Production REST API.

Endpoints:
    POST /api/v1/compress     - Compress a model
    POST /api/v1/train        - Start training job
    GET  /api/v1/jobs/{id}    - Get job status
    POST /api/v1/evaluate     - Evaluate a model
    GET  /api/v1/health       - Health check
    GET  /api/v1/info         - System info
"""

import os
import uuid
import time
import asyncio
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import igqk

app = FastAPI(
    title="IGQK API",
    description="Information-Geometric Quantum Compression REST API",
    version=igqk.__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (use Redis/DB in production)
_jobs: dict = {}

UPLOAD_DIR = os.environ.get("IGQK_UPLOAD_DIR", "/tmp/igqk_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --- Models ---

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    gpu_available: bool


class InfoResponse(BaseModel):
    version: str
    pytorch_version: str
    cuda_available: bool
    gpu_name: Optional[str] = None
    compression_methods: list
    supported_datasets: list


class CompressRequest(BaseModel):
    method: str = "ternary"
    rank: int = 10
    sparsity: float = 0.1
    keep_ratio: float = 0.5


class TrainRequest(BaseModel):
    model: str = "simple_fc"
    dataset: str = "mnist"
    compression: str = "ternary"
    epochs: int = 10
    hbar: float = 0.1
    gamma: float = 0.01
    batch_size: int = 64


class JobResponse(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    result: Optional[dict] = None
    created_at: str
    error: Optional[str] = None


# --- Endpoints ---

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    import torch

    return HealthResponse(
        status="healthy",
        version=igqk.__version__,
        timestamp=datetime.utcnow().isoformat(),
        gpu_available=torch.cuda.is_available(),
    )


@app.get("/api/v1/info", response_model=InfoResponse)
async def system_info():
    import torch

    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    return InfoResponse(
        version=igqk.__version__,
        pytorch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        gpu_name=gpu_name,
        compression_methods=["ternary", "lowrank", "sparse", "wavelet"],
        supported_datasets=["mnist", "cifar10"],
    )


@app.post("/api/v1/compress")
async def compress_model(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    method: str = Form("ternary"),
    rank: int = Form(10),
    sparsity: float = Form(0.1),
):
    """Upload a PyTorch checkpoint and compress it."""
    job_id = str(uuid.uuid4())[:8]
    filepath = os.path.join(UPLOAD_DIR, f"{job_id}_input.pt")

    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)

    _jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "created_at": datetime.utcnow().isoformat(),
        "result": None,
        "error": None,
    }

    background_tasks.add_task(_run_compression, job_id, filepath, method, rank, sparsity)

    return {"job_id": job_id, "status": "queued"}


@app.post("/api/v1/train")
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """Start an IGQK training job."""
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "created_at": datetime.utcnow().isoformat(),
        "result": None,
        "error": None,
    }

    background_tasks.add_task(_run_training, job_id, request)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    return JobResponse(job_id=job_id, **job)


@app.get("/api/v1/jobs/{job_id}/download")
async def download_result(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")

    output_path = job["result"].get("output_path")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(output_path, filename=os.path.basename(output_path))


# --- Background Tasks ---

def _run_compression(job_id: str, filepath: str, method: str, rank: int, sparsity: float):
    import torch
    from igqk.theory.tlgt import TernaryLieGroup
    from igqk.theory.hlwt import HybridLaplaceWavelet
    from igqk.compression.projection import OptimalProjection

    try:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["progress"] = 0.1

        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        compressed_state = {}
        keys = list(state_dict.keys())

        for i, (name, param) in enumerate(state_dict.items()):
            if method == "ternary":
                tlgt = TernaryLieGroup(param.numel())
                compressed, scale = tlgt.quantize(param)
                compressed_state[name] = compressed
            elif method == "wavelet":
                hlwt = HybridLaplaceWavelet()
                compressed, ratio = hlwt.compress(param, keep_ratio=0.3)
                compressed_state[name] = compressed
            elif method == "lowrank":
                proj = OptimalProjection(submanifold_type="lowrank", rank=rank)
                compressed_state[name] = proj.projector.project(param.flatten()).view_as(param)
            elif method == "sparse":
                proj = OptimalProjection(submanifold_type="sparse", sparsity=sparsity)
                compressed_state[name] = proj.projector.project(param.flatten()).view_as(param)

            _jobs[job_id]["progress"] = 0.1 + 0.8 * (i + 1) / len(keys)

        output_path = os.path.join(UPLOAD_DIR, f"{job_id}_compressed.pt")
        torch.save({"model_state_dict": compressed_state, "method": method}, output_path)

        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["progress"] = 1.0
        _jobs[job_id]["result"] = {"output_path": output_path, "method": method}

    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)


def _run_training(job_id: str, request: TrainRequest):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from igqk.integration.pytorch import IGQKTrainer

    try:
        _jobs[job_id]["status"] = "running"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if request.dataset == "mnist":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
            val_ds = datasets.MNIST("./data", train=False, transform=transform)
            in_features, num_classes = 784, 10
        else:
            raise ValueError(f"Unsupported dataset: {request.dataset}")

        train_loader = DataLoader(train_ds, batch_size=request.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=request.batch_size)

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

        trainer = IGQKTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            hbar=request.hbar,
            gamma=request.gamma,
            compression_type=request.compression,
            device=device,
        )

        def progress_callback(epoch, metrics):
            _jobs[job_id]["progress"] = (epoch + 1) / request.epochs

        trainer.train(num_epochs=request.epochs, callback=progress_callback)
        compressed_model = trainer.compress()

        output_path = os.path.join(UPLOAD_DIR, f"{job_id}_trained.pt")
        torch.save({
            "model_state_dict": compressed_model.state_dict(),
            "history": trainer.history,
            "compression_type": request.compression,
        }, output_path)

        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["progress"] = 1.0
        _jobs[job_id]["result"] = {
            "output_path": output_path,
            "final_loss": trainer.history["train_loss"][-1] if trainer.history["train_loss"] else None,
            "history": trainer.history,
        }

    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)
