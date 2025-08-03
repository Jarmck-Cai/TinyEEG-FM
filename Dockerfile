FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install mne==1.7.0 datasets webdataset pytorch-lightning==2.2.3 \
               accelerate==0.28.0 peft==0.11.1 evaluate wandb \
               scikit-learn==1.5.0 umap-learn einops

# Copy repository
COPY . .

# Optional: code format / lint
RUN pip install black ruff

# Default command prints help
CMD ["python", "-m", "tiny_eeg_fm.src.train_ssl", "--help"]