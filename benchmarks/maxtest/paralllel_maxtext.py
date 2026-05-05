import concurrent.futures
import pathwaysutils
import jax
import numpy as np
from jax.sharding import AxisType, Mesh
# Import MaxText training logic
from maxtext.trainers.pre_train import train

pathwaysutils.initialize()

def run_subslice(subslice_coord):
  print(f"Starting MaxText job on subslice coordinate: {subslice_coord}...")

  # Configure maxtext parameters programmatically for this subslice
  config_args = [
    "base_output_directory=gs://pengyisu-tpu-testing/maxtext_output",
    "per_device_batch_size=1",
    "enable_checkpointing=false",
    "dataset_type=synthetic",
    "enable_single_controller=True",
    "steps=100",
    "subslice_shape=2,2,1",  # Partition mesh to 2x2x1 (16 chips)
    f"subslice_coord={subslice_coord}",
    f"run_name=maxtext-subslice-{subslice_coord}"
  ]

  # Run the training loop for this subslice
  train.main(config_args)

if __name__ == "__main__":
  # JAX + Pathways initialization (single connection)
  # Ensure JAX_PLATFORMS=proxy and JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 are exported in your shell

  print(f"Connected to Pathways. Total devices available: {len(jax.devices())}")

  # Run Job 0 (coord 0,0,0) and Job 1 (coord 0,0,1) concurrently in parallel threads
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    futures = [
      executor.submit(run_subslice, "0,0,0"),
      executor.submit(run_subslice, "0,0,1")
    ]
    concurrent.futures.wait(futures)

  print("Both parallel MaxText jobs completed successfully!")
