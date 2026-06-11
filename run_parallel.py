# run_parallel.py - JAX-native Multi-Mesh Parallel Orchestration for MaxText
import concurrent.futures
import sys
import os
import pathwaysutils
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import partitioning as nn_partitioning
from maxtext.configs import pyconfig
from maxtext.utils import train_utils
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding
from maxtext.trainers.pre_train import train as train_lib

def run_step(step, p_train_step, state, data_loader, rampup_manager, init_rng, mesh, config, state_mesh_shardings):
  example_batch = data_loader.load_next_batch(rampup_manager=rampup_manager)
  nextrng = jax.jit(jax.random.fold_in)(init_rng, step)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
    if config.shard_optimizer_over_data:
      state = sharding.maybe_shard_with_name(state, state_mesh_shardings, config.shard_mode)
    state, metrics = p_train_step(state, example_batch, nextrng)
  return state, metrics

def execute_training(thread_id, config, devices):
  print(f"[Thread {thread_id}] Initializing training loop on devices: {devices}...")
  
  # Initialize recorder
  recorder = train_lib.create_goodput_recorder(config)
  
  # Setup train loop restricted to this thread's devices
  (
      init_rng,
      checkpoint_manager,
      state_mesh_shardings,
      model,
      mesh,
      learning_rate_schedule,
      data_iterator,
      data_loader,
      rampup_manager,
      eval_data_iterator,
      state,
  ) = train_utils.setup_train_loop(config, recorder, devices=devices)
  
  print(f"[Thread {thread_id}] Mesh initialized successfully: {mesh}")

  params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)

  # JIT compilation on this mesh
  with jax.set_mesh(mesh), mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    p_train_step, p_eval_step = train_utils.jit_train_and_eval_step(
        config,
        model,
        mesh,
        state,
        state_mesh_shardings,
        train_lib.train_step,
        train_lib.eval_step,
        eval_data_iterator,
        params_shardings,
    )
    
  return {
      "p_train_step": p_train_step,
      "state": state,
      "data_loader": data_loader,
      "rampup_manager": rampup_manager,
      "init_rng": init_rng,
      "mesh": mesh,
      "config": config,
      "state_mesh_shardings": state_mesh_shardings
  }

def main():
  # 1. Initialize pathwaysutils first to register the proxy backend dynamically
  print("Initializing Pathways backend...")
  pathwaysutils.initialize()

  # 2. Parse global config once from argv (JAX proxy variables exported)
  print("Parsing configuration...")
  argv = sys.argv
  # Ensure we have synthetic dataset configured by default
  if "dataset_type=synthetic" not in argv:
    argv.append("dataset_type=synthetic")
  if "enable_checkpointing=false" not in argv:
    argv.append("enable_checkpointing=false")
  if "per_device_batch_size=1" not in argv:
    argv.append("per_device_batch_size=1")
  
  base_config = pyconfig.initialize(argv)
  
  # 3. Get all 32 TPU devices and split into two 16-chip sublists
  all_devices = jax.devices()
  print(f"Connected to Pathways. Total devices: {len(all_devices)}")
  if len(all_devices) < 32:
    print(f"Warning: Expected 32 devices for v4-32, but got {len(all_devices)}. Splitting available devices in half.")
  
  half = len(all_devices) // 2
  devices_0 = all_devices[:half]
  devices_1 = all_devices[half:]
  
  # 4. Clone base config and customize run_names to isolate them
  import copy
  config_0 = copy.deepcopy(base_config)
  config_0.run_name = f"{base_config.run_name}-subslice-0"
  
  config_1 = copy.deepcopy(base_config)
  config_1.run_name = f"{base_config.run_name}-subslice-1"
  config_1.compiled_trainstep_file = "" # Disable load compiled to avoid collisions
  
  # 5. Initialize both training loops in parallel threads
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    future_0 = executor.submit(execute_training, 0, config_0, devices_0)
    future_1 = executor.submit(execute_training, 1, config_1, devices_1)
    
    job_0 = future_0.result()
    job_1 = future_1.result()

  print("\nBoth parallel models initialized successfully! Starting concurrent parallel training steps...")

  # 6. Run concurrent training steps in a loop
  steps = base_config.steps
  for step in range(steps):
    # Launch step 0 and step 1 concurrently in the background
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
      f0 = executor.submit(
          run_step, step, job_0["p_train_step"], job_0["state"], job_0["data_loader"],
          job_0["rampup_manager"], job_0["init_rng"], job_0["mesh"], job_0["config"], job_0["state_mesh_shardings"]
      )
      f1 = executor.submit(
          run_step, step, job_1["p_train_step"], job_1["state"], job_1["data_loader"],
          job_1["rampup_manager"], job_1["init_rng"], job_1["mesh"], job_1["config"], job_1["state_mesh_shardings"]
      )
      
      job_0["state"], metrics_0 = f0.result()
      job_1["state"], metrics_1 = f1.result()
    
    # Wait for both steps to complete concurrently on the hardware
    jax.block_until_ready((job_0["state"], job_1["state"]))
    print(f"Step {step}/{steps} completed in parallel: Job 0 Loss={metrics_0['scalar']['learning/loss']:.4f} | Job 1 Loss={metrics_1['scalar']['learning/loss']:.4f}")

  print("\nSuccessfully completed all parallel training steps on both subslices concurrently!")

if __name__ == "__main__":
  main()
