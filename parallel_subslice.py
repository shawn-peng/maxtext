"""Example of subslicing a JAX array from a 4x4 mesh to a 2x4 subslice.

This script demonstrates two methods for resharing a JAX array from a 4x4 mesh
to a 2x4 subslice, specifically handling a scenario where one of the potential
2x4 subslices might contain a "bad" device.
"""
import concurrent.futures
import contextlib
import functools
import os
import time
from typing import Callable, Iterator

import jax
from jax.experimental import mesh_utils
import numpy as np
import pathwaysutils
from pathwaysutils.experimental import split_by_mesh_axis


BAD_DEVICE_ID = 4


# Single host physical coordinate shape map per TPU generation / device kind
TPU_GENERATION_HOST_SHAPES = {
    "TPU v4": (2, 2),
    "TPU v5 lite": (2, 4),
    "TPU v5p": (2, 2),
    "TPU v6 lite": (2, 4),
    "cpu": (2, 2),
}


@jax.jit
def f(x: jax.Array) -> jax.Array:
  return x**2 + x - 2


@jax.jit
def f_prime(x: jax.Array) -> jax.Array:
  return 2 * x + 1


@contextlib.contextmanager
def timer(prefix: str) -> Iterator[None]:
  """A context manager that measures the time elapsed within the context."""
  start_time = time.time()
  yield
  end_time = time.time()
  print(prefix)
  print(f"Elapsed time: {end_time - start_time} seconds")


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def run_newton_iterations(
    x: jax.Array,
    f: Callable[[jax.Array], jax.Array],
    f_prime: Callable[[jax.Array], jax.Array],
    num_iterations: int,
) -> jax.Array:
  """Runs Newton's method iterations.

  Args:
    x: The initial value.
    f: The function for which to find roots.
    f_prime: The derivative of f.
    num_iterations: The number of iterations to run.

  Returns:
    The value of x after the iterations.
  """
  for _ in range(num_iterations):
    x = x - f(x) / f_prime(x)
  return x


def create_subslices(
    devices_orig: list,
    host_mesh_shape: tuple[int, ...],
) -> np.ndarray:
  """Creates a multidimensional grid of host-aligned subslice device meshes.

  Args:
    devices_orig: A coordinate-assigned parent device array of shape orig_mesh_shape or a list of devices.
    host_mesh_shape: The shape of each host subslice mesh.

  Returns:
    A multidimensional numpy array of shape grid_shape (dtype=object)
    where each element is a numpy array of devices representing a subslice.
  """
  # devices_orig = np.asarray(devices_orig)

  # create a slice (ndarray) based on device coords
  shape = np.array(devices_orig[-1].coords[::-1]) + 1
  print('shape:', shape)

  orig_slice = np.ndarray(shape, dtype=object)
  print('orig_slice:', orig_slice)
  print('orig_slice_shape:', orig_slice.shape)
  for d in devices_orig:
    print('d.coords:', d.coords)
    orig_slice[tuple(d.coords[::-1])] = d

  print(orig_slice)
  print('orig_slice.shape:', orig_slice.shape)

  padded_host_mesh_shape = np.pad(
      host_mesh_shape[::-1],
      (0, max(0, len(orig_slice.shape) - len(host_mesh_shape))),
      mode='constant',
      constant_values=1,
  )[::-1]
  print('padded_host_mesh_shape:', padded_host_mesh_shape)

  subslices_shape = np.array(orig_slice.shape) // padded_host_mesh_shape
  print('subslices_shape:', subslices_shape)

  subslices = np.ndarray(subslices_shape, dtype=object)
  for idx in np.ndindex(*subslices_shape):
    print("creating subslice", idx)
    lower = np.array(idx) * padded_host_mesh_shape
    upper = (np.array(idx) + 1) * padded_host_mesh_shape
    subslice_indices = tuple(
        slice(lower[d], upper[d])
        for d in range(len(idx))
    )
    print("subslice_indices", subslice_indices)
    subslice_devices = orig_slice[subslice_indices].flatten()
    subslices[idx] = mesh_utils.create_device_mesh(
        mesh_shape=host_mesh_shape,
        devices=subslice_devices,
        allow_split_physical_axes=False,
    )
    print("created subslice %s" % (idx,))
    print("created subslice %s" % (subslices[idx],))
    print("created subslice shape: %s" % (subslices[idx].shape,))

  return subslices


def solve(
    x_subs: list[jax.Array] | np.ndarray | jax.Array,
    subslices: np.ndarray[jax.sharding.Mesh],
    executor: concurrent.futures.ThreadPoolExecutor,
) -> list[jax.Array]:
  """Solves the equation x**2 + x - 2 = 0 using Newton's method in parallel subslices.

  The function continues the iterations on all subslices in parallel.

  Args:
    x_subs: The JAX arrays sharded on the subslices.
    subslice_mesh: The subslice mesh to use for execution.
    executor: The ThreadPoolExecutor to run the parallel tasks.

  Returns:
    A list of JAX arrays containing the solutions on the subslices.
  """
  # Standardize to a flat list of non-None JAX arrays for parallel execution

  def solve_single_subslice(sub_x: jax.Array, idx: tuple[int]) -> jax.Array:
    print(f"[Subslice {idx}] Starting Newton iterations...")
    with jax.set_mesh(sub_x.sharding.mesh):
      print(f"Running subslice {idx}")
      sub_x = run_newton_iterations(sub_x, f, f_prime, num_iterations=10)
      print(f"Finished subslice {idx}, x: {np.unique(jax.device_get(sub_x))}")

    # Iterate over the shards of the subslice array local to this process
    for shard in sub_x.addressable_shards:
      shard_host_data = jax.device_get(shard.data)
      print(f"[Subslice {idx}] Retrieved shard at {shard.index} with shape {shard_host_data.shape}")
    
    return sub_x
  
  x_subs_sharded = np.ndarray(subslices.shape, dtype=object)
  for idx in np.ndindex(*subslices.shape):
    sub_mesh = jax.sharding.Mesh(subslices[idx], axis_names=('x', 'y'))
    sharding = jax.sharding.NamedSharding(sub_mesh, jax.sharding.PartitionSpec(None, 'x', 'y'))
    x_subs_sharded[idx] = jax.device_put(x_subs[idx], sharding)

  # Run all subslices concurrently using the provided ThreadPoolExecutor
  futures = [
      executor.submit(solve_single_subslice, x_subs_sharded[idx], idx)
      for idx in np.ndindex(*subslices.shape)
  ]
  results = [f.result() for f in futures]

  return results


def main() -> None:
  if not (os.environ.get("JAX_PLATFORMS") == "cpu"):
    pathwaysutils.initialize()

  import argparse
  parser = argparse.ArgumentParser(description="Parallel subslice orchestration.")
  parser.add_argument(
      "--topology",
      type=str,
      required=True,
      help="Topology shape as string (e.g., '4x4' or '4x4x2' or '2x2x4').",
  )
  args, _ = parser.parse_known_args()

  num_devices = jax.device_count()
  print(f"Found {num_devices} devices.")

  delim = ',' if ',' in args.topology else 'x'
  orig_mesh_shape = tuple(int(x) for x in args.topology.split(delim))
  print(f"Using logical slice topology: {orig_mesh_shape}")

  print(f"Dynamically creating original mesh with shape {orig_mesh_shape} from {num_devices} devices.")

  devices = jax.devices()[:num_devices]
  devices_orig = mesh_utils.create_device_mesh(
      mesh_shape=orig_mesh_shape,
      devices=devices,
      allow_split_physical_axes=False,
  )
  print('devices_orig: ', devices_orig)
  print('devices_orig.shape: ', devices_orig.shape)

  host_mesh_shape = TPU_GENERATION_HOST_SHAPES[jax.devices()[0].device_kind]

  subslices = create_subslices(devices, host_mesh_shape)
  print('subslices: ', subslices)

  # return

  print(f"Successfully created {len(subslices)} host-aligned subslices of shape {host_mesh_shape}.")

  print(subslices)
  print(subslices.shape)

  # return


  # Map dimensions to axis names dynamically (e.g., x, y, z, ...)
  standard_axis_names = ("x", "y", "z", "w")
  orig_axis_names = standard_axis_names[:len(orig_mesh_shape)]

  mesh_orig = jax.sharding.Mesh(
      devices=devices_orig,
      axis_names=orig_axis_names,
  )
  print("Original mesh:", mesh_orig)
  sharding_orig = jax.sharding.NamedSharding(
      mesh_orig,
      jax.sharding.PartitionSpec(*orig_axis_names),
  )
  print("Original sharding:", sharding_orig)

  # Create array where last dimensions match mesh shape for partitioning spec
  array_shape = (100,) + orig_mesh_shape[1:]
  total_elements = int(100 * np.prod(orig_mesh_shape[1:]))
  x = np.hstack((
          np.linspace(-3, -1, total_elements // 2),
          np.linspace(0.5, 1.6, total_elements // 2),
      )).reshape(array_shape)
  print("Original X:", x)
  print("Original array shape:", x.shape)
  x_full_shard = jax.device_put(x, sharding_orig)
  x = run_newton_iterations(x_full_shard, f, f_prime, num_iterations=10)
  print("Newton solved array (on full mesh): ", np.unique(jax.device_get(x)))

  x_shards = x.reshape(*subslices.shape, -1, *host_mesh_shape) # + (slice_shape,)
  print("X shards shape:", x_shards.shape)
  # print(x_shards)

  with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, np.prod(subslices.shape))) as executor:
  # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    with timer("Solve with Newton's Method"):
      ans = solve(x_shards, subslices, executor)
  print("Final Solution is ", np.unique(ans))

  if jax.distributed.is_initialized():
    jax.distributed.shutdown()


if __name__ == "__main__":
  main()

