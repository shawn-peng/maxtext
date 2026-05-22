"""Example of subslicing a JAX array from a 4x4 mesh to a 2x4 subslice.

This script demonstrates two methods for resharing a JAX array from a 4x4 mesh
to a 2x4 subslice, specifically handling a scenario where one of the potential
2x4 subslices might contain a "bad" device.
"""
import contextlib
import time
from typing import Callable, Iterator

import jax
from jax.experimental import mesh_utils
import numpy as np
import pathwaysutils
from pathwaysutils.experimental import split_by_mesh_axis


BAD_DEVICE_ID = 4


@contextlib.contextmanager
def timer(prefix: str) -> Iterator[None]:
  """A context manager that measures the time elapsed within the context."""
  start_time = time.time()
  yield
  end_time = time.time()
  print(prefix)
  print(f"Elapsed time: {end_time - start_time} seconds")


def _get_devices_split_in_two(
    all_devices_list: list[jax.Device],
) -> tuple[list[jax.Device], list[jax.Device]]:
  """Splits devices into two equal sets based on device coords."""
  if len(all_devices_list) % 2 != 0:
    raise ValueError(
        f"Total devices ({len(all_devices_list)}) must be even to split in two."
    )
  device_coord_tuples = [
      (d.coords[0], d.coords[1], d.coords[2], d.core_on_chip)
      for d in all_devices_list
  ]
  device_coords = np.array(device_coord_tuples)
  min_coords = np.min(device_coords, axis=0)
  device_coords -= min_coords
  max_coords = np.max(device_coords, axis=0)
  shape = max_coords + 1
  device_array = np.empty(shape, dtype=object)
  for i, d in enumerate(all_devices_list):
    device_array[tuple(device_coords[i])] = d

  z_dim = 2
  y_dim = 1
  if shape[z_dim] > 1 and shape[z_dim] % 2 == 0:
    split_dim_idx = z_dim
  elif shape[y_dim] > 1 and shape[y_dim] % 2 == 0:
    split_dim_idx = y_dim
  else:
    raise ValueError(
        "Cannot find a dimension to split for host-aligned subslicing."
    )

  mid = shape[split_dim_idx] // 2
  if split_dim_idx == 1:
    devices1 = device_array[:, :mid, :, :].flatten().tolist()
    devices2 = device_array[:, mid:, :, :].flatten().tolist()
  elif split_dim_idx == 2:
    devices1 = device_array[:, :, :mid, :].flatten().tolist()
    devices2 = device_array[:, :, mid:, :].flatten().tolist()
  else:
    raise ValueError(
        f"Unsupported split dimension: {split_dim_idx}. Must be 1 or 2."
    )

  # Devices need to be sorted to maintain canonical order
  key = lambda d: (d.coords[0], d.coords[1], d.coords[2], d.core_on_chip, d.id)
  devices1.sort(key=key)
  devices2.sort(key=key)
  return devices1, devices2


def reshard_to_2x4_via_device_put(x_4x4: jax.Array) -> jax.Array:
  """Reshards an array from a 4x4 mesh to a 2x4 subslice using jax.device_put.

  This function selects one of the two possible 2x4 subslices from the 4x4 mesh
  based on whether `BAD_DEVICE_ID` is present in the first subslice. It then
  uses `jax.device_put` to reshard the input array to the chosen 2x4 subslice.
  Note that `jax.device_put` transfers the array through the controller, which
  can be very slow for large arrays.

  Args:
    x_4x4: The input JAX array sharded on a 4x4 mesh.

  Returns:
    A JAX array sharded on a 2x4 mesh.
  """

  def get_subslice_devices(
      all_devices_list: list[jax.Device], sub_mesh_shape: tuple[int, int]
  ) -> np.ndarray:
    """Selects a host-aligned subslice of devices."""
    devices1, devices2 = _get_devices_split_in_two(all_devices_list)

    # Choose the set that doesn't contain the bad device
    if any(d.id == BAD_DEVICE_ID for d in devices1):
      print(f"Bad device {BAD_DEVICE_ID} found in first set, using second.")
      selected_set = devices2
    else:
      print(f"No bad device {BAD_DEVICE_ID} found in first set, using first.")
      selected_set = devices1

    return np.array(selected_set).reshape(sub_mesh_shape)

  devices = get_subslice_devices(
      list(x_4x4.sharding.mesh.devices.flat), (2, 4)
  )

  mesh_2x4 = jax.sharding.Mesh(
      devices=devices,
      axis_names=("x", "y"),
  )
  sharding_2x4 = jax.sharding.NamedSharding(
      mesh_2x4,
      jax.sharding.PartitionSpec("x", "y"),
  )

  with jax.transfer_guard("log"), timer("Resharding to 2x4 via device_put"):
    x_2x4 = jax.device_put(x_4x4, sharding_2x4)

  return x_2x4


def reshard_to_2x4_via_intermediate_and_device_put(
    x_4x4: jax.Array,
) -> jax.Array:
  """Reshards an array from a 4x4 mesh to a 2x4 subslice via an intermediate sharding.

  This function first reshares the input `x_4x4` to a 2x2x4 mesh, and then
  splits along the first axis to select one of the 2x4 subslices. It checks
  for `BAD_DEVICE_ID` in the first subslice and returns the other if found. This
  approach avoids expensive transfers to the controller.

  Args:
    x_4x4: The input JAX array sharded on a 4x4 mesh.

  Returns:
    A JAX array sharded on a 2x4 mesh, representing one of the subslices.
  """
  all_devices_list = list(x_4x4.sharding.mesh.devices.flat)
  sub_mesh_shape = (2, 4)
  devices1, devices2 = _get_devices_split_in_two(all_devices_list)

  # Construct 2x2x4 mesh
  d1_reshaped = np.array(devices1).reshape(sub_mesh_shape)
  d2_reshaped = np.array(devices2).reshape(sub_mesh_shape)

  # Stack them along a new first axis
  devices_stacked = np.stack([d1_reshaped, d2_reshaped], axis=0)  # (2, 2, 4)

  mesh_2x2x4 = jax.sharding.Mesh(
      devices=devices_stacked,
      axis_names=("y_replica", "x", "y"),
  )
  intermediate_sharding_2x2x4 = jax.sharding.NamedSharding(
      mesh_2x2x4,
      jax.sharding.PartitionSpec("x", "y"),
  )

  with (
      jax.transfer_guard("log"),
      timer("Resharding to 2x4 via intermediate sharding"),
  ):
    x_2x2x4 = jax.device_put(
        x_4x4,
        intermediate_sharding_2x2x4,
    )

    x_2x4, other_x_2x4 = split_by_mesh_axis.split_by_mesh_axis(
        x_2x2x4, "y_replica"
    )

  if any(device.id == BAD_DEVICE_ID for device in x_2x4.sharding.device_set):
    print(f"Bad device {BAD_DEVICE_ID} found in first set, using second.")
    return other_x_2x4
  else:
    print(f"No bad device {BAD_DEVICE_ID} found in first set, using first.")
    return x_2x4


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


def solve(
    x_4x4: jax.Array, get_x_2x4: Callable[[jax.Array], jax.Array]
) -> jax.Array:
  """Solves the equation x**2 + x - 2 = 0 using Newton's method.

  The function first performs a few iterations of Newton's method on the full
  4x4 mesh, then uses the provided `get_x_2x4` function to reshard to a 2x4
  subslice, and continues the iterations on the subslice.

  Args:
    x_4x4: The initial JAX array sharded on a 4x4 mesh.
    get_x_2x4: A function that takes the array on the 4x4 mesh and returns
      a reshared array on a 2x4 subslice.

  Returns:
    A JAX array on a 2x4 mesh containing the solutions.
  """
  # solve (x - 1)*(x + 2) = x**2 + x - 2 = 0
  # Solutions of x = 1, x = -2
  @jax.jit
  def f(x: jax.Array) -> jax.Array:
    return x**2 + x - 2

  @jax.jit
  def f_prime(x: jax.Array) -> jax.Array:
    return 2 * x + 1

  print("Starting solve on 4x4")
  x_4x4 = run_newton_iterations(x_4x4, f, f_prime, num_iterations=3)

  print("Continuing solve on 2x4 subslice")
  x_2x4 = get_x_2x4(x_4x4)

  x_2x4 = run_newton_iterations(x_2x4, f, f_prime, num_iterations=3)

  return x_2x4


def main() -> None:
  pathwaysutils.initialize()

  num_devices = jax.device_count()
  print(f"Found {num_devices} devices.")

  if num_devices < 16:
    print(
        f"Warning: This script is tuned for at least 16 devices. Found {num_devices}"
    )

  devices_4x4 = mesh_utils.create_device_mesh(
      mesh_shape=(4, 4),
      devices=jax.devices()[:16],
  )

  mesh_4x4 = jax.sharding.Mesh(
      devices=devices_4x4,
      axis_names=("x", "y"),
  )
  sharding_4x4 = jax.sharding.NamedSharding(
      mesh_4x4,
      jax.sharding.PartitionSpec("x", "y"),
  )

  x_4x4 = jax.device_put(
      np.hstack((
          np.linspace(-3, -1, 50 * 4),
          np.linspace(0.5, 1.6, 50 * 4),
      )).reshape(100, 4),
      sharding_4x4,
  )

  # Establish the computation origin for device-to-host transfers on the 4x4 mesh.
  # Since x_2x4 will have shape (100, 4) sharded on (2, 4) mesh (shard shape 50x1),
  # we must perform a transfer of shard shape 50x1 on the 4x4 mesh by transferring
  # an array of shape (200, 4).
  print("Initial device-to-host transfers on 4x4 mesh to establish origin...")
  _ = np.array(x_4x4)
  dummy_200x4 = jax.device_put(np.zeros((200, 4), dtype=np.float32), sharding_4x4)
  _ = np.array(dummy_200x4)



  get_x_2x4_dict = {
      "reshard_to_2x4_via_device_put": reshard_to_2x4_via_device_put,
      "reshard_to_2x4_via_intermediate_and_device_put": (
          reshard_to_2x4_via_intermediate_and_device_put
      ),
  }

  for method_name, get_x_2x4 in get_x_2x4_dict.items():
    with timer(f"Solve with {method_name}"):
      x_2x4 = solve(x_4x4, get_x_2x4)
    print(f"Solution to x**2 + x - 2 = 0 is x={np.unique(x_2x4)}")


if __name__ == "__main__":
  main()

