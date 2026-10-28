A tier file that sets `extra_mounts`, `seed_base` or `image` at its top level,
as `osprey init --revalidate` writes them when it retires the render catalog,
builds instead of being refused with "Unknown profile key(s)".
