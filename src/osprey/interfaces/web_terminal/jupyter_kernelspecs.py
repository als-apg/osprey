"""Kernel-spec manager that starts only the kernels the sidecar lists.

``KernelSpecManager.allowed_kernelspecs`` filters what the server LISTS:
:meth:`~jupyter_client.kernelspec.KernelSpecManager.find_kernel_specs` drops
every other name. A start does not go through that filter. The kernel manager
asks :meth:`~jupyter_client.kernelspec.KernelSpecManager.get_kernel_spec` for
the name it was handed, and that method resolves any spec directory on the
kernel path, and the interpreter's own ``python3`` spec from ``ipykernel``,
whatever the allow-list and ``ensure_native_kernel`` say. So a request naming a
kernel the panel never shows would start it.

:class:`AllowListKernelSpecManager` closes that: a name outside the allow-list
is refused with :exc:`~jupyter_client.kernelspec.NoSuchKernel`, the error the
server already answers as "no such kernel". An empty allow-list refuses every
start rather than allowing every one. The listing needs no override: for a
subclass, ``get_all_specs`` resolves each listed name through
``get_kernel_spec``, so the two cannot disagree.

Importing this module must not build the web application: it is imported only
inside the sidecar process, which serves no FastAPI route.
"""

from __future__ import annotations

from jupyter_client.kernelspec import KernelSpec, KernelSpecManager, NoSuchKernel

__all__ = ["AllowListKernelSpecManager"]


class AllowListKernelSpecManager(KernelSpecManager):
    """A kernel-spec manager whose allow-list bounds starts as well as the listing."""

    def get_kernel_spec(self, kernel_name: str) -> KernelSpec:
        """Return the spec for *kernel_name*, refusing a name outside the allow-list.

        Args:
            kernel_name: The kernel name a start or a listing asked for. Matched
                exactly: the base class resolves directories case-insensitively,
                and this check does not.

        Returns:
            The kernel spec, exactly as the base class resolves it.

        Raises:
            NoSuchKernel: *kernel_name* is not in ``allowed_kernelspecs``, or the
                base class finds no such spec.
        """
        if kernel_name not in self.allowed_kernelspecs:
            self.log.warning(
                "Kernel %r refused: the sidecar starts only %s",
                kernel_name,
                sorted(self.allowed_kernelspecs),
            )
            raise NoSuchKernel(kernel_name)
        return super().get_kernel_spec(kernel_name)
