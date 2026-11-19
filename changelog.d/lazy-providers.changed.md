Python executions and Jupyter kernels start seconds sooner, and `osprey build`
and `osprey facility validate` take seconds less CPU: registry initialization
and the `.env.example` render no longer import the built-in provider adapters.
A provider is imported when it is first used, so a registered provider whose
module raises while importing fails that first use, with an error naming the
provider and its module, instead of failing registry initialization; one whose
module or class cannot be found still reads as an unknown provider.
