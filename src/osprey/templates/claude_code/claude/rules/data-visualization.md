# Data Visualization

When generating matplotlib plots, save figures to the osprey-workspace:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(timestamps, values)
ax.set_xlabel("Time")
ax.set_ylabel("Value")
fig.savefig("osprey-workspace/artifacts/plot.png", dpi=150, bbox_inches="tight")
plt.close(fig)
```
