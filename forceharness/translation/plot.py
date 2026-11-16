"""The "plot" force form (an ablation): force / torque versus time rendered as a PNG for a multimodal model."""
from __future__ import annotations

import io
from typing import List

import numpy as np

from forceharness.primitives.report import ContactReport
from forceharness.translation.base import Injector, Part


def render_wrench_png(rep: ContactReport, size=(640, 420), dpi=100) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = rep.wrench_array()
    fig, axes = plt.subplots(2, 1, figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi, sharex=True)
    if len(arr):
        t = arr[:, 0]
        for i, (lab, c) in enumerate(zip(["Fx", "Fy", "Fz"], ["tab:red", "tab:green", "tab:blue"])):
            axes[0].plot(t, arr[:, 1 + i], label=lab, color=c, lw=1.5)
        axes[0].plot(t, np.linalg.norm(arr[:, 1:4], axis=1), label="|F|", color="k", lw=1.0, ls="--")
        for i, (lab, c) in enumerate(zip(["Tx", "Ty", "Tz"], ["tab:red", "tab:green", "tab:blue"])):
            axes[1].plot(t, arr[:, 4 + i], label=lab, color=c, lw=1.5)
        for e in rep.events:
            for ax in axes:
                ax.axvline(e.t, color="gray", lw=0.8, ls=":")
            axes[0].text(e.t, axes[0].get_ylim()[1] if len(arr) else 0, e.kind, fontsize=7, rotation=90, va="top", ha="right")
    axes[0].set_ylabel("force (N)")
    axes[1].set_ylabel("torque (Nm)")
    axes[1].set_xlabel("time since primitive start (s)")
    axes[0].set_title(f"wrist wrench during '{rep.primitive}' (world frame, bias removed)", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right", ncol=4)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


class PlotInjector(Injector):
    name = "plot"

    def __init__(self, *a, **k):
        pass

    def render_force(self, rep: ContactReport, state) -> List[Part]:
        png = render_wrench_png(rep)
        return [{"type": "text", "text": "wrist force/torque during this primitive is shown in the attached plot (world frame, N / Nm, "
                                         "bias removed; dotted vertical lines mark detected events)."},
                {"type": "image", "png": png, "label": f"wrench_plot_{rep.primitive}"}]

    def describe(self):
        return "After each primitive you receive a PLOT (image) of the wrist force and torque versus time."
