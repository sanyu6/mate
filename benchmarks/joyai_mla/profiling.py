"""MUSA profiler compatibility for the pinned SGLang benchmark runtime."""


def enable_musa_activity():
    import torch

    original = torch.profiler.profile
    if getattr(original, "_joyai_musa_activity", False):
        return

    def profile(*args, **kwargs):
        activities = kwargs.get("activities")
        if activities is not None:
            activities = [
                a for a in activities if a != torch.profiler.ProfilerActivity.CUDA
            ]
            if torch.profiler.ProfilerActivity.MUSA not in activities:
                activities.append(torch.profiler.ProfilerActivity.MUSA)
            kwargs["activities"] = activities
        return original(*args, **kwargs)

    profile._joyai_musa_activity = True
    torch.profiler.profile = profile
