"""Look definitions shared by both renderers (no bpy / GL imports).

Palette and lighting (used by the Blender look builder and the GL renderer):
  ramp            drive -> colour stops
  accent          colour for accent=1 elements (scaled by drive)
  strength        drive -> emission strength (low end near zero so calm
                  elements read as lit surfaces)
  horizon/zenith  world gradient
  fog             (near, far, max) distance fade toward the horizon colour
  key/rim         (energy, colour) sun from front-above and from behind

Post (GL renderer only -- Blender has no post chain):
  bloom           (intensity, threshold)
  rays            light-shaft strength from the scene's hero point
  dof             depth-of-field strength (0 = off)
  grain, vignette, aberration (chromatic, fraction of the frame)
  grade           (saturation, contrast, exposure)
"""

LOOKS = {
    "neon_night": {
        "ramp": [(0.0, (0.01, 0.015, 0.2)), (0.5, (0.03, 0.25, 1.0)), (1.0, (1.0, 0.3, 0.04))],
        "accent": (1.0, 0.04, 0.45), "strength": (0.03, 4.5),
        "horizon": (0.0090, 0.0036, 0.0210), "zenith": (0.0, 0.0, 0.004),
        "fog": (18.0, 90.0, 0.85), "key": (1.2, (0.7, 0.8, 1.0)), "rim": (3.0, (1.0, 0.25, 0.6)),
        "post": {"bloom": (1.0, 1.0), "rays": 0.35, "dof": 1.0, "grain": 0.035, "vignette": 0.9,
                 "aberration": 0.0015, "grade": (1.1, 1.05, 1.0)},
    },
    "ember": {
        "ramp": [(0.0, (0.04, 0.008, 0.004)), (0.45, (0.6, 0.07, 0.01)), (1.0, (1.0, 0.8, 0.35))],
        "accent": (0.08, 0.55, 0.7), "strength": (0.02, 5.0),
        "horizon": (0.0270, 0.0075, 0.0018), "zenith": (0.002, 0.001, 0.001),
        "fog": (15.0, 80.0, 0.9), "key": (1.5, (1.0, 0.75, 0.5)), "rim": (2.5, (1.0, 0.45, 0.1)),
        "post": {"bloom": (1.2, 0.9), "rays": 0.5, "dof": 1.2, "grain": 0.05, "vignette": 1.0,
                 "aberration": 0.001, "grade": (1.05, 1.1, 1.0)},
    },
    "ice": {
        "ramp": [(0.0, (0.01, 0.03, 0.08)), (0.5, (0.1, 0.6, 0.9)), (1.0, (0.9, 0.97, 1.0))],
        "accent": (0.55, 0.3, 1.0), "strength": (0.02, 3.5),
        "horizon": (0.0090, 0.0210, 0.0330), "zenith": (0.001, 0.003, 0.01),
        "fog": (20.0, 100.0, 0.9), "key": (2.5, (0.85, 0.93, 1.0)), "rim": (2.0, (0.4, 0.8, 1.0)),
        "post": {"bloom": (0.8, 1.1), "rays": 0.25, "dof": 0.8, "grain": 0.025, "vignette": 0.7,
                 "aberration": 0.002, "grade": (0.95, 1.0, 1.05)},
    },
    "acid": {
        "ramp": [(0.0, (0.005, 0.03, 0.005)), (0.5, (0.25, 0.9, 0.05)), (1.0, (1.0, 1.0, 0.2))],
        "accent": (1.0, 0.0, 0.55), "strength": (0.03, 4.0),
        "horizon": (0.0045, 0.0120, 0.0015), "zenith": (0.0, 0.002, 0.0),
        "fog": (15.0, 80.0, 0.85), "key": (1.0, (0.9, 1.0, 0.8)), "rim": (3.0, (0.6, 1.0, 0.1)),
        "post": {"bloom": (1.3, 0.9), "rays": 0.3, "dof": 1.0, "grain": 0.04, "vignette": 0.9,
                 "aberration": 0.003, "grade": (1.25, 1.1, 1.0)},
    },
    "mono_red": {
        "ramp": [(0.0, (0.01, 0.01, 0.01)), (0.6, (0.35, 0.35, 0.35)), (1.0, (1.0, 1.0, 1.0))],
        "accent": (1.0, 0.02, 0.02), "strength": (0.0, 2.5),
        "horizon": (0.0120, 0.0120, 0.0135), "zenith": (0.002, 0.002, 0.002),
        "fog": (20.0, 100.0, 0.9), "key": (3.0, (1.0, 1.0, 1.0)), "rim": (2.0, (1.0, 1.0, 1.0)),
        "post": {"bloom": (0.6, 1.2), "rays": 0.15, "dof": 1.0, "grain": 0.06, "vignette": 1.1,
                 "aberration": 0.0, "grade": (1.0, 1.2, 1.0)},
    },
    "sunset": {
        "ramp": [(0.0, (0.06, 0.01, 0.1)), (0.5, (0.9, 0.12, 0.4)), (1.0, (1.0, 0.75, 0.2))],
        "accent": (0.0, 0.7, 1.0), "strength": (0.02, 4.0),
        "horizon": (0.1600, 0.0360, 0.0520), "zenith": (0.015, 0.008, 0.06),
        "fog": (15.0, 110.0, 0.95), "key": (1.8, (1.0, 0.6, 0.45)), "rim": (3.0, (1.0, 0.4, 0.2)),
        "post": {"bloom": (1.1, 1.0), "rays": 0.6, "dof": 1.0, "grain": 0.04, "vignette": 0.8,
                 "aberration": 0.0015, "grade": (1.15, 1.0, 1.0)},
    },
}
