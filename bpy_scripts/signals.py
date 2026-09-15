"""Signals contract: audio analysis -> per-output-frame arrays.

Computed once before rendering; scenes only index into them, so frame_step
skipping can't change the animation. Index f is 0-based; t[f] is song time.
"""
import bisect
import math


def clamp01(v):
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def smoothstep(v):
    v = clamp01(v)
    return v * v * (3 - 2 * v)


def interp(xs, ys, x):
    if not xs:
        return 0.0
    i = bisect.bisect_right(xs, x)
    if i <= 0:
        return ys[0]
    if i >= len(xs):
        return ys[-1]
    x0, x1 = xs[i - 1], xs[i]
    return ys[i - 1] + (ys[i] - ys[i - 1]) * ((x - x0) / (x1 - x0) if x1 > x0 else 0.0)


def hash01(*keys):
    """Deterministic pseudo-random 0-1 from integer keys (no RNG state)."""
    h = 2166136261
    for k in keys:
        h = ((h ^ (int(k) & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 1274126177) & 0xFFFFFFFF
    return (h & 0xFFFFFF) / float(0x1000000)


class Signals:
    """energy, low, mid, high  smoothed 0-1 levels (fast attack, slower release)
    onset                     0-1 transient envelope (instant attack, short release)
    beat, beat_index, since_beat   beat-grid envelope
    section, section_t, section_elapsed   song structure
    cut, since_cut            camera cut index: new cut at each section change, or
                              on the first beat after MAX_SHOT_S without one."""

    ATTACK_S = 0.03
    RELEASE_S = 0.25
    ONSET_RELEASE_S = 0.12
    BEAT_DECAY_S = 0.15
    MIN_SECTION_S = 2.0
    MAX_SHOT_S = 7.0

    def __init__(self, analysis, frame_count, fps, start_seconds):
        n = frame_count
        dt = 1.0 / fps
        self.n, self.fps = n, fps
        self.t = [start_seconds + f * dt for f in range(n)]

        timeline = analysis.get("energy_timeline") or []
        et = [p["time"] for p in timeline]
        ev = [p["energy"] for p in timeline]
        self.energy = self._level([interp(et, ev, t) for t in self.t], self.RELEASE_S)

        bands = analysis.get("bands")
        if bands:
            hop = bands["hop_s"]

            def sample(key):
                arr = bands[key]
                out = []
                for t in self.t:
                    x = t / hop
                    i = int(x)
                    if i >= len(arr) - 1:
                        out.append(arr[-1] if arr else 0.0)
                    else:
                        out.append(arr[i] + (arr[i + 1] - arr[i]) * (x - i))
                return out

            self.low = self._level(sample("low"), self.RELEASE_S)
            self.mid = self._level(sample("mid"), self.RELEASE_S * 0.6)
            self.high = self._level(sample("high"), self.RELEASE_S * 0.4)
            self.onset = self._level(sample("onset"), self.ONSET_RELEASE_S, attack_s=0.0)
            self.brightness = self._level(sample("centroid"), self.RELEASE_S) if "centroid" in bands else self.mid
        else:
            # Older analysis JSON without bands: every band follows broadband energy.
            self.low = self.mid = self.high = self.brightness = self.energy
            self.onset = None

        duration = float(analysis.get("duration") or (self.t[-1] if self.t else 0.0))
        beats = self._complete_beat_grid(sorted(analysis.get("beat_times") or []), duration)
        self.beat, self.beat_index, self.since_beat = [], [], []
        for t in self.t:
            i = bisect.bisect_right(beats, t) - 1
            if i < 0:
                self.beat.append(0.0)
                self.beat_index.append(-1)
                self.since_beat.append(1e9)
            else:
                since = t - beats[i]
                self.beat.append(math.exp(-since / self.BEAT_DECAY_S))
                self.beat_index.append(i)
                self.since_beat.append(since)
        if self.onset is None:
            self.onset = self.beat

        # librosa's agglomerative segmentation emits near-duplicate boundaries
        # (e.g. 18.62, 18.72) -- merge them or every section-driven move restarts
        # several times in a fraction of a second.
        bounds = [0.0]
        for b in sorted(analysis.get("section_times") or []):
            if b - bounds[-1] >= self.MIN_SECTION_S:
                bounds.append(b)
        bounds.append(max(duration, self.t[-1] + dt if n else 0.0))
        self.section, self.section_t, self.section_elapsed = [], [], []
        for t in self.t:
            i = min(max(bisect.bisect_right(bounds, t) - 1, 0), len(bounds) - 2)
            lo, hi = bounds[i], bounds[i + 1]
            self.section.append(i)
            self.section_t.append(clamp01((t - lo) / (hi - lo)) if hi > lo else 0.0)
            self.section_elapsed.append(t - lo)

        self.cut, self.since_cut = [], []
        cut, last_cut_t, prev_section, prev_beat = 0, self.t[0] if n else 0.0, None, None
        for f, t in enumerate(self.t):
            new_beat = self.beat_index[f] != prev_beat and self.beat_index[f] >= 0
            if prev_section is not None and self.section[f] != prev_section:
                cut, last_cut_t = cut + 1, t
            elif t - last_cut_t >= self.MAX_SHOT_S and new_beat:
                cut, last_cut_t = cut + 1, t
            prev_section, prev_beat = self.section[f], self.beat_index[f]
            self.cut.append(cut)
            self.since_cut.append(t - last_cut_t)

    @staticmethod
    def _complete_beat_grid(beats, duration):
        """librosa's beat tracker often only locks on after a sparse intro (a live
        30 s song at 129 bpm came back with 29 beats, all in the second half), which
        leaves every beat-driven element dead until then. Extend the detected grid at
        its median period back to 0, forward to the end, and through gaps."""
        if len(beats) < 4:
            return beats
        diffs = sorted(b - a for a, b in zip(beats, beats[1:]))
        period = diffs[len(diffs) // 2]
        if period <= 0.2:
            return beats
        filled = []
        t = beats[0]
        while t - period >= 0.0:
            t -= period
        while t < beats[0] - period * 0.5:
            filled.append(round(t, 3))
            t += period
        for a, b in zip(beats, beats[1:]):
            filled.append(a)
            gap = b - a
            if gap > period * 1.75:
                steps = round(gap / period)
                filled.extend(round(a + gap * i / steps, 3) for i in range(1, steps))
        filled.append(beats[-1])
        t = beats[-1] + period
        while t < duration:
            filled.append(round(t, 3))
            t += period
        return filled

    def _level(self, raw, release_s, attack_s=None):
        """Normalise against the rendered window (so a quiet excerpt still gets
        full range), then smooth with separate attack/release."""
        n = len(raw)
        if not n:
            return []
        ref = sorted(raw)[int(0.95 * (n - 1))]
        ref = ref if ref > 1e-6 else 1.0
        dt = 1.0 / self.fps
        attack_s = self.ATTACK_S if attack_s is None else attack_s
        k_att = 1.0 if attack_s <= 0 else 1 - math.exp(-dt / attack_s)
        k_rel = 1 - math.exp(-dt / release_s)
        out, s = [], 0.0
        for v in raw:
            v = min(1.0, v / ref)
            s += (v - s) * (k_att if v > s else k_rel)
            out.append(s)
        return out

    def at(self, arr, f):
        return arr[min(max(f, 0), self.n - 1)]

    def cumulative(self, rate_fn):
        """Integrate a per-frame rate (units/second) so speed can follow the
        music without position jumping back when the level drops."""
        out, acc = [], 0.0
        for f in range(self.n):
            out.append(acc)
            acc += rate_fn(f) / self.fps
        return out
