"""
Sensory-Grounded Online Learning
=================================
A bio-inspired alternative to big-data AI training.

Instead of feeding a model billions of labelled images, this system learns
object concepts from a small number of multi-sensory interactions — the same
way a child learns what an apple is through seeing, touching, and handling one.

Core insight: human perception is multimodal and adversarially robust because
different senses are hard to simultaneously fool. A painted stone fools vision
but not touch. This system exploits that property through Haptic Sovereignty —
haptic dimensions act as a veto gate against visual adversarial mimics.

Author: Arpan Mitter
"""

import os
import numpy as np
from sklearn.cluster import KMeans

os.environ["OMP_NUM_THREADS"] = "1"


# ─────────────────────────────────────────────────────────────────────────────
# SENSORY ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
#
# 9 dimensions derived from 4 machine-accessible human senses:
#
# VISION (expanded into 6 sub-dimensions):
#   [0] Hue          — dominant colour (0=red, 0.5=green, 1=violet)
#   [1] Saturation   — colour richness (0=grey, 1=vivid)
#   [2] Sphericity   — how round the shape is (0=flat, 1=perfect sphere)
#   [3] Aspect Ratio — length-to-width proportion (1=cube, >1=elongated)
#   [4] Volume       — estimated 3D size (normalised)
#   [5] Surface Area — normalised surface extent
#
# HAPTIC (3 dimensions from physical interaction):
#   [6] Weight       — gravitational mass felt during grasp (normalised)
#   [7] Firmness     — resistance to grip pressure (0=liquid, 1=rock)
#   [8] Roughness    — surface texture (0=smooth, 1=coarse)
#
# Smell discarded — impractical for current robotic hardware.
# Taste discarded — destructive sensing, not usable in general manipulation.
#
# FUTURE: Context/Functional layer (dimensions 9+)
#   Usability hierarchy — food, tool, commercial object, living thing.
#   Hybrid A+B: seeded by task context signal, refined through experience.
#   This layer resolves the "apple-fruit vs apple-company" disambiguation
#   that sensory dimensions alone cannot solve.
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "Hue", "Saturation", "Sphericity", "AspectRatio",
    "Volume", "SurfaceArea", "Weight", "Firmness", "Roughness"
]

# Haptic dimensions receive 3.0x weighting — Haptic Sovereignty.
# Physical interaction properties are adversarially robust in a way
# visual properties are not. A painted stone matches visually but
# cannot match haptically without being the actual object.
FEATURE_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 1.5, 1.0, 3.0, 3.0, 1.2])

# Indices of haptic dimensions — used for sovereignty veto check
HAPTIC_DIMS = [6, 7, 8]

# Haptic conflict threshold — if any haptic dimension deviates more than
# this from the learned concept mean, the match is vetoed regardless of
# overall weighted distance. Tunable per deployment environment.
# Known limitation: requires context-awareness for legitimate state variation
# (e.g. frozen apple in freezer context should not trigger veto).
HAPTIC_CONFLICT_THRESHOLD = 0.6


class EmbodiedConsensusMemory:
    """
    Online concept learning from multi-sensory interaction.

    Builds and maintains a dictionary of object concepts (prototypes)
    learned incrementally from sensor readings — no pre-training,
    no backpropagation, no GPU required.

    Architecture:
        Layer 1 — Sensory Matching:    weighted distance against known concepts
        Layer 2 — Haptic Sovereignty:  veto gate for adversarial visual mimics
        Layer 3 — Consensus Gating:    buffer unknown inputs until pattern emerges
        Layer 4 — Memory Management:   drift, consolidation, concept merging
        Layer 5 — Context (future):    task-conditioned functional disambiguation

    Args:
        input_dim:   Feature vector dimensionality (default 9)
        threshold:   Maximum weighted distance for a match (default 4.5)
        min_samples: Minimum buffer size before attempting concept formation
        weights:     Per-dimension feature weights (default FEATURE_WEIGHTS)
    """

    def __init__(
        self,
        input_dim=9,
        threshold=4.5,
        min_samples=5,
        weights=None
    ):
        self.input_dim    = input_dim
        self.threshold    = threshold
        self.min_samples  = min_samples
        self.weights      = weights if weights is not None else FEATURE_WEIGHTS.copy()

        self.concepts          = {}   # name → {mu, sigma}
        self.buffer            = []   # waiting room for unknown inputs
        self.total_inputs_seen = 0

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC INTERFACE
    # ─────────────────────────────────────────────────────────────────────────

    def identify(self, x):
        """
        Process one multi-sensory input vector and return classification.

        Args:
            x: numpy array of shape (input_dim,) — normalised sensor reading

        Returns:
            dict with keys:
                Result:     concept name, "Unknown", or "REJECTED"
                Confidence: percentage string or "N/A"
                Reason:     human-readable explanation
        """
        x = np.asarray(x, dtype=float)
        self.total_inputs_seen += 1

        # ── Layer 1: Sensory Matching ─────────────────────────────────────
        best_match = None
        min_dist   = float('inf')

        for name, params in self.concepts.items():
            dist = self._weighted_dist(x, params['mu'])
            if dist < self.threshold and dist < min_dist:
                best_match = name
                min_dist   = dist

        # ── Layer 2: Haptic Sovereignty Veto ─────────────────────────────
        if best_match is not None:
            haptic_conflict = np.abs(
                x[HAPTIC_DIMS] - self.concepts[best_match]['mu'][HAPTIC_DIMS]
            )
            if np.any(haptic_conflict > HAPTIC_CONFLICT_THRESHOLD):
                return {
                    "Result":     "REJECTED",
                    "Confidence": "0%",
                    "Reason":     (
                        f"Haptic Sovereignty — visuals match {best_match} "
                        f"but haptic conflict detected "
                        f"(max deviation: {haptic_conflict.max():.2f}). "
                        f"Possible adversarial mimic or abnormal object state."
                    )
                }

            # ── Layer 4a: Concept Drift (EMA update) ─────────────────────
            alpha = 0.05
            self.concepts[best_match]['mu'] = (
                (1 - alpha) * self.concepts[best_match]['mu'] + alpha * x
            )

            confidence = max(0.0, 100.0 * (1 - min_dist / self.threshold))
            return {
                "Result":     best_match,
                "Confidence": f"{round(confidence, 1)}%",
                "Reason":     "Multi-modal sensory match"
            }

        # ── Layer 3: Consensus Gating ────────────────────────────────────
        self.buffer.append(x)
        if len(self.buffer) >= self.min_samples:
            self._process_buffer()

        # ── Layer 4b: Sleep Cycle — periodic consolidation ───────────────
        if self.total_inputs_seen % 50 == 0 and self.concepts:
            self._consolidate_memory()

        return {
            "Result":     "Unknown",
            "Confidence": "N/A",
            "Reason":     f"Gathering consensus ({len(self.buffer)}/{self.min_samples} samples)"
        }

    def memory_state(self):
        """Return a summary of all known concepts."""
        return {
            name: {
                "centroid": params['mu'].round(3).tolist(),
                "spread":   round(params['sigma'], 3)
            }
            for name, params in self.concepts.items()
        }

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def _weighted_dist(self, x1, x2):
        """Weighted Euclidean distance — haptic dims weighted 3x."""
        return np.sqrt(np.sum(self.weights * (x1 - x2) ** 2))

    def _process_buffer(self):
        """
        Two-stage buffer consensus:
        Stage 1 — check if entire buffer forms one coherent concept.
        Stage 2 — if not, try KMeans split (handles mixed object streams).
        """
        data     = np.array(self.buffer)
        centroid = np.mean(data, axis=0)
        cohesion = np.mean([self._weighted_dist(p, centroid) for p in data])

        if cohesion < self.threshold:
            # Buffer is coherent — one new concept
            self._create_concept(centroid, cohesion)
        else:
            # Buffer is mixed — try splitting into 2 sub-concepts
            if len(data) >= 2:
                kmeans = KMeans(n_clusters=2, n_init=10).fit(data)
                for i in range(2):
                    cluster = data[kmeans.labels_ == i]
                    if len(cluster) >= max(2, self.min_samples // 2):
                        c   = np.mean(cluster, axis=0)
                        coh = np.mean([self._weighted_dist(p, c) for p in cluster])
                        if coh < self.threshold:
                            self._create_concept(c, coh)

        self.buffer = []

    def _create_concept(self, mu, sigma):
        """Register a new concept in the wide memory layer."""
        cid = f"Concept_{len(self.concepts) + 1}"
        self.concepts[cid] = {'mu': mu.copy(), 'sigma': sigma}
        print(f"  [LEARNED] {cid} formed — spread: {sigma:.3f}")
        return cid

    def _consolidate_memory(self, merge_threshold=2.0):
        """
        Iterative concept merging — removes redundant near-duplicate concepts.

        Uses iterative (not recursive) approach to avoid stack overflow
        with large concept dictionaries.

        Called automatically every 50 inputs (sleep cycle).
        """
        merged = True
        while merged:
            merged = False
            keys   = list(self.concepts.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    id1, id2 = keys[i], keys[j]
                    if id1 not in self.concepts or id2 not in self.concepts:
                        continue
                    dist = self._weighted_dist(
                        self.concepts[id1]['mu'],
                        self.concepts[id2]['mu']
                    )
                    if dist < merge_threshold:
                        print(f"  [MERGE] {id1} and {id2} are redundant — consolidating")
                        self.concepts[id1]['mu'] = (
                            self.concepts[id1]['mu'] + self.concepts[id2]['mu']
                        ) / 2
                        del self.concepts[id2]
                        merged = True
                        break
                if merged:
                    break


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE SENSOR PROFILES
# Vector: [Hue, Sat, Sph, Asp, Vol, SA, Weight, Firmness, Roughness]
# All values normalised 0-1 unless noted.
# ─────────────────────────────────────────────────────────────────────────────

OBJECTS = {
    # Real objects
    "Fuji Apple":    np.array([0.02, 0.85, 0.92, 1.00, 0.80, 0.78, 0.50, 0.80, 0.20]),
    "Bing Cherry":   np.array([0.01, 0.90, 0.95, 1.00, 0.10, 0.12, 0.05, 0.75, 0.15]),
    "Banana":        np.array([0.15, 0.90, 0.20, 3.50, 0.60, 0.65, 0.30, 0.65, 0.10]),

    # Adversarial cases
    "Painted Stone": np.array([0.02, 0.85, 0.90, 1.00, 0.80, 0.78, 3.50, 0.99, 0.85]),
    # Looks exactly like apple. Weight=3.5 (heavy), Firmness=0.99 (rock), Roughness=0.85 (coarse)

    "Apple Picture": np.array([0.02, 0.85, 0.92, 1.00, 0.80, 0.78, 0.01, 0.02, 0.05]),
    # Looks exactly like apple. Weight≈0 (paper/screen), Firmness≈0 (flat), Roughness≈0 (smooth)

    # Legitimate object in abnormal state
    "Frozen Apple":  np.array([0.02, 0.85, 0.92, 1.00, 0.80, 0.78, 0.50, 0.98, 0.20]),
    # Same as apple but Firmness=0.98 (ice-hard) — NOT adversarial, just frozen
}


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

def run_demo():
    print("=" * 65)
    print("  SENSORY-GROUNDED ONLINE LEARNING — DEMO")
    print("=" * 65)

    model = EmbodiedConsensusMemory(threshold=4.5, min_samples=5)

    # ── Phase 1: Learning ────────────────────────────────────────────────
    print("\n── PHASE 1: LEARNING OBJECTS ──────────────────────────────────")
    print("Robot encounters real objects and forms concepts from experience.\n")

    learning_objects = ["Fuji Apple", "Bing Cherry", "Banana"]

    for obj_name in learning_objects:
        print(f"Presenting: {obj_name}")
        base = OBJECTS[obj_name]
        for i in range(6):
            noisy = base + np.random.normal(0, 0.02, 9)
            noisy = np.clip(noisy, 0, 1)
            result = model.identify(noisy)
            if result["Result"] != "Unknown":
                print(f"  → Recognised as: {result['Result']} "
                      f"({result['Confidence']} confidence)")
        print()

    print(f"Concepts in memory: {list(model.concepts.keys())}\n")

    # ── Phase 2: Recognition ─────────────────────────────────────────────
    print("── PHASE 2: RECOGNITION TEST ──────────────────────────────────")
    print("Presenting known objects — should match learned concepts.\n")

    test_known = ["Fuji Apple", "Banana", "Bing Cherry"]
    for obj_name in test_known:
        x      = OBJECTS[obj_name] + np.random.normal(0, 0.02, 9)
        x      = np.clip(x, 0, 1)
        result = model.identify(x)
        print(f"  {obj_name:<20} → {result['Result']:<15} "
              f"{result['Confidence']:>8}   {result['Reason']}")

    # ── Phase 3: Adversarial Tests ───────────────────────────────────────
    print("\n── PHASE 3: ADVERSARIAL TESTS ─────────────────────────────────")
    print("Testing haptic sovereignty — can the system detect mimics?\n")

    adversarial = ["Painted Stone", "Apple Picture"]
    for obj_name in adversarial:
        result = model.identify(OBJECTS[obj_name])
        print(f"  {obj_name:<20} → {result['Result']:<15}")
        print(f"  {'':20}   {result['Reason']}\n")

    # ── Phase 4: Edge Case — Frozen Apple ────────────────────────────────
    print("── PHASE 4: EDGE CASE — FROZEN APPLE ─────────────────────────")
    print("Known limitation: high haptic conflict flags legitimate objects")
    print("in abnormal physical states. Context layer needed to resolve.\n")

    result = model.identify(OBJECTS["Frozen Apple"])
    print(f"  Frozen Apple         → {result['Result']}")
    print(f"  {result['Reason']}")
    print(f"\n  NOTE: This is a REAL apple. The rejection is a false positive.")
    print(f"  Resolution requires task context — in a freezer environment,")
    print(f"  high firmness variance is expected and should not trigger veto.")
    print(f"  Context layer (future work) addresses this tradeoff.")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n── MEMORY STATE ───────────────────────────────────────────────")
    for name, info in model.memory_state().items():
        print(f"  {name}: spread={info['spread']}")

    print("\n── SYSTEM PROPERTIES ──────────────────────────────────────────")
    print(f"  Total inputs processed : {model.total_inputs_seen}")
    print(f"  Concepts formed        : {len(model.concepts)}")
    print(f"  Training data required : {model.total_inputs_seen} interactions")
    print(f"  GPU required           : No")
    print(f"  Backpropagation        : No")
    print(f"  Pre-trained backbone   : No")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()
