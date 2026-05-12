Sensory-Grounded Online Learning 
A bio-inspired alternative to big-data AI training — machines that learn object concepts through multi
sensory interaction, not billions of labelled images. 
 The Motivation 
This project is an attempt to explore an alternative to current AI training paradigms, which rely on neural architectures, massive
datasets, and enormous computational energy. Instead, it draws inspiration from how humans learn through their senses — a child
doesn’t need a million pictures of an apple to recognise one. Through repeated physical interaction, they build a rich, multi-sensory
idea of what an apple is. 
The aim here is to see if a machine, through its sensors — vision, touch, weight, texture — can build and retain concepts about
objects from a small number of real interactions, rather than passive exposure to billions of labelled images. If this approach can be
made to work, the energy efficiency implications for AI deployment — particularly on resource-constrained edge hardware — are
significant. 
 The Problem With Current AI Training 
Training GPT-4 consumed an estimated 50 gigawatt-hours of electricity. Every major AI lab is hitting the same wall — models require
exponentially more data and compute for diminishing returns in capability. 
The root assumption being challenged here is this: 
“To recognise an apple, a machine must see millions of apples from every possible angle, in every possible lighting condition,
labelled by humans.” 
Humans don’t learn this way. A child sees a few apples, handles them, and builds a concept. That concept generalises — to pictures
of apples, to apples of different colours, to apples in unusual orientations. The generalisation comes from understanding
characteristics, not from memorising appearances. 
This project asks: can we build a machine that learns the same way? 
 The Core Idea 
Instead of training on images, the system learns from multi-sensory interaction vectors — structured numerical representations of
what an object looks like, feels like, and weighs. 
When a robot encounters a new object: 1. Sensors collect a 9-dimensional reading 2. The system checks if this matches anything
already learned 3. If yes — it recognises the object and gently updates its memory 4. If no — it buffers the reading, waits for more
examples, and forms a new concept when a pattern emerges 
No backpropagation. No GPU. No pre-trained backbone. Learning happens in real time from a handful of interactions. 
 The Sensory Architecture — 9 Dimensions 
Derived from four machine-accessible human senses. Smell and taste are discarded — impractical for current robotic hardware. 
Vision — expanded into 6 sub-dimensionsDimension What It CapturesHue Dominant colour (0=red, 0.5=green, 1=violet)Saturation Colour richness (0=grey, 1=vivid)Sphericity How round the shape is (0=flat, 1=perfect sphere)Aspect Ratio Length-to-width proportion (1=cube, >1=elongated)Volume Estimated 3D size (normalised)Surface Area Normalised surface extent 
Vision is expanded because a single colour value cannot distinguish a cherry from a red marble. Shape, size, and proportion together
create a richer visual fingerprint that generalises better.Haptic — 3 dimensions from physical interactionDimension What It CapturesWeight Gravitational mass felt during graspFirmness Resistance to grip pressure (0=liquid, 1=rock)Roughness Surface texture (0=smooth, 1=coarse) 
Haptic sensing is what makes this architecture adversarially robust. A painted stone can match an apple on all 6 visual dimensions. It
cannot match on weight, firmness, and roughness simultaneously. 
 How The System Works 
New sensor reading (9D vector)
│
▼
┌───────────────────────────────┐
│ Layer 1: Sensory Matching │ Weighted distance against all known concepts
│ Haptic dims weighted 3x │ Most similar concept returned if within threshold
└───────────────┬───────────────┘
│ Match found?
┌───────┴───────┐
YES NO
│ │
▼ ▼
┌──────────────┐ ┌─────────────────────────┐
│ Layer 2: │ │ Layer 3: │
│ Haptic Veto │ │ Consensus Gating │
│ │ │ Buffer unknown input │
│ Check haptic │ │ Wait for min_samples │
│ dimensions │ │ Form new concept when │
│ independently│ │ pattern emerges │
└──────┬───────┘ └─────────────────────────┘
│
Conflict?
┌────┴────┐
YES NO
│ │
▼ ▼
REJECTED MATCH
Adversarial confirmed EMA drift update
mimic detected (concept evolves)

Layer 4 — Memory Management runs automatically every 50 inputs: - Concept drift via exponential moving average — concepts
evolve as objects are re-encountered - Concept merging — near-duplicate concepts are consolidated iteratively - No concept is ever
permanently frozen 
 Haptic Sovereignty 
The most important design decision in this system. 
Visual features are easy to spoof. A painted stone, a plastic fruit, a printed photograph — all can fool a vision-only classifier. Haptic
features are physically grounded and far harder to simultaneously deceive. 
The system implements a haptic sovereignty veto: even if the overall weighted distance places an input close to a known concept, if
any haptic dimension deviates significantly from the learned mean, the match is overridden and the input is flagged as a potential
adversarial mimic. 
# Even if overall distance passes, check haptic dimensions independently
haptic_conflict = np.abs(x[HAPTIC_DIMS] - concept['mu'][HAPTIC_DIMS])
if np.any(haptic_conflict > HAPTIC_CONFLICT_THRESHOLD):
return "REJECTED — Haptic Sovereignty"

Test ResultsObject Visual Match Haptic Match System DecisionReal Apple ✅ Apple ✅ Apple ✅ RecognisedBing Cherry ❌ No match ❌ No match ✅ New concept formedBanana ❌ No match ❌ No match ✅ New concept formedPainted Stone ✅ Apple ❌ Stone ✅ REJECTEDApple Picture ✅ Apple ❌ Paper/Screen ✅ REJECTED 
 The Frozen Apple Problem — Honest Limitation 
The haptic sovereignty mechanism creates a precision-recall tradeoff: 
A frozen apple has identical visual properties to a room-temperature apple but anomalous firmness (ice-hard). The system correctly
identifies the visual match but incorrectly rejects it as adversarial — it is a real apple, just in an abnormal physical state. 
This is not a bug. It is a fundamental limitation of context-free haptic thresholding. 
The resolution requires a context layer — in a freezer environment, high firmness variance is expected and the conflict threshold
should adapt accordingly. At room temperature, it should not. 
This tradeoff — between adversarial robustness and tolerance for legitimate state variation — is an open research problem
documented here as a direction for future work. 
 The Context Layer — Future Work 
The current system has no way to distinguish: - Painted stone (adversarial — should be rejected) - Frozen apple (legitimate — should
be recognised with a state flag) 
Resolving this requires a task-conditioned functional disambiguation layer — a hybrid of: 
Option A — Explicit context signal: task environment feeds in a context vector (“kitchen, room temperature, grasping task”) that
adjusts thresholds per domain. 
Option B — Learned functional associations: through repeated experience, the system builds associations between objects and
contexts — a frozen apple encountered repeatedly in freezer contexts becomes a known variant, not an anomaly. 
The full architecture vision: 
Layer 1: Sensory Identity — what does it look and feel like? [implemented]
Layer 2: Haptic Sovereignty — is it physically consistent? [implemented]
Layer 3: Consensus Gating — is this pattern new or known? [implemented]
Layer 4: Memory Management — drift, merge, consolidate [implemented]
Layer 5: Functional Context — what is it for, in this context? [future work]

The functional context layer also addresses the symbol grounding problem — “apple” as fruit vs “Apple” as technology company.
Same sensory fingerprint, different functional hierarchy. Usability and domain context disambiguate what pure sensation cannot. 
 What This Is And Isn’t 
What it is: - A conceptual prototype and proof-of-concept for embodied multimodal perception - A lightweight, CPU-deployable
online learning system - An exploration of bio-inspired alternatives to data-hungry training paradigms - A documented investigation
of the adversarial robustness properties of haptic sensing 
What it isn’t: - A production-ready classifier - A replacement for deep learning on complex perception tasks - Validated on real sensor
hardware (simulation only at this stage) - Benchmarked against a baseline on a standard dataset (future work) 
The 9D feature vectors are currently hand-crafted for simulation. A complete implementation would replace these with real sensor
pipelines — a CNN for visual feature extraction, force/torque sensors for haptic dimensions — and measure performance on a real or
semi-real manipulation dataset. 
 Quickstart 
pip install numpy scikit-learn python
embodied_memory.py
Expected output walks through 4 phases: 1. Learning — robot forms concepts from 6 noisy examples per object 2. Recognition —
known objects correctly identified 3. Adversarial — painted stone and apple picture correctly rejected 4. Edge case — frozen apple
limitation documented with explanation 
 System PropertiesProperty ValueTraining examples required ~5 per conceptGPU required NoBackpropagation NoPre-trained backbone NoInference complexity O(concepts)Consolidation complexity O(concepts2) per sleep cycleHardware target CPU, embedded systems 
 Related Work 
This project connects to several active research areas: 
Adaptive Resonance Theory — Carpenter & Grossberg (1987) — online concept formation with stability-plasticity tradeoff
Few-Shot Learning / Prototypical Networks — Snell et al. (2017) — classification from few examples via prototype centroids
Embodied Cognition in Robotics — Brooks (1991), Pfeifer & Scheier (1999) — intelligence grounded in physical interaction
Green AI / Energy-Efficient ML — Schwartz et al. (2020) — “Green AI”, Communications of the ACM
Multimodal Robotic Perception — ongoing work at MIT, Stanford, ETH Zurich on haptic-visual fusion for manipulation 
 Relation To Other Projects 
This project is part of a broader research thread across this portfolio: 
Boids — emergent behaviour from local rules
GA TSP — optimisation without gradient descent
MATD3 swarm — decentralised coordination, bandwidth efficiency
KAN vs MLP — parameter efficiency, interpretable functions
Sensory-Grounded Learning — concept formation without massive datasets

The unifying question across all of them: 
“Can we achieve intelligent behaviour with less — less data, less compute, less communication, less energy?” 
 Author 
Independent research exploring bio-inspired alternatives to data-hungry AI training paradigms, with a focus on energy-efficient
deployment on resource-constrained hardware
