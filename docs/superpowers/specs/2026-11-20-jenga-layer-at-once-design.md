# Jenga episodes: build a layer at a time (design)

Status: proposed, 2026-11-20. Stacked on `dn/jenga-scatter` (gloriacai01/cosmos-tutorial#3).

## Intent

The training plan ("Layer at once, to compare", Dhriti) proposes a second
way to build the tower: lay 3 blocks side by side on the table, pick the
whole layer up at once, and place it on the tower. That is 12 tower
placements instead of 36 (about 89% of towers at 99% per placement, against
70% for 36). We generate episodes for both ways in sim and keep the better
one. This adds the layer-at-once episodes to the existing generator.

**Goals**
- `place_episode.py --build layer` generates layer-at-once episodes; the
  default (`--build block`) is today's behaviour, unchanged.
- One script, no copied plumbing: batching, error handling, capture, stage
  prompts, nudges, upload and the end hold are shared with block mode.
- `--chain --build layer` builds a whole tower: twelve place-layer steps,
  the reset spawning the next layer on the pad before each one.

**Non-goals**
- Choosing between the two ways (that's the sim comparison, later).
- Real-hardware runs, or resetting the table between towers.
- New sim props or module changes.

**Anti-goals**
- Breaking or reshaping block mode, which is still under review in #2 and #3.
- A wrist turn with the bundle above the tower (Dhriti's #2: a big turn there
  swings the fingers through the layer).

## Approaches considered

- **A. One script, `--build` switch, layer code in its own module (picked).**
  Only the layer place is new; block mode is untouched.
- B. A separate `layer_episode.py` script. No merge risk with the open stack,
  but it copies connection, batching, error handling and capture.
- C. Generalise `execute_place` to a 1- or 3-block payload. Cleanest model, but
  a large rewrite of code still under review.

The user picked A ("make it an arg but same script so code is not repetitive").

## Decided

- **Grip for the bundle: side grip across all three blocks** (user, 2026-11-20).
  The G2's pads are ~29.6 mm wide, so an end grip on the 81 mm bundle would
  clamp only the middle block. Closing across the three widths (81 mm on the
  pad, inside the G2's 95 mm opening) clamps the outer blocks; friction holds
  the middle one. Block mode keeps the end grip.

## The design

### One episode kind: place the layer

The layer is given, not built (user, 2026-11-20): the episode reset spawns
the 3 blocks side by side on the **layer pad**, and every layer-mode episode
is a place-layer episode. Side-grip the bundle, lift, carry above the tower,
lower onto the top layer, release, then Dhriti's end (lift 10 cm above the
new top, open, hold 3 s). Prompt: "place the layer on the tower". Stages:
pick up the layer / carry the layer above the tower / lower the layer onto
the tower and open the gripper / lift the gripper away from the tower /
hold still.

### Layer pad

- Where the reset spawns the layer: default world (0.30, −0.15), 25 cm from
  the default tower centre (0.30, 0.10) and inside the 850's reach.
  `--pad-xy` overrides it.
- The spawned layer points in the direction of the tower's next layer, so the
  carry is nearly a pure translation. Per seed (my call, so the policy has to
  look rather than memorise): the pad centre is jittered ±1.5 cm and its yaw
  ±10°, and the three blocks get the tower's seam jitter. The seam between
  blocks is 1 mm (no end-grip finger has to fit between them).
- Blocks not in the tower or on the pad are scattered on the table as in
  block mode, as distractors (my call: same scene as block mode), and keep
  clear of the pad footprint by `ScatterConfig.tower_clearance_m`.

### Place-layer geometry

- Tool yaw = layer yaw + 90° + the G2's jaw offset, so the jaws close across
  the block widths. Grasp height: the existing G2 `grasp_z_offset_m`
  (fingertips 3 mm above the block bottoms).
- At release the fingertips are 3 mm above the held layer's bottom, so they
  stay above the layer below; they never come down beside the tower.
- Target: the three slot poses of the next tower layer. The bundle's measured
  offset in the hand after the lift is used to re-aim (as `release_target`
  does for one block), from the middle block.

### Episode sets in layer mode

- A (clean tower), B (drifted tower), C (perturbed start). D and E are left
  out for now; the result says which set ran.
- F (`--chain`): build the whole tower, layer by layer.
- Tower height: uniform over 0–11 complete layers.

### Results

Same record as block mode, plus `build: "layer"`, the episode kind, and for
place layer each block's placement error. Success: all three blocks within
the placement tolerance (2 mm / 2°) and no lower block disturbed.

### Code shape

- `jenga_tower.py`: `PadConfig`, pad layer poses (jittered), layer-mode
  spec builder (tower of whole layers, pad blocks, next layer's slot poses),
  scatter keeps clear of the pad.
- `layer_episode.py` (new): `execute_place_layer` using `_Episode`'s
  `go/transit/grab/open/mark/finish`.
- `place_episode.py`: `--build {block,layer}`, `--pad-xy`; dispatch only.
- Tests in the kinematic fake: side-grip yaw, no wrist turn over the tower,
  the pad layer follows the tower's next layer, scatter clear of the pad,
  a chained layer build completes, and block mode unchanged.

## Left to the builder

- Exact capture timings per stage (reuse block mode's minimum durations).
- How the fake models a 3-block grasp (the three move together if the jaw
  closes across all three).

## Not verified until a sim run

- That the bundle holds in a side grip (middle block by friction) through
  the carry. If it slips, options are a slower carry or a higher grip force.
- That the 1 mm pad seams and the ±10° pad yaw are within what the side grip
  squares up.
