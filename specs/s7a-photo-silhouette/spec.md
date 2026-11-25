# S7a: from a real phone photo to the full measurement set

**Status:** approved by Ryan, 25 Nov 2026 (test both tools, D-043; side direction detected and mirrored
automatically, D-044). Question 4 (test people) explained again, answer pending: it does not block the build.
**Branch:** `s7a-photo-silhouette`

## Goal (one sentence)

Let the model measure a new person from two ordinary photos (front and side) plus height, by turning each
photo into a silhouette like BodyM's, refusing photos it can't trust, and passing the pair to the existing
`predict-silhouettes` chain (24 rows for women, 18 for men).

## Why this step matters

Everything so far runs on BodyM's ready-made silhouettes. BodyM has **no photos**, so nobody new can be
measured yet. S7a is the missing link between the app's photos and the model, and the first step where
real Ghanaian photos can go in.

## The core problem, honestly

The photo model learned from BodyM's silhouettes, made by Amazon's own (unpublished) tool. Our silhouettes
will come from a different tool, so their edges will differ slightly. At about 2 mm per pixel, one pixel of
width moves a waist girth by about 0.8 cm (S3b). So **the choice of tool is a measurement-accuracy choice**,
not just a convenience, and it has to be judged on evidence (below), not taste.

## Method and why

### 1. Segmentation: two candidates, tested side by side

| | A: torchvision DeepLabV3 (person class) | B: rembg with a human-segmentation model (U²-Net / ISNet) |
| --- | --- | --- |
| New package | none (torchvision is already used) | `rembg` + `onnxruntime` (MIT / MIT) |
| Download | model weights on first use (about 170 MB, cached on Drive) | model weights on first use (about 170 MB) |
| Strength | no new package; well-known | made for cutting people out; usually sharper edges |
| Weakness | trained for scenes, edges softer | two new packages |

Both run on Colab CPU or GPU. Rejected for now: SAM (heavy, needs prompts) and MediaPipe (good for the phone
later, but a third new package now). The interface is one function, so a tool can be swapped later without
touching the rest.

**How the winner is chosen (on Ryan's test photos, decided before looking at the outputs):**
1. **Height agreement:** the model's own height estimate vs the tape height typed in; the D-029 rule already
   flags gaps of 2 cm or more. Fewer flags is better: it is the one check with a tape value for every photo.
2. **Repeat stability:** two photo pairs of the same person, taken a minute apart: the smaller the spread of
   the girths, the better.
3. **Retake rate:** how often the outline and the photo model disagree (S3b warning).
4. **A visual check** of the outline drawn over each photo, shown in Ryan's Colab only, never saved.
5. **If tape values exist for those people** (question 4), their per-person errors decide, above all else.

### 2. Cleaning the mask
Keep the largest connected body; fill holes (between legs is kept open); drop specks; smooth one pixel.
**Why:** a stray blob or a hole changes widths, and so girths.

### 3. Checks that refuse a photo (each refusal says what to fix)
- no person, or more than one person;
- head or feet cut off by the frame edge;
- body shorter than 600 pixels in the photo (too far away: each pixel would be worth too much);
- tilt beyond the app's own limit when the app recorded it (`level_pitch_deg`, `level_roll_deg`);
- in the side photo, arms in front of the body.

### 4. Making it look like BodyM
- White body on black, cropped head to heel with margin and scaled to BodyM's full-resolution size (body
  about 830 pixels tall), so the geometry (S3b) sees the same scale it was tuned on.
- **Side direction:** BodyM's side silhouettes all face one way. The first step checks which way on BodyM's
  data; side photos facing the other way are mirrored. (Question 3: which side the app asks people to turn.)

### 5. Output
`anthro predict-photos --front F.jpg --side S.jpg --height H --sex female [...]` gives exactly the
`anthro-output/1` result of S7-0, plus `inputs.kind: "photos"` and the segmentation tool's name and
version. `--save-silhouettes DIR` writes the two silhouettes for checking (to Drive, never the repo).

## Inputs and outputs

| In | Out |
| --- | --- |
| Front and side photos (JPEG/PNG, any phone), tape height, sex, age range, optional weight | The same full result as `predict-silhouettes`, or a refusal saying what to fix |

## Acceptance criteria (testable)

- [ ] A fake photo with one person gives a clean silhouette of the right size and direction.
- [ ] Each refusal case above refuses, with its reason.
- [ ] `predict-photos` output validates as `anthro-output/1` and equals `predict-silhouettes` on the same masks.
- [ ] Both candidate tools run behind one interface; the comparison command prints the four scores per tool.
- [ ] No photo or silhouette is written anywhere except where `--save-silhouettes` points.
- [ ] Tests use a fake segmenter (no download in CI); coverage stays ≥ 80%.
- [ ] Ryan's Colab run with his test photos picks the tool; the decision is logged.

## How it is scored

Not against BodyM (no photos there). On Ryan's test photos: the four checks above per tool. If tape values
exist for the people photographed, per-person error and % within ±1.5 cm for each measurement: the first
real numbers on Ghanaian bodies (reported as a small pilot, not a result).

## Security, licence and privacy checks

- New packages (B only) and model downloads approved by Ryan first (question 1); licences recorded in
  docs/DATA.md and specs/tech-stack.md.
- Photos never committed and never copied outside Colab/Drive; faces are not needed (silhouettes only), and
  the app's photos are already face-blurred.
- Test photos only of people who agreed (question 4); treated as pilot material, not research data, until the
  IRB answer (policy).

## Accessibility and fairness checks

- Refusal rates and retake rates by sex and BMI band in the comparison (dark skin on dark backgrounds and
  loose clothing are known weak spots for segmentation tools: reported, not hidden).
- Clothing: the result keeps the "everyday clothing not tested" warning until tested.

## Tests to write

| Test | Allow or refuse |
| --- | --- |
| Synthetic photo, one person: silhouette size, crop, direction | allow |
| No person / two people / head cut / feet cut / too small / tilted | refuse, each with reason |
| Side photo facing the other way is mirrored | allow |
| Holes and specks cleaned; the gap between the legs kept | allow |
| `predict-photos` equals `predict-silhouettes` on the same masks | allow |
| Nothing written without `--save-silhouettes` | allow |

## After this

- **S7b:** run the whole chain over an AnthropometryDC export folder (the loader exists: S7-0) and, where tape
  values exist, report per-person errors: the bridge to S6/S8 (calibration and training on Ghanaian data).
- Update notebooks 05 and 06 to clone `main` (they still point at the merged feature branches).

## Ryan's answers (25 Nov 2026)

1. Test both A and B (D-043). 2. Approved. 3. Not sure which shoulder faces the camera: the code detects the
direction and mirrors when needed (D-044). 4. Pending (explained again in plain words).

## Questions as asked

1. **New dependency:** approve testing both candidates (A needs only a weight download; B adds `rembg` and
   `onnxruntime`), or A only?
2. **Approve S7a as written?**
3. **Side photo:** in the app, which side faces the camera (left shoulder or right shoulder), and is the pose
   the same as BodyM's (arms slightly away from the body in the front photo, arms down in the side photo)?
4. **Test photos:** can you take 3–5 people (yourself, Patience, consenting friends) with front and side photos
   from the app **and** tape measurements? That gives the first real error numbers. Twice per person, a minute
   apart, for the stability check.
