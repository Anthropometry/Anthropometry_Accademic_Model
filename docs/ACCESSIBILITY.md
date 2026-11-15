# Accessibility for this model

For this project, accessibility has four parts: the **model works for every body it claims to
serve** (and says clearly when it doesn't), the **demo is usable** by people with different
abilities and phones, the **work can be run** on free tools, and the **results can be understood**
by non-experts.

## 1. Bodies: who the model serves, and who it doesn't yet

A measurement model can look accurate on average and still fail particular groups. BodyM is mostly
BMI 18.5–30, adults, lab conditions, tight clothing; 14% Black/African American, none from Ghana.

| Group | Risk | What we do |
| --- | --- | --- |
| Larger bodies (BMI 30+) | Few training examples → larger errors | Report % within ±1.5 cm by BMI band; flag bands with < 20 people as unreliable |
| Women vs men | Different error patterns (bust, hip) | Every result split by sex |
| Ghanaian body shapes | Training data has none | Step 6 calibration and a per-measurement report on Ghanaian participants |
| People who can't stand unaided or hold A-pose (wheelchair users, crutches, some older people) | Model assumes a standing A-pose | **Out of scope.** The demo says so and routes to tape measurement; never returns a guess |
| Limb differences, prosthetics, braces | Silhouette doesn't match the training bodies | Same: out of scope, clearly stated, tape route offered |
| Children | BodyM is adults only | Out of scope for the semester |

Silhouettes remove skin colour, faces and hair, which reduces one source of bias. They don't remove
body-shape bias, which is why we measure results by group.

## 2. The demo (step 7)

Built to WCAG 2.2 level AA:
- Every image has text alternatives; capture guidance is written as well as drawn.
- Works by keyboard and screen reader; visible focus; colour is never the only signal (flags are
  words, not just red); contrast at least 4.5:1.
- Plain language, short sentences; units in cm with inches available.
- Works on a low-cost Android phone and a slow connection: small pages, no heavy downloads.
- Height entry accepts cm or feet/inches and shows the value back before submitting.
- Results show which measurements to confirm with a tape, and why.

## 3. Running the work

- Everything runs on the free Google Colab tier; no paid GPU or service.
- Tests run on a laptop CPU in about 10 seconds with a fake dataset.
- Every command is documented in README.md and CLAUDE.md.

## 4. Understanding the results

- Every result is in centimetres, next to the baseline, with the % of people within ±1.5 cm.
- Charts use colour-blind-safe palettes and label lines directly.
- Each step's report opens with a two-sentence plain-language summary for the supervisor.
