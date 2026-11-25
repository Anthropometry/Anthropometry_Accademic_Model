# Data

## BodyM (Amazon, 2022)

- Download: `aws s3 sync --no-sign-request s3://amazon-bodym/ bodym` (no AWS account needed)
- Licence: Creative Commons Attribution-NonCommercial 4.0. School use only; never commit it here.
- 2,505 adults; front and side silhouettes; height, weight, gender; 14 measurements in cm.
- Captured in a well-lit lab, A-pose, tight clothing, 5.5–6.5 ft from the camera. Test-B photos
  were taken in less controlled conditions.
- **Measurements come from 3D scans registered to SMPL and measured on the mesh** — a virtual
  tape, not a tailor's tape (D-009).
- Ethnicity: White 40%, Asian 30%, Black/African American 14%, American Indian or Alaska Native 1%,
  other 15%. No Ghana-specific data; that gap is what step 6 measures.

Layout (confirmed 15 Nov 2026):

```
bodym/<train|testA|testB>/
    hwg_metadata.csv          subject_id, gender, height_cm, weight_kg
    measurements.csv          subject_id, ankle, arm-length, bicep, calf, chest, forearm, height,
                              hip, leg-length, shoulder-breadth, shoulder-to-crotch, thigh, waist, wrist
    subject_to_photo_map.csv  subject_id, photo_id
    mask/<photo_id>.png       front silhouette
    mask_left/<photo_id>.png  side silhouette
```

| Split | People | Photo pairs | Pairs per person |
| --- | ---: | ---: | ---: |
| train | 2,018 | 6,134 | ~3 |
| testA | 87 | 1,684 | ~19 |
| testB | 400 | 1,160 | ~3 |

## SMPL-X (step 5)

Register at smpl-x.is.tue.mpg.de and download the model files yourself (non-commercial research
licence). Keep them on your own Drive; they are blocked by `.gitignore`.

## Segmentation models (S7a, D-043)

| Model | Used by | Licence | Note |
| --- | --- | --- | --- |
| DeepLabV3-ResNet101 (torchvision, COCO person class) | tool A | torchvision code BSD-3-Clause; weights trained on COCO | downloaded at run time, never committed |
| U²-Net human segmentation (via rembg) | tool B | rembg MIT; U²-Net Apache-2.0 | downloaded at run time, never committed |

Test photos (Ryan's pilot) stay on Drive; never committed, never printed, never saved by the code unless
`--save-silhouettes` points somewhere. Treated as pilot material, not research data, until the IRB answer.

## Ghanaian data (step 6)

Collected by AnthropometryDC after ethics approval. Reaches this project only as a de-identified
export prepared by Ryan (no names, faces or phone numbers), stored outside the repository.
