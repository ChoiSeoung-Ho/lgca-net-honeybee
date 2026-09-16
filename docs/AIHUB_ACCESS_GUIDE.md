# Obtaining the images from AI-Hub (English step-by-step guide)

The images used in this study belong to the **AI-Hub Honeybee Disease Diagnosis Image
Dataset (Dataset No. 71667)**, published in 2023 by the National Information Society
Agency (NIA) of the Republic of Korea. AI-Hub is a Korean-language portal, access
requires a free account and acceptance of a data-use agreement, and the agreement does
not permit third parties to redistribute the image files. This guide therefore
explains, in English, how to obtain the exact images used in the article. Everything
*derived* from the images (per-image statistics, split manifests, matched-subset lists,
per-image predictions, training logs) is already in this repository, so **all tables and
figures of the article can be reproduced without downloading anything** (`make
reproduce-tables`). The images are needed only to re-train the networks (Level B).

## 1. Create an account

1. Open https://www.aihub.or.kr and click **회원가입** ("Sign up", top right).
2. Choose **개인회원** (individual member). A Korean mobile number is *not* required:
   select e-mail verification (**이메일 인증**) where offered. Foreign researchers can
   register with an institutional e-mail address; if the form insists on identity
   verification, use the **외국인** ("foreigner") option.
3. Confirm the verification e-mail and log in (**로그인**).

## 2. Locate the dataset

Direct link: https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71667

Or search for the Korean title **꿀벌 질병 진단 이미지 데이터** ("honeybee disease
diagnosis image data") in the search box, or browse **데이터 찾기 → 농축수산**
(Find data → Agriculture/livestock/fisheries).

## 3. Request and download

1. On the dataset page click **다운로드** ("Download").
2. Accept the data-use agreement (**이용약관 동의**). Non-commercial research use is
   permitted; redistribution is not.
3. AI-Hub offers a browser download and a command-line downloader (**aihubshell**).
   For the browser route, tick the folders listed below and click **다운로드**.
   For the command-line route, follow the instructions on the page
   (`aihubshell -mode d -datasetkey 71667 -filekey <keys>`); the file keys of the
   folders are shown next to each folder on the download page.
4. The archive is split into training (**Training**) and validation (**Validation**)
   portions, each with **원천데이터** (raw images, JPG) and **라벨링데이터** (JSON labels).
   Only the raw-image folders are needed for this study; the JSON boxes are used by
   the (not yet run) lesion-localisation script only.

## 4. Select the images used in the article

File names have the form `f1_f2_f3_YYYYMMDDhhmmss_f5_f6_f7_f8.jpg`; the last field is
the disease code (`000` normal, `002` Chalkbrood, `003` Foulbrood). The exact 1,500
file names used in the study (500 normal, shared by both tasks; 500 Chalkbrood-positive;
500 Foulbrood-positive) are listed in `data/image_lists/`. Copy them into

```
code/data/you_chalk_brood/normal/     ← data/image_lists/you_chalk_brood_normal.txt
code/data/you_chalk_brood/abnormal/   ← data/image_lists/you_chalk_brood_abnormal.txt
code/data/you_foulbrood/normal/       ← data/image_lists/you_foulbrood_normal.txt
code/data/you_foulbrood/abnormal/     ← data/image_lists/you_foulbrood_abnormal.txt
```

A helper does this for you once the archive is extracted:

```bash
python code/select_images.py --aihub-root /path/to/extracted/71667 --out code/data
```

It searches the extracted tree for the listed file names, copies (or symlinks with
`--link`) them into the task folders, and reports any name it could not find.

## 5. Verify

`python code/make_splits.py --data-root code/data` must reproduce
`data/splits/split_manifest_grouped_<task>.csv` exactly (same file names, groups and
folds; seed 42), and `python code/extract_image_features.py --data-root code/data` must
reproduce `data/features/features_<task>.csv` to floating-point precision. If either
differs, the wrong files were selected.

## 6. If you cannot obtain the images

Contact the corresponding author (jcn99250@naver.com). The author cannot send the
images, but can confirm file lists, checksums and provide model checkpoints (6.3 GB)
on request, and every analysis in the article runs from the released derived data.
