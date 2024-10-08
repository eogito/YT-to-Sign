WLASL: A large-scale dataset for Word-Level American Sign Language (WACV 20' Best Paper Honourable Mention)
============================================================================================

This repository contains the `WLASL` dataset described in "Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison".

Please visit the [project homepage](https://dxli94.github.io/WLASL/) for news update.

Please **star the repo** to help with the visibility if you find it useful.

**yt-dlp vs youtube-dl** youtube-dl has had low maintance for a while now and does not work for some youtube videos, see this [issue](https://github.com/ytdl-org/youtube-dl/issues/30568).
[yt-dlp](https://github.com/yt-dlp/yt-dlp) is a more up to date fork, which seems to work for all youtube videos. Therefore `./start_kit/video_downloader.py` uses yt-dlp by default but can be switched back to youtube-dl in the future by adjusting the `youtube_downloader` variable.
If you have trouble with yt-dlp make sure update to the latest version, as Youtube is constantly changing.

Download Original Videos
-----------------
1. Download repo.
```
git clone https://github.com/dxli94/WLASL.git
```

2. Install [youtube-dl](https://github.com/ytdl-org/youtube-dl) for downloading YouTube videos.
3. Download raw videos.
```
cd start_kit
python video_downloader.py
```
4. Extract video samples from raw videos.
```
python preprocess.py
```
5. You should expect to see video samples under directory ```videos/```.

Requesting Missing / Pre-processed Videos
-----------------

Videos can dissapear over time due to expired urls, so you may find the downloaded videos incomplete. In this regard, we provide the following solution for you to have access to missing videos.

We also provide pre-processed videos for the full WLASL dataset on request, which saves troubles of video processing for you.

 (a) Run
```
python find_missing.py
```
to generate text file missing.txt containing missing video IDs.

 (b)  Submit a video request by agreeing to terms of use at:  https://docs.google.com/forms/d/e/1FAIpQLSc3yHyAranhpkC9ur_Z-Gu5gS5M0WnKtHV07Vo6eL6nZHzruw/viewform?usp=sf_link. You will get links to the missing videos within 7 days (recently I got more occupied, some delays may be expected yet I'll try to share in one week. If you are in urgent need, drop me an email.)

File Description
-----------------
The repository contains following files:

 * `WLASL_vx.x.json`: JSON file including all the data samples.

 * `data_reader.py`: Sample code for loading the dataset.

 * `video_downloader.py`: Sample code demonstrating how to download data samples.

 * `C-UDA-1.0.pdf`: the Computational Use of Data Agreement (C-UDA) agreement. You must read and agree with the terms before using the dataset.

 * `README.md`: this file.