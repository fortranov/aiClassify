@echo off
REM ============================================================
REM photo-tagger — build and run
REM
REM Adjust the three paths below before first run:
REM   PHOTOS  — folder with your photos (already mounted on host)
REM   DATA    — persistent cache for embeddings / done markers
REM   MODELS  — HuggingFace model cache (kept between container restarts)
REM
REM Docker Toolbox note: use forward slashes and /c/... notation
REM   e.g. Z:\photos  →  /z/photos   (VirtualBox shared folder)
REM ============================================================

set PHOTOS=/z/photos
set DATA=C:\docker-volumes\tagger-data
set MODELS=C:\docker-volumes\tagger-models

REM -- build ---------------------------------------------------------
docker build -t photo-tagger .

REM -- run -----------------------------------------------------------
docker run -d ^
  --name photo-tagger ^
  --restart unless-stopped ^
  -p 8080:8080 ^
  -v "%PHOTOS%":/photos ^
  -v "%DATA%":/app/cache ^
  -v "%MODELS%":/app/models ^
  -e PHOTO_ROOT=/photos ^
  -e SCORE_THRESHOLD=0.24 ^
  -e MAX_TAGS=10 ^
  -e USE_POLLING=true ^
  photo-tagger

echo.
echo Container started.  API: http://localhost:8080
echo Health:  http://localhost:8080/health
echo Stats:   http://localhost:8080/stats
echo Tags:    http://localhost:8080/tags
