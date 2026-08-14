@echo off
cls
echo Running modified LivePortriat
set OPENCV_IO_ENABLE_OPENEXR=1
python inference.py %*
@REM python inference.py %* --flag_eye_retargeting --flag_lip_retargeting

echo Warping high resolution image
python high_resolution_warp.py %*

start /b python assemble_patches.py -nip 1 %*

echo Preparing and Inpainting occlusion patches
start /b python prepare_references.py %*
python prepare_patches.py %*
python inpaint_custom_lora.py %*

echo Assembling final image
python assemble_patches.py %*

echo Generating baseline comparisons and metrics
start /b python compare.py %*
@REM python compare.py %*
@REM python measure_metrics.py %*
@echo on
