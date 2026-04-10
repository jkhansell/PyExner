module load ffmpeg

cd fields

ffmpeg -framerate 7 -pattern_type glob -i "fields_*.png" -vf "palettegen=stats_mode=diff" palette.png
ffmpeg -framerate 7 -pattern_type glob -i "fields_*.png"    \
    -i palette.png                                          \
    -lavfi "paletteuse=dither=bayer:bayer_scale=5"          \
    fields.gif

cd ..
