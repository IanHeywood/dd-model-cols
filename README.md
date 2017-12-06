# dd-model-cols

Applies masks to channel model images (typically from wsclean) and re-predicts a per-mask visiblity column into the MS for use with direction-dependent calibration via CubiCal. This is an attempt to do facet-calibration in a quick and dirty way, with the price being the use of a lot of intermediate scratch space on the disk. As you can see it's geared towards scenarios where the primary beam is the principal source of DD trouble.

![](https://i.imgur.com/5JHTjlf.gif)
