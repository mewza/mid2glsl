<div align="center">

# $${\Huge\color{cyan}\textsf{🎵 MID2GLSL v1.3}}$$

$${\Large\color{orange}\textsf{Add MIDI soundtrack to your ShaderToy shaders}}$$
$${\Large\color{orange}\textsf{very light-weight music player FM + additive synthesis}}$$$${\small\color{lightgray}\textsf{© 2026 Orblivius — All rights reserved}}$$

## $${\color{deepskyblue}\textsf{🎬 Video demo}}$$

<div align="center">
https://github.com/user-attachments/assets/7761e718-f032-4d33-b57e-ca971a535e86
<p align=center>
<b>Live Demo: <a href="https://www.shadertoy.com/view/sc33zs">Return to the Monkey Island</a></b>
  </p>
</div>


## $${\color{cyan}\textsf{📖 Overview}}$$
<p align=left>
`mid2glsl.py` allows you to embed midi files (soundtrack) by adding FM and additive type of soft-synth into ShaderToy shaders making presentation more alive and interesting to watch. It has been optimized for fastest possible load time so that your shader will not be marked as a slow loading type.
</p>

<p>
  Usage: python mid2glsl.py in.mid [out.glsl] [--bpm N] [--mode grains|macro|buffer|player] [--inst brass|...|piano6 (default: piano6)] [--no-reverb] [--reverb N] [--viz [1|2|3|4|5|led|cd|piano|melee|island]] [--no-viz] [--gm] [--time N]  (piano viz is the DEFAULT; --reverb N sets REVERB_TAPS, e.g. 32 = denser room, default 16; --gm = multi-timbral General-MIDI engine, per-channel program -> voice, organ family voiced; --time N = playback/data limit seconds, default 180)
</p>
![Version](https://img.shields.io/badge/version-1.0-orange?style=flat-square)
![Python](https://img.shields.io/badge/python-3.x-blue?style=flat-square&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/platform-ShaderToy-7e57c2?style=flat-square)
![License](https://img.shields.io/badge/license-non--commercial-green?style=flat-square)
![MIDI](https://img.shields.io/badge/MIDI-Midi-ff69b4?style=flat-square)
