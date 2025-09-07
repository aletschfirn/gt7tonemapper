# GT7 Tonemapper implemented in HLSL

This is a basic implementation of GT7 Tonemapper from ["Driving Toward Reality: Physically Based Tone Mapping and Perceptual Fidelity in Gran Turismo 7"](https://s3.amazonaws.com/gran-turismo.com/pdi_publications/s2025_PBS_Physically_Based_Tone_Mapping_GT7.pdf) presentation and it's [Tone Mapper operator example code](https://blog.selfshadow.com/publications/s2025-shading-course/pdi/supplemental/gt7_tone_mapping.cpp) in HLSL language with ReShade support.

Notes:

- Currently implements only curve (e.g. fixing highlights)
- Doesn't have eye adaptation effect _yet_
- Doesn't have any LUT to make colours "dramatic"
- Might be unoptimised (float in hlsl doesn't really support numbers with lot of digits it seems?)

**"My image looks dark... what do?"**

The reason for this is how tone mapper works. It expects high dynamic range, open domain, scene-referred, and clean input, while ReShade gives low dynamic range, closed domain, display-referred input. Use "Exposure" setting at ~2.30 to achieve nice result with SDR display mode. There's also HDR display mode option, which, on SDR displays, might be what you need.

While Tonemapper might improve some stuff with renderer, it doesn't fully fix all the problems you might have or make image cinematic like shown in ["Driving Toward Reality"](https://s3.amazonaws.com/gran-turismo.com/pdi_publications/s2025_PBS_Physically_Based_Tone_Mapping_GT7.pdf).
