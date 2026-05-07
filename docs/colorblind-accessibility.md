# Colorblind accessibility

All Earthmover colormaps are designed to be readable by people with color vision deficiencies (CVD). This document describes the design approach, testing methodology, and results.

## Design approach

The primary defense against colorblind inaccessibility is **monotonic lightness**. Regardless of how hues are perceived, a map that goes steadily from light to dark (or vice versa) will always convey data magnitude correctly. Every sequential Earthmover colormap has a monotonic J' (CAM02-UCS lightness) profile, the diverging map has a V-shaped profile, and the cyclic map has a symmetric arch.

This is the same principle behind matplotlib's `viridis` and the cmocean colormaps.

## Simulation methodology

Colormaps were tested using the Viénot et al. (1999) simulation matrices at **full severity** (100%), representing complete dichromacy — the most extreme form of color vision deficiency. The three simulated conditions are:

| Condition | Affects | Prevalence |
|-----------|---------|------------|
| **Deuteranopia** | Green cone (M-cone) absent | ~1% of males |
| **Protanopia** | Red cone (L-cone) absent | ~1% of males |
| **Tritanopia** | Blue cone (S-cone) absent | ~0.003% of population |

Deuteranopia and protanopia (red-green color blindness) account for ~99% of CVD cases, so these were the primary focus.

## Results

![Colorblind simulation](../images/colorblind_sim.png)

### Summary by colormap

**em.violet** (sequential, single-hue)
- Deuteranopia: Shifts to a blue-white gradient. Lightness gradient fully preserved.
- Protanopia: Shifts to a blue-white gradient. Lightness gradient fully preserved.
- Assessment: **Excellent.** Single-hue maps degrade gracefully under CVD since there's no hue contrast to lose.

**em.lime** (sequential, green family)
- Deuteranopia: Shifts to dark → golden yellow gradient. Lightness gradient preserved.
- Protanopia: Shifts to dark → blue-white gradient. Lightness gradient preserved.
- Assessment: **Excellent.** The lightness ramp carries all the information.

**em.signal** (sequential, violet → lime)
- Deuteranopia: Shifts to dark blue → light blue → golden → white. The violet-lime hue transition becomes blue-yellow, which is well-separated.
- Protanopia: Shifts to dark blue → light grey → yellow → white. Clear gradation maintained.
- Assessment: **Excellent.** The large lightness range (J' 15–97) ensures readability. Under CVD, the violet↔lime hue axis maps onto the blue↔yellow axis, which is the most discriminable axis for dichromats.

**em.diverging** (diverging, violet ↔ grey ↔ lime)
- Deuteranopia: Becomes blue ↔ light ↔ yellow. The neutral center is preserved, and both arms remain visually distinct.
- Protanopia: Becomes blue ↔ light ↔ yellow. Same positive result.
- Assessment: **Excellent.** The violet↔lime axis is ideal for a colorblind-safe diverging map because it maps onto blue↔yellow under both forms of red-green CVD.

**em.ocean** (sequential, cool multi-hue)
- Deuteranopia: Shifts to dark → blue → green-gold → yellow. Maintains a clear multi-step gradient.
- Protanopia: Shifts to dark → blue → light → golden. Some hue compression in the middle, but lightness carries the signal.
- Assessment: **Good.** The monotonic lightness ensures readability, though some of the blue↔green hue variation is compressed.

**em.bloom** (sequential, pink family)
- Deuteranopia: Shifts to dark → blue → light blue → white. Clean lightness ramp.
- Protanopia: Shifts to dark → blue-grey → light grey → white. Less saturated but fully readable.
- Assessment: **Excellent.** Near-single-hue path means CVD simply desaturates without disrupting the lightness gradient.

**em.cycle** (cyclic, brand axis loop)
- Deuteranopia: Violet half shifts to blue, lime half shifts to yellow. Both halves remain distinguishable.
- Protanopia: Similar result. The symmetric lightness arch (dark → light → dark) carries the structure.
- Assessment: **Good.** The cyclic structure is preserved via lightness symmetry, though some hue contrast is reduced at extreme CVD severity.

## Perceptual uniformity metrics

Perceptual uniformity is quantified using the coefficient of variation (CV) of the step-wise ΔE in CAM02-UCS color space. Lower CV means more uniform perceptual steps. All colormaps are optimized via arc-length parameterization in CAM02-UCS.

| Colormap | J' range | Mean ΔE | CV |
|----------|----------|---------|-----|
| em.violet | 15–97 | 0.44 | 0.1% |
| em.lime | 15–94 | 0.42 | 0.5% |
| em.signal | 15–97 | 0.67 | 1.5% |
| em.diverging | 38–97 | 0.52 | 0.7% |
| em.ocean | 15–86 | 0.50 | 0.5% |
| em.bloom | 15–97 | 0.42 | 0.0% |
| em.cycle | 15–97 | 0.77 | 0.3% |

The pipeline now prioritizes **linear J' progression** over minimum CV; em.ocean and em.diverging exceed `viridis`'s ~2–3% baseline (5.3% and 4.8% respectively) as a tradeoff for a perfectly straight lightness ramp.

## Recommendations

For maximum accessibility:

1. **Use `em.signal` or `em.diverging` as defaults.** The violet↔lime axis maps onto blue↔yellow under CVD — the most distinguishable pair for dichromats.
2. **Add contour lines or hatching** when colorblind safety is critical and data has fine structure.
3. **Avoid encoding information in hue alone.** The monotonic lightness of these colormaps means they work in grayscale too — test by printing in black and white.

## References

- Brettel, H., Viénot, F., & Mollon, J. D. (1997). Computerized simulation of color appearance for dichromats. *JOSA A*, 14(10), 2647–2655.
- Kovesi, P. (2015). Good colour maps: How to design them. *arXiv:1509.03700*.
- Crameri, F., Shephard, G. E., & Heron, P. J. (2020). The misuse of colour in science communication. *Nature Communications*, 11, 5444.
- Luo, M. R., Cui, G., & Li, C. (2006). Uniform colour spaces based on CIECAM02 colour appearance model. *Color Research & Application*, 31(4), 320–330.
