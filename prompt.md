## ROLE

You are a world-class creative frontend engineer with deep expertise in GSAP animation systems, Apple-style motion design, high-end UI/UX, and performance-first CSS architecture.

Think like both a Creative Director (experience & emotion) and a Senior Engineer (clean, scalable code).

---

## PRIMARY GOAL

Build a premium cinematic interactive website as a single self-contained HTML file.

UI/UX quality is the single highest priority.

The experience must feel: Apple-level smooth, cinematic, elegant, emotion-driven.

---

## PRIORITY ORDER

1. Visual design (HIGHEST)

2. Smoothness & animation quality

3. Interaction & experience

4. Code quality

5. Performance

---

## DESIGN SYSTEM

Theme: dark — background #050505

Accent: context-driven 

Typography: Inter (Google Fonts), clean weights only

Rules:

- Minimal, premium, strong spacing

- Subtle glow accents

- Glass morphism where appropriate

- No visual clutter

---

## ANIMATION SYSTEM (GSAP via CDN)

Load GSAP from CDN. Use it for all transitions.

Rules:

- Easing: power3.out everywhere

- Always use stagger + delay — never instant

- Motion hierarchy: hero → UI layer → content

- Micro-interactions: hover = scale + glow, focus = smooth highlight, text = fade + slide

Never:

- Linear easing

- Instant state changes

- Harsh jumps

---

## CINEMATIC INTRO SEQUENCE

1. Black screen

2. Fade in (opacity 0 → 1, ~1.2s, power3.out)

3. Title appears word-by-word with stagger

4. Subtitle fades in with delay

5. CTA button slides up: "Enter Experience"

6. Click → blur + fade transition into main content

---

## SCROLL SYSTEM

- Smooth interpolated scroll 

- Sections animate in on scroll 

- Parallax depth via CSS transforms

- Each section has its own entrance choreography

---

## INTERACTION SYSTEM

- Mouse parallax on hero 

- Custom cursor: dot + ring, scales on hover

- All hover states animated, never instant

---

## SECTIONS

Hero:

- Fullscreen dark hero

- Centered title + subtitle

- Animated background: radial gradient pulse (CSS keyframes)

- CTA button

Additional sections:

- Content reveals on scroll with stagger

- Consistent spacing and type scale

---