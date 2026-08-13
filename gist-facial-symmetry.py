from math import exp

# Minimal version of Neuro-Symmetry's bilateral feature extraction.
PAIRS = (
    (33,263),(7,249),(163,466),(144,373),(145,374),
    (153,380),(154,381),(155,382),(133,362),
    (160,387),(158,385),(157,384),(159,386),
    (70,300),(63,293),(105,334),(66,296),(107,336),
    (55,285),(65,295),(52,282),(53,283),(46,276),
    (49,279),(48,278),(115,344),(220,440),(45,275),
    (61,291),(57,287),(185,409),(84,314),
    (234,454),(227,447),(132,361),(58,288),(172,397),
    (21,251),(54,284),(103,332),
)

def symmetry_error(points):
    """Mirror right landmarks and measure bilateral geometric error."""
    distances = []
    for left, right in PAIRS:
        lx, ly, lz = points[left]
        rx, ry, rz = points[right]
        distances.append(((lx + rx)**2 + (ly - ry)**2 + (lz - rz)**2)**0.5)
    error = sum(distances) / len(distances)
    return error, exp(-error)  # 1.0 = symmetric

# MediaPipe Face Landmarker provides 478 normalized (x, y, z) landmarks.
face = {i: (0.0, 0.0, 0.0) for i in range(478)}
error, score = symmetry_error(face)
print(f"symmetry error={error:.4f} | score={score:.4f}")
