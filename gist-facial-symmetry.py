from math import hypot

# A tiny, model-free asymmetry detector from mirrored facial landmarks.
def asymmetry(points, pairs):
    score = 0.0
    details = []
    for a, b in pairs:
        d = hypot(points[a][0] - points[b][0],
                  points[a][1] - points[b][1])
        details.append(d)
        score += d
    return score / len(details) if details else 0.0

# Example: landmark pairs mirrored around the face centerline.
MIRRORS = [(33, 263), (61, 291), (13, 14), (78, 308),
           (105, 334), (159, 386), (145, 374)]

if __name__ == "__main__":
    face = {i: (0.5 + i * 0.001, 0.4 + i * 0.0003) for i in range(478)}
    print(f"symmetry error: {asymmetry(face, MIRRORS):.4f}")
