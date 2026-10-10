"""Audit the identity database and print per-face SFace distances."""

import sys

from face_utils import (
    audit_encoding_database,
    load_encoded_faces,
    load_image,
    normalize_image_path,
    recognize_faces_with_attributes,
)


def main():
    audit_encoding_database()
    if len(sys.argv) < 2:
        print("\nUsage: python diagnose_recognition.py <image_path>")
        return 0

    image_path = normalize_image_path(sys.argv[1])
    if image_path is None:
        print("Image path does not exist or is not a file.")
        return 1
    image = load_image(image_path)
    encodings, names = load_encoded_faces()
    results = recognize_faces_with_attributes(
        image,
        encodings,
        names,
        diagnostics=True,
    )
    print("\n====================================")
    print("FACE RECOGNITION DIAGNOSTICS")
    print("====================================")
    print(f"Face detected: {bool(results)}")
    print(f"Number of faces: {len(results)}")
    for index, (box, name, distance, _) in enumerate(results, 1):
        print(f"Face {index}: box={box}")
        print(f"  Best match: {name}")
        print(f"  Best distance: {distance:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())