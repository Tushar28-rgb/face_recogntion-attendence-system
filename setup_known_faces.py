"""
One-time setup: copy images from the 'image' folder into 'known_faces'
so each person has a folder named after them (e.g. image/tushar.jpg -> known_faces/Tushar/tushar.jpg).
Run once: python setup_known_faces.py
"""
import os
import shutil

IMAGE_DIR = "image"
KNOWN_FACES_DIR = "known_faces"

def main():
    if not os.path.isdir(IMAGE_DIR):
        print(f"Folder '{IMAGE_DIR}' not found. Create it and add photos (e.g. tushar.jpg, sarthak.jpg).")
        return

    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    count = 0

    for fname in os.listdir(IMAGE_DIR):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        # Person name = filename without extension, capitalized (tushar -> Tushar)
        name = fname.rsplit(".", 1)[0].strip()
        if not name:
            continue
        person_dir = os.path.join(KNOWN_FACES_DIR, name.capitalize())
        os.makedirs(person_dir, exist_ok=True)
        src = os.path.join(IMAGE_DIR, fname)
        dst = os.path.join(person_dir, fname)
        if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
            shutil.copy2(src, dst)
            print(f"Added: {name.capitalize()} <- {fname}")
        count += 1

    if count == 0:
        print(f"No images found in '{IMAGE_DIR}'. Add .jpg/.png files (e.g. tushar.jpg).")
    else:
        print(f"Done. {count} image(s) in known_faces. Run: python main.py")

if __name__ == "__main__":
    main()
