"""
Main entry point for Face Recognition Application
Provides menu interface to encode faces, recognize from images, or use webcam

Usage:
    python main.py
"""

import os
import sys
import subprocess

from face_utils import normalize_image_path


def print_menu():
    """Display main menu"""
    print("\n" + "=" * 60)
    print("              FACE RECOGNITION SYSTEM")
    print("=" * 60)
    print("1. Encode new faces (from known_faces/ directory)")
    print("2. Recognize faces in an image")
    print("3. Recognize faces from webcam (real-time)")
    print("4. Play Emotion Challenge")
    print("5. View configuration")
    print("6. Exit")
    print("=" * 60)


def encode_menu():
    """Handle encoding faces"""
    print("\n[ENCODING FACES]")
    print("This will scan the 'known_faces' directory and create a database")
    print("of face encodings.\n")
    print("Directory structure should be:")
    print("  known_faces/")
    print("    ├── person1/")
    print("    │   ├── photo1.jpg")
    print("    │   ├── photo2.jpg")
    print("    │   └── ...")
    print("    ├── person2/")
    print("    │   ├── photo1.jpg")
    print("    │   └── ...")
    print("    └── ...\n")
    
    response = input("Continue with encoding? (yes/no): ").strip().lower()
    if response in ['yes', 'y']:
        print("Starting encoding process...\n")
        try:
            subprocess.run([sys.executable, "encode_faces.py"], check=True)
        except subprocess.CalledProcessError:
            print("Error: Encoding process failed")
        except FileNotFoundError:
            print("Error: encode_faces.py not found")
    else:
        print("Cancelled.")


def recognize_image_menu():
    """Handle image recognition"""
    print("\n[RECOGNIZE FACES IN IMAGE]")
    entered_path = input("Enter path to image file: ")
    image_path = normalize_image_path(entered_path)

    if image_path is None:
        print("Error: File not found or path is not a file")
        return
    
    print("Processing image...\n")
    try:
        subprocess.run([sys.executable, "recognize_image.py", image_path], check=True)
    except subprocess.CalledProcessError:
        print("Error: Image recognition failed")
    except FileNotFoundError:
        print("Error: recognize_image.py not found")


def recognize_webcam_menu():
    """Handle webcam recognition"""
    print("\n[RECOGNIZE FACES FROM WEBCAM]")
    print("Starting real-time face recognition from webcam...")
    print("Controls:")
    print("  ESC or 'q' - Exit")
    print("  Space - Pause/Resume")
    print()
    
    try:
        subprocess.run([sys.executable, "recognize_webcam.py"], check=True)
    except subprocess.CalledProcessError:
        pass  # User likely just exited normally
    except FileNotFoundError:
        print("Error: recognize_webcam.py not found")


def view_config_menu():
    """Display current configuration"""
    print("\n[CURRENT CONFIGURATION]")
    import config
    
    settings = [
        ("Known Faces Directory", config.KNOWN_FACES_DIR),
        ("Encodings File", config.ENCODINGS_FILE),
        ("Face Detection Model", config.MODEL),
        ("Tolerance (0-1)", config.TOLERANCE),
        ("Process Every N Frames", config.PROCESS_EVERY_N_FRAMES),
        ("Frame Size", f"{config.FRAME_WIDTH}x{config.FRAME_HEIGHT}"),
        ("Emotion Game Rounds", config.EMOTION_GAME_ROUNDS),
        ("Emotion Game Round Time", config.EMOTION_GAME_ROUND_TIME),
        ("Emotion Game Threshold", config.EMOTION_GAME_THRESHOLD),
    ]
    
    print("\nCurrent Settings:")
    for setting_name, value in settings:
        print(f"  {setting_name:.<40} {value}")
    
    print("\nTo change settings, edit config.py")


def emotion_challenge_menu():
    """Run the isolated emotion challenge game."""
    print("\n[EMOTION CHALLENGE]")
    try:
        from emotion_game import run_emotion_challenge
        run_emotion_challenge()
    except Exception as error:
        print(f"Error starting Emotion Challenge: {error}")


def main():
    """Main menu loop"""
    print("\n" + "=" * 60)
    print("Face Recognition App - Starting up...")
    print("=" * 60)
    
    # Check dependencies
    try:
        import cv2
        import imutils
        import onnxruntime
        print("✓ All dependencies found")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    print()
    
    while True:
        try:
            print_menu()
            choice = input("Select option (1-6): ").strip()
            
            if choice == '1':
                encode_menu()
            elif choice == '2':
                recognize_image_menu()
            elif choice == '3':
                recognize_webcam_menu()
            elif choice == '4':
                emotion_challenge_menu()
            elif choice == '5':
                view_config_menu()
            elif choice == '6':
                print("\nExiting Face Recognition App. Goodbye!")
                break
            else:
                print("Invalid option. Please enter 1-6.")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Returning to menu...")


if __name__ == "__main__":
    main()
