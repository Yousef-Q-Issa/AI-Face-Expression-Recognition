"""Emotion Challenge game mode built on the existing face and emotion pipeline."""

import math
import random
import time
from collections import Counter, deque
from typing import Optional

import cv2
from imutils.video import VideoStream

import config
from face_utils import FaceEngine, VerifiedAttributeEngine

WINDOW_NAME = "Emotion Challenge"
START_BUTTON_RECT = (0, 0, 0, 0)
START_BUTTON_CLICKED = False
GAME_OVER_BUTTON_RECTS = {}
AI_BUTTON_RECTS = {}
BUTTON_ACTION = None


def _maximize_emotion_window():
    """Maximize the existing OpenCV window on Windows using the native API."""
    try:
        import win32con
        import win32gui
    except ImportError:
        return False

    hwnd = win32gui.FindWindow(None, WINDOW_NAME)
    if not hwnd:
        return False

    win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
    return True


EMOTION_ICONS = {
    "Neutral": "😐",
    "Happy": "😀",
    "Surprise": "😮",
    "Sad": "😢",
    "Angry": "😡",
    "Disgust": "🤢",
    "Fear": "😨",
    "Contempt": "😬",
}


def get_game_emotions():
    """Return only the allowed target emotions for the challenge."""
    allowed = ["Happy", "Sad", "Angry", "Neutral", "Surprise"]
    return [emotion for emotion in VerifiedAttributeEngine.EMOTIONS if emotion in allowed]


def choose_target_emotion(previous_emotion: Optional[str] = None):
    """Choose a different emotion than the previous target."""
    emotions = get_game_emotions()
    if not emotions:
        return "Happy"
    if previous_emotion is None:
        return random.choice(emotions)
    options = [emotion for emotion in emotions if emotion != previous_emotion]
    return random.choice(options) if options else random.choice(emotions)


def _parse_confidence(raw_value):
    if raw_value is None:
        return 0.0
    if isinstance(raw_value, (int, float)):
        value = float(raw_value)
        return value if 0.0 <= value <= 1.0 else value / 100.0
    text = str(raw_value).strip().replace("%", "")
    if not text:
        return 0.0
    try:
        value = float(text)
    except ValueError:
        return 0.0
    return value if 0.0 <= value <= 1.0 else value / 100.0


def _draw_centered_text(frame, text, y, font_scale=0.8, color=(255, 255, 255), thickness=2):
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = max(0, (frame.shape[1] - text_size[0]) // 2)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def _handle_start_screen_mouse(event, x, y, _flags, _param):
    global START_BUTTON_CLICKED, BUTTON_ACTION
    if event == cv2.EVENT_LBUTTONUP:
        left, top, right, bottom = START_BUTTON_RECT
        START_BUTTON_CLICKED = left <= x <= right and top <= y <= bottom
        for action, rect in {**GAME_OVER_BUTTON_RECTS, **AI_BUTTON_RECTS}.items():
            left, top, right, bottom = rect
            if left <= x <= right and top <= y <= bottom:
                BUTTON_ACTION = action


def _draw_button(frame, label, rect, color=(0, 210, 160), scale=1.0):
    left, top, right, bottom = rect
    cv2.rectangle(frame, (left, top), (right, bottom), (18, 30, 31), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2, cv2.LINE_AA)
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48 * scale, 1)
    cv2.putText(
        frame,
        label,
        (left + (right - left - text_size[0]) // 2, top + (bottom - top + text_size[1]) // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48 * scale,
        (235, 250, 247),
        1,
        cv2.LINE_AA,
    )


def _smooth_emotions(history):
    if not history:
        return {}
    combined = {emotion: 0.0 for emotion in VerifiedAttributeEngine.EMOTIONS[:8]}
    for prediction in history:
        for emotion, probability in prediction.items():
            combined[emotion] = combined.get(emotion, 0.0) + probability
    count = max(1, len(history))
    averaged = {emotion: value / count for emotion, value in combined.items()}
    total = sum(averaged.values())
    if total > 0:
        averaged = {emotion: value / total for emotion, value in averaged.items()}
    return averaged


def _render_start_screen(frame):
    global START_BUTTON_RECT

    height, width = frame.shape[:2]
    frame[:] = (10, 14, 18)

    # Lightweight booth backdrop: a restrained grid and accent line keep the screen from feeling empty.
    grid_color = (18, 30, 32)
    for x in range(0, width, max(40, width // 12)):
        cv2.line(frame, (x, 0), (x, height), grid_color, 1, cv2.LINE_AA)
    for y in range(0, height, max(40, height // 8)):
        cv2.line(frame, (0, y), (width, y), grid_color, 1, cv2.LINE_AA)
    cv2.line(frame, (width // 2 - min(150, width // 4), 70), (width // 2 + min(150, width // 4), 70), (0, 185, 150), 1, cv2.LINE_AA)

    scale = max(0.32, min(width / 900.0, height / 620.0))
    center_x = width // 2
    _draw_centered_text(frame, "IEEE", int(height * 0.13), font_scale=0.48 * scale, color=(0, 205, 165), thickness=1)
    _draw_centered_text(frame, "AI EMOTION LAB", int(height * 0.19), font_scale=0.56 * scale, color=(190, 205, 208), thickness=1)
    _draw_centered_text(frame, "EMOTION CHALLENGE", int(height * 0.27), font_scale=1.25 * scale, color=(245, 250, 250), thickness=2)
    _draw_centered_text(frame, "AI-POWERED FACIAL EXPRESSION CHALLENGE", int(height * 0.32), font_scale=0.42 * scale, color=(125, 155, 157), thickness=1)

    _draw_centered_text(frame, "Can you match the AI?", int(height * 0.39), font_scale=0.72 * scale, color=(235, 245, 243), thickness=1)
    _draw_centered_text(frame, "Make the facial expression shown on screen.", int(height * 0.43), font_scale=0.40 * scale, color=(160, 175, 177), thickness=1)

    emotions = (
        ("HA", "HAPPY", (0, 210, 220)),
        ("SA", "SAD", (220, 150, 70)),
        ("AN", "ANGRY", (60, 80, 230)),
        ("NE", "NEUTRAL", (170, 170, 170)),
        ("SU", "SURPRISE", (190, 90, 220)),
    )
    gap = max(4, int(10 * scale))
    card_width = max(45, min(int(88 * scale), (width - 24 - 4 * gap) // 5))
    card_height = max(38, int(62 * scale))
    cards_width = len(emotions) * card_width + (len(emotions) - 1) * gap
    cards_left = max(8, center_x - cards_width // 2)
    cards_top = int(height * 0.47)
    for index, (badge, label, color) in enumerate(emotions):
        left = cards_left + index * (card_width + gap)
        right = left + card_width
        panel = frame.copy()
        cv2.rectangle(panel, (left, cards_top), (right, cards_top + card_height), (22, 30, 33), -1)
        cv2.addWeighted(panel, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (left, cards_top), (right, cards_top + card_height), color, 1, cv2.LINE_AA)
        cv2.circle(frame, (left + card_width // 2, cards_top + int(card_height * 0.35)), max(8, int(11 * scale)), color, -1, cv2.LINE_AA)
        badge_size, _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.25 * scale, 1)
        cv2.putText(frame, badge, (left + card_width // 2 - badge_size[0] // 2, cards_top + int(card_height * 0.35) + badge_size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.25 * scale, (20, 25, 26), 1, cv2.LINE_AA)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.28 * scale, 1)
        cv2.putText(frame, label, (left + card_width // 2 - label_size[0] // 2, cards_top + card_height - int(9 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.28 * scale, (220, 230, 230), 1, cv2.LINE_AA)

    info_top = int(height * 0.66)
    info_width = max(58, int(105 * scale))
    info_height = max(36, int(54 * scale))
    info_gap = max(8, int(14 * scale))
    info_left = center_x - (3 * info_width + 2 * info_gap) // 2
    info = (("5", "ROUNDS"), ("5s", "PER ROUND"), ("AI", "REAL-TIME"))
    for index, (value, label) in enumerate(info):
        left = info_left + index * (info_width + info_gap)
        panel = frame.copy()
        cv2.rectangle(panel, (left, info_top), (left + info_width, info_top + info_height), (20, 28, 31), -1)
        cv2.addWeighted(panel, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (left, info_top), (left + info_width, info_top + info_height), (45, 75, 75), 1, cv2.LINE_AA)
        value_size, _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, 1)
        cv2.putText(frame, value, (left + info_width // 2 - value_size[0] // 2, info_top + int(22 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (0, 220, 170), 1, cv2.LINE_AA)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.24 * scale, 1)
        cv2.putText(frame, label, (left + info_width // 2 - label_size[0] // 2, info_top + info_height - int(8 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.24 * scale, (170, 185, 185), 1, cv2.LINE_AA)

    button_width = max(150, int(210 * scale))
    button_height = max(38, int(48 * scale))
    button_left = center_x - button_width // 2
    button_top = min(height - button_height - 34, int(height * 0.80))
    START_BUTTON_RECT = (button_left, button_top, button_left + button_width, button_top + button_height)
    pulse = 0.5 + 0.5 * math.sin(time.time() * 3.0)
    button_color = (0, int(175 + 35 * pulse), int(135 + 30 * pulse))
    cv2.rectangle(frame, (button_left, button_top), (button_left + button_width, button_top + button_height), (18, 34, 35), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (button_left, button_top), (button_left + button_width, button_top + button_height), button_color, 2, cv2.LINE_AA)
    _draw_centered_text(frame, "START GAME", button_top + int(button_height * 0.64), font_scale=0.55 * scale, color=(235, 250, 247), thickness=1)
    _draw_centered_text(frame, "ENTER to start    |    Q / ESC to return", min(height - 10, button_top + button_height + int(25 * scale)), font_scale=0.30 * scale, color=(125, 145, 147), thickness=1)

    return START_BUTTON_RECT


def _render_countdown(frame, round_number, countdown_value):
    frame[:] = (15, 15, 15)
    _draw_centered_text(frame, f"ROUND {round_number} / {config.EMOTION_GAME_ROUNDS}", 80, font_scale=0.8, color=(0, 255, 0), thickness=2)
    _draw_centered_text(frame, "Get ready...", 120, font_scale=0.8, color=(255, 255, 255), thickness=2)
    if countdown_value == "GO":
        _draw_centered_text(frame, "GO!", 240, font_scale=2.2, color=(0, 255, 0), thickness=3)
    else:
        _draw_centered_text(frame, str(countdown_value), 240, font_scale=2.4, color=(0, 255, 0), thickness=3)


def _draw_face_box(frame, box, color):
    if not box:
        return
    x, y, width, height = box
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 1, cv2.LINE_AA)


def _draw_emotion_badge(frame, emotion, center):
    colors = {
        "Happy": (0, 210, 220),
        "Sad": (220, 150, 70),
        "Angry": (60, 80, 230),
        "Neutral": (170, 170, 170),
        "Surprise": (190, 90, 220),
    }
    cv2.circle(frame, center, 18, colors.get(emotion, (180, 180, 180)), -1, cv2.LINE_AA)
    badge_text = emotion[:2].upper()
    text_size, _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(
        frame,
        badge_text,
        (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )


def _draw_timer(frame, timer_value):
    height, width = frame.shape[:2]
    center = (width - 62, 58)
    radius = max(22, min(width, height) // 18)
    fraction = max(0.0, min(1.0, timer_value / config.EMOTION_GAME_ROUND_TIME))
    cv2.circle(frame, center, radius, (80, 80, 80), 2, cv2.LINE_AA)
    cv2.ellipse(frame, center, (radius, radius), -90, 0, int(360 * fraction), (0, 220, 140), 3, cv2.LINE_AA)
    timer_text = f"{timer_value:.1f}s"
    text_size, _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(
        frame,
        timer_text,
        (center[0] - text_size[0] // 2, center[1] + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _render_playing(frame, target_emotion, detected_emotion, confidence, timer_value, round_number, score, face_count=0, debug_probabilities=None, predicted_index=None):
    alpha_overlay = frame.copy()
    cv2.rectangle(alpha_overlay, (12, 12), (frame.shape[1] - 12, 78), (18, 18, 18), -1)
    cv2.addWeighted(alpha_overlay, 0.35, frame, 0.65, 0, frame)

    cv2.putText(frame, "EMOTION CHALLENGE", (28, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"ROUND {round_number} / {config.EMOTION_GAME_ROUNDS}", (28, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    for index in range(config.EMOTION_GAME_ROUNDS):
        color = (0, 220, 140) if index < score else (110, 110, 110)
        cv2.circle(frame, (frame.shape[1] // 2 - 36 + index * 18, 50), 4, color, -1, cv2.LINE_AA)

    target_panel = frame.copy()
    cv2.rectangle(target_panel, (18, 96), (142, 190), (18, 18, 18), -1)
    cv2.addWeighted(target_panel, 0.30, frame, 0.70, 0, frame)
    cv2.putText(frame, "TARGET", (30, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    _draw_emotion_badge(frame, target_emotion, (112, 135))
    cv2.putText(frame, target_emotion.upper(), (30, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    _draw_timer(frame, timer_value)

    score_panel = frame.copy()
    cv2.rectangle(score_panel, (18, frame.shape[0] - 74), (150, frame.shape[0] - 16), (18, 18, 18), -1)
    cv2.addWeighted(score_panel, 0.30, frame, 0.70, 0, frame)
    cv2.putText(frame, "SCORE", (30, frame.shape[0] - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{score} / {config.EMOTION_GAME_ROUNDS}", (30, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 140), 1, cv2.LINE_AA)

    right_panel = frame.copy()
    cv2.rectangle(right_panel, (frame.shape[1] - 190, frame.shape[0] - 88), (frame.shape[1] - 18, frame.shape[0] - 16), (18, 18, 18), -1)
    cv2.addWeighted(right_panel, 0.30, frame, 0.70, 0, frame)
    cv2.putText(frame, "AI DETECTS", (frame.shape[1] - 176, frame.shape[0] - 64), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, detected_emotion.upper(), (frame.shape[1] - 176, frame.shape[0] - 39), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{confidence * 100:.0f}%", (frame.shape[1] - 176, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 140), 1, cv2.LINE_AA)

    if face_count > 1:
        cv2.putText(frame, "ONE PLAYER", (frame.shape[1] // 2 - 45, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 220), 1, cv2.LINE_AA)
    elif detected_emotion == "FACE NOT DETECTED":
        cv2.putText(frame, "LOOK AT THE CAMERA", (frame.shape[1] // 2 - 72, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)

    if getattr(config, "DEBUG_MODE", False) and target_emotion == "Sad" and debug_probabilities:
        debug_y = 106
        debug_x = max(8, frame.shape[1] - 150)
        cv2.putText(frame, "TARGET: SAD", (debug_x, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
        cv2.putText(frame, "MODEL PREDICTIONS", (debug_x, debug_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (255, 255, 255), 1)
        for offset, emotion in enumerate(("Sad", "Neutral", "Happy", "Angry", "Surprise")):
            probability = debug_probabilities.get(emotion, 0.0)
            text = f"{emotion}: {probability * 100:.0f}%"
            cv2.putText(frame, text, (debug_x, debug_y + 32 + offset * 13), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1)
        predicted_text = detected_emotion if detected_emotion != "FACE NOT DETECTED" else "Unavailable"
        cv2.putText(frame, f"Predicted class: {predicted_text}", (debug_x, debug_y + 101), cv2.FONT_HERSHEY_SIMPLEX, 0.24, (200, 200, 200), 1)
        index_text = "-" if predicted_index is None else str(predicted_index)
        cv2.putText(frame, f"Predicted index: {index_text}", (debug_x, debug_y + 126), cv2.FONT_HERSHEY_SIMPLEX, 0.24, (200, 200, 200), 1)


def _render_status_message(frame, title, subtitle, color=(0, 255, 0)):
    overlay = frame.copy()
    center_y = frame.shape[0] // 2
    cv2.rectangle(overlay, (frame.shape[1] // 2 - 160, center_y - 55), (frame.shape[1] // 2 + 160, center_y + 55), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    _draw_centered_text(frame, title, center_y - 8, font_scale=0.85, color=color, thickness=2)
    for index, line in enumerate(str(subtitle).splitlines()):
        _draw_centered_text(frame, line, center_y + 28 + index * 18, font_scale=0.55, color=(255, 255, 255), thickness=1)


def _render_game_over(frame, score):
    global GAME_OVER_BUTTON_RECTS
    frame[:] = (15, 15, 15)
    _draw_centered_text(frame, "🎉 GAME OVER", 90, font_scale=1.4, color=(0, 255, 0), thickness=3)
    _draw_centered_text(frame, f"{score} / {config.EMOTION_GAME_ROUNDS}", 160, font_scale=1.4, color=(255, 255, 255), thickness=2)
    percent = (score / config.EMOTION_GAME_ROUNDS) * 100
    _draw_centered_text(frame, f"{percent:.0f}% SUCCESS", 210, font_scale=0.9, color=(200, 200, 200), thickness=2)
    if score >= int(config.EMOTION_GAME_ROUNDS * 0.7):
        _draw_centered_text(frame, "GREAT JOB!", 260, font_scale=1.0, color=(0, 255, 0), thickness=2)
    elif score >= 3:
        _draw_centered_text(frame, "GOOD EFFORT!", 260, font_scale=1.0, color=(255, 255, 0), thickness=2)
    else:
        _draw_centered_text(frame, "KEEP PRACTICING!", 260, font_scale=1.0, color=(255, 255, 255), thickness=2)
    width, height = frame.shape[1], frame.shape[0]
    button_width = min(220, max(130, width // 3))
    button_height = 42
    gap = 14
    left = width // 2 - (button_width * 2 + gap) // 2
    top = min(height - button_height - 36, 320)
    GAME_OVER_BUTTON_RECTS = {
        "PLAY_AGAIN": (left, top, left + button_width, top + button_height),
        "MAIN_MENU": (left + button_width + gap, top, left + button_width * 2 + gap, top + button_height),
    }
    _draw_button(frame, "PLAY AGAIN", GAME_OVER_BUTTON_RECTS["PLAY_AGAIN"], (0, 170, 130))
    _draw_button(frame, "MAIN MENU", GAME_OVER_BUTTON_RECTS["MAIN_MENU"], (120, 145, 145))
    _draw_centered_text(frame, "PLAY: R    MENU: Q / ESC", top + button_height + 34, font_scale=0.52, color=(150, 165, 165), thickness=1)


def _render_ai_prediction(frame, phase, age, gender, gender_confidence, error=False):
    global AI_BUTTON_RECTS
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (24, 18), (width - 24, min(height - 18, 218)), (12, 18, 19), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    if phase == "INTRO":
        _draw_centered_text(frame, "🤖 ONE MORE THING...", 70, font_scale=0.75, color=(0, 230, 165), thickness=2)
        _draw_centered_text(frame, "Let's see what the AI", 112, font_scale=0.48, color=(235, 240, 240), thickness=1)
        _draw_centered_text(frame, "can predict about you!", 136, font_scale=0.48, color=(235, 240, 240), thickness=1)
        AI_BUTTON_RECTS = {}
        return

    if phase == "COUNTDOWN":
        _draw_centered_text(frame, "🤖 AI ANALYSIS", 55, font_scale=0.62, color=(0, 230, 165), thickness=2)
        _draw_centered_text(frame, "GET READY...", 82, font_scale=0.45, color=(220, 230, 230), thickness=1)
        _draw_centered_text(frame, str(age), height // 2 + 28, font_scale=2.4, color=(245, 250, 250), thickness=3)
        _draw_centered_text(frame, "I will estimate your AGE + GENDER", height - 30, font_scale=0.36, color=(165, 180, 180), thickness=1)
        AI_BUTTON_RECTS = {}
        return

    if phase == "ANALYZING":
        _draw_centered_text(frame, "🤖 AI PREDICTION", 65, font_scale=0.72, color=(0, 230, 165), thickness=2)
        dots = "." * (int(time.time() * 4) % 4)
        _draw_centered_text(frame, f"ANALYZING{dots}", height // 2 + 20, font_scale=0.75, color=(0, 220, 150), thickness=2)
        _draw_centered_text(frame, "Please face the camera", height // 2 + 52, font_scale=0.43, color=(190, 200, 200), thickness=1)
        AI_BUTTON_RECTS = {}
        return

    if error:
        _draw_centered_text(frame, "AI PREDICTION UNAVAILABLE", height // 2, font_scale=0.72, color=(80, 190, 220), thickness=2)
    else:
        compact = height < 360
        age_label_y = 74 if compact else int(height * 0.38)
        age_value_y = 96 if compact else int(height * 0.49)
        gender_label_y = 119 if compact else int(height * 0.60)
        gender_value_y = 143 if compact else int(height * 0.70)
        confidence_y = 160 if compact else int(height * 0.77)
        disclaimer_y = 180 if compact else int(height * 0.84)
        _draw_centered_text(frame, "ESTIMATED AGE", age_label_y, font_scale=0.40 if compact else 0.44, color=(170, 185, 185), thickness=1)
        _draw_centered_text(frame, str(age), age_value_y, font_scale=0.90 if compact else 1.15, color=(245, 250, 250), thickness=2)
        _draw_centered_text(frame, "PREDICTED GENDER", gender_label_y, font_scale=0.40 if compact else 0.44, color=(170, 185, 185), thickness=1)
        _draw_centered_text(frame, str(gender).upper(), gender_value_y, font_scale=0.70 if compact else 0.9, color=(0, 220, 150), thickness=2)
        if gender_confidence:
            _draw_centered_text(frame, f"Gender confidence: {gender_confidence}", confidence_y, font_scale=0.30 if compact else 0.36, color=(180, 195, 195), thickness=1)
        _draw_centered_text(frame, "AI estimate - may be inaccurate", disclaimer_y, font_scale=0.30 if compact else 0.36, color=(145, 160, 160), thickness=1)

    button_height = 40
    gap = 14
    top = height - button_height - (4 if height < 360 else 20)
    if error:
        button_width = min(220, max(150, width // 4))
        left = width // 2 - (button_width * 2 + gap) // 2
        AI_BUTTON_RECTS = {
            "PLAY_AGAIN": (left, top, left + button_width, top + button_height),
            "MAIN_MENU": (left + button_width + gap, top, left + button_width * 2 + gap, top + button_height),
        }
        _draw_button(frame, "PLAY AGAIN", AI_BUTTON_RECTS["PLAY_AGAIN"], (0, 190, 145))
        _draw_button(frame, "MAIN MENU", AI_BUTTON_RECTS["MAIN_MENU"], (120, 145, 145))
    else:
        button_width = min(145, max(92, (width - 40) // 2))
        left = width // 2 - (button_width * 2 + gap) // 2
        AI_BUTTON_RECTS = {
            "PLAY_AGAIN": (left, top, left + button_width, top + button_height),
            "MAIN_MENU": (left + button_width + gap, top, left + button_width * 2 + gap, top + button_height),
        }
        _draw_button(frame, "PLAY AGAIN", AI_BUTTON_RECTS["PLAY_AGAIN"], (0, 220, 150))
        _draw_button(frame, "MAIN MENU", AI_BUTTON_RECTS["MAIN_MENU"], (120, 145, 145))


def _render_ai_feedback(frame):
    global AI_BUTTON_RECTS
    frame[:] = (15, 15, 15)
    _draw_centered_text(frame, "THANK YOU FOR YOUR FEEDBACK", 150, font_scale=0.78, color=(235, 245, 243), thickness=2)
    width, height = frame.shape[1], frame.shape[0]
    button_width, button_height, gap = min(220, max(150, width // 4)), 42, 16
    left = width // 2 - (button_width * 2 + gap) // 2
    top = min(height - button_height - 30, 220)
    AI_BUTTON_RECTS = {
        "PLAY_AGAIN": (left, top, left + button_width, top + button_height),
        "MAIN_MENU": (left + button_width + gap, top, left + button_width * 2 + gap, top + button_height),
    }
    _draw_button(frame, "PLAY AGAIN", AI_BUTTON_RECTS["PLAY_AGAIN"], (0, 220, 150))
    _draw_button(frame, "MAIN MENU", AI_BUTTON_RECTS["MAIN_MENU"], (120, 145, 145))


def run_emotion_challenge():
    """Run the emotion challenge game until the user leaves the mode."""
    global START_BUTTON_CLICKED, BUTTON_ACTION

    print("Initializing Emotion Challenge...")
    face_engine = FaceEngine()
    attribute_engine = getattr(__import__("face_utils", fromlist=["get_attribute_engine"]), "get_attribute_engine")()
    print(f"Loaded emotion model: {list(attribute_engine.providers.keys())}")

    try:
        stream = VideoStream(
            src=0,
            resolution=(config.FRAME_WIDTH, config.FRAME_HEIGHT),
            framerate=config.CAMERA_FPS,
        ).start()
        time.sleep(1.0)
        first_frame = stream.read()
        if first_frame is None:
            print("Failed to read initial Emotion Challenge frame.")
            return False
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, _handle_start_screen_mouse)
        cv2.imshow(WINDOW_NAME, first_frame)
        _maximize_emotion_window()
    except Exception as error:
        print(f"Error starting Emotion Challenge webcam: {error}")
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except Exception:
            pass
        return False

    state = "IDLE"
    round_number = 1
    score = 0
    target_emotion = None
    previous_emotion = None
    detection_text = "FACE NOT DETECTED"
    confidence = 0.0
    streak = 0
    countdown_value = 3
    start_time = time.time()
    round_deadline = 0.0
    transition_until = 0.0
    emotion_history = deque(maxlen=6)
    debug_probabilities = {}
    predicted_index = None
    ai_started = 0.0
    ai_samples = []
    ai_age = "Unavailable"
    ai_gender = "Unavailable"
    ai_gender_confidence = ""
    ai_error = False
    game_over_started = 0.0

    try:
        while True:
            frame = stream.read()
            if frame is None:
                print("Failed to read frame from webcam.")
                break

            key = cv2.waitKey(1) & 0xFF
            action = BUTTON_ACTION
            BUTTON_ACTION = None

            if state == "IDLE":
                _render_start_screen(frame)
                start_clicked = START_BUTTON_CLICKED
                if start_clicked:
                    START_BUTTON_CLICKED = False
                if key == 13 or start_clicked:
                    state = "COUNTDOWN"
                    round_number = 1
                    score = 0
                    previous_emotion = None
                    target_emotion = choose_target_emotion(previous_emotion)
                    start_time = time.time()
                    countdown_value = 3
                    detection_text = "FACE NOT DETECTED"
                    streak = 0
                    confidence = 0.0
                    emotion_history.clear()
                    debug_probabilities = {}
                    predicted_index = None
                elif key in (27, ord("q")):
                    break

            elif state == "COUNTDOWN":
                elapsed = time.time() - start_time
                if elapsed >= 3.1:
                    state = "PLAYING"
                    round_deadline = time.time() + config.EMOTION_GAME_ROUND_TIME
                    detection_text = "FACE NOT DETECTED"
                    streak = 0
                    confidence = 0.0
                    emotion_history.clear()
                    debug_probabilities = {}
                    predicted_index = None
                else:
                    if elapsed < 1.0:
                        countdown_value = 3
                    elif elapsed < 2.0:
                        countdown_value = 2
                    elif elapsed < 3.0:
                        countdown_value = 1
                    else:
                        countdown_value = "GO"
                    _render_countdown(frame, round_number, countdown_value)

            elif state == "PLAYING":
                timer_remaining = max(0.0, round_deadline - time.time())
                faces = face_engine.detect(frame)
                face_box = None
                face_count = len(faces)
                if len(faces) == 0:
                    detection_text = "FACE NOT DETECTED"
                    streak = 0
                    confidence = 0.0
                    emotion_history.clear()
                    debug_probabilities = {}
                    predicted_index = None
                    predicted_index = None
                else:
                    face = max(faces, key=lambda item: item["box"][2] * item["box"][3])
                    face_box = face["box"]
                    attrs = attribute_engine.predict(frame, face["box"])
                    detected_emotion = attrs.get("emotion", "Unavailable")
                    predicted_index = attrs.get("emotion_index")
                    confidence = _parse_confidence(attrs.get("emotion_confidence", 0.0))
                    probabilities = attrs.get("emotion_probabilities", {})
                    if probabilities:
                        emotion_history.append(probabilities)
                        smoothed = _smooth_emotions(emotion_history)
                        if smoothed:
                            detection_text = max(smoothed, key=smoothed.get)
                            confidence = smoothed.get(detection_text, 0.0)
                            debug_probabilities = probabilities
                            if detection_text == target_emotion and confidence >= config.EMOTION_GAME_THRESHOLD:
                                streak += 1
                            else:
                                streak = 0
                        else:
                            detection_text = detected_emotion if detected_emotion != "Unavailable" else "FACE NOT DETECTED"
                            debug_probabilities = {}
                    else:
                        if detected_emotion == "Unavailable":
                            detection_text = "FACE NOT DETECTED"
                            streak = 0
                        else:
                            detection_text = detected_emotion
                            if detected_emotion == target_emotion and confidence >= config.EMOTION_GAME_THRESHOLD:
                                streak += 1
                            else:
                                streak = 0
                        debug_probabilities = {}

                    if detected_emotion == "Unavailable":
                        detection_text = "FACE NOT DETECTED"
                        streak = 0
                        emotion_history.clear()
                        debug_probabilities = {}
                        predicted_index = None

                if streak >= config.EMOTION_GAME_REQUIRED_STREAK:
                    state = "SUCCESS"
                    score += 1
                    previous_emotion = target_emotion
                    transition_until = time.time() + config.EMOTION_GAME_SUCCESS_DELAY
                elif timer_remaining <= 0.0:
                    state = "FAILURE"
                    transition_until = time.time() + config.EMOTION_GAME_FAILURE_DELAY
                    previous_emotion = target_emotion

                box_color = (0, 220, 140) if detection_text == target_emotion else (180, 180, 180)
                _draw_face_box(frame, face_box, box_color)
                _render_playing(frame, target_emotion, detection_text, confidence, timer_remaining, round_number, score, face_count, debug_probabilities, predicted_index)

            elif state in ("SUCCESS", "FAILURE"):
                if state == "SUCCESS":
                    _render_status_message(frame, "✅ SUCCESS!", f"{target_emotion.upper()} MATCHED\n+1 POINT", color=(0, 255, 0))
                else:
                    _render_status_message(frame, "⏰ TIME'S UP!", "+0 POINTS", color=(255, 255, 0))
                if time.time() >= transition_until:
                    if round_number >= config.EMOTION_GAME_ROUNDS:
                        state = "GAME_OVER"
                        game_over_started = time.time()
                    else:
                        round_number += 1
                        target_emotion = choose_target_emotion(previous_emotion)
                        state = "COUNTDOWN"
                        start_time = time.time()
                        detection_text = "FACE NOT DETECTED"
                        confidence = 0.0
                        streak = 0
                        emotion_history.clear()
                        debug_probabilities = {}
                        predicted_index = None

            elif state == "GAME_OVER":
                _render_game_over(frame, score)
                if key in (27, ord("q"), ord("Q")):
                    break
                if action == "MAIN_MENU" or key in (ord("q"), ord("Q")):
                    break
                if time.time() - game_over_started >= config.GAME_OVER_RESULT_TIME:
                    state = "AI_INTRO"
                    ai_started = time.time()
                    ai_samples = []
                    ai_age = "Unavailable"
                    ai_gender = "Unavailable"
                    ai_gender_confidence = ""
                    ai_error = False
                elif action == "PLAY_AGAIN" or key in (ord("r"), ord("R")):
                    round_number = 1
                    score = 0
                    target_emotion = choose_target_emotion(None)
                    previous_emotion = None
                    transition_until = 0.0
                    round_deadline = 0.0
                    detection_text = "FACE NOT DETECTED"
                    confidence = 0.0
                    streak = 0
                    emotion_history.clear()
                    debug_probabilities = {}
                    predicted_index = None
                    state = "COUNTDOWN"
                    start_time = time.time()
                    countdown_value = 3

            elif state == "AI_INTRO":
                _render_ai_prediction(frame, "INTRO", ai_age, ai_gender, ai_gender_confidence)
                if time.time() - ai_started >= config.AI_INTRO_TIME:
                    state = "AI_COUNTDOWN"
                    ai_started = time.time()

            elif state == "AI_COUNTDOWN":
                elapsed = time.time() - ai_started
                if elapsed >= config.AI_COUNTDOWN_TIME:
                    state = "AI_ANALYZING"
                    ai_started = time.time()
                    ai_samples = []
                else:
                    countdown_value = max(1, math.ceil(config.AI_COUNTDOWN_TIME - elapsed))
                    _render_ai_prediction(frame, "COUNTDOWN", countdown_value, ai_gender, ai_gender_confidence)

            elif state == "AI_ANALYZING":
                faces = face_engine.detect(frame)
                if faces:
                    face = max(faces, key=lambda item: item["box"][2] * item["box"][3])
                    attrs = attribute_engine.predict(frame, face["box"])
                    if len(ai_samples) < config.AI_STABILIZATION_SAMPLES:
                        ai_samples.append(attrs)
                if time.time() - ai_started >= config.AI_ANALYZING_TIME:
                    ages = [sample.get("age") for sample in ai_samples if sample.get("age") not in (None, "Unavailable", "N/A")]
                    genders = [sample.get("gender") for sample in ai_samples if sample.get("gender") not in (None, "Unavailable", "N/A")]
                    if ages:
                        ai_age = Counter(ages).most_common(1)[0][0]
                    if genders:
                        ai_gender = Counter(genders).most_common(1)[0][0]
                    confidence_values = [_parse_confidence(sample.get("gender_confidence")) for sample in ai_samples]
                    confidence_values = [value for value in confidence_values if value > 0]
                    if confidence_values:
                        ai_gender_confidence = f"{sum(confidence_values) / len(confidence_values) * 100:.0f}%"
                    ai_error = not ages or not genders
                    state = "AI_RESULT"
                _render_ai_prediction(frame, "ANALYZING", ai_age, ai_gender, ai_gender_confidence)

            elif state == "AI_RESULT":
                _render_ai_prediction(frame, "RESULT", ai_age, ai_gender, ai_gender_confidence, error=ai_error)
                if action == "PLAY_AGAIN" or key in (ord("r"), ord("R")):
                    round_number = 1
                    score = 0
                    target_emotion = choose_target_emotion(None)
                    previous_emotion = None
                    transition_until = 0.0
                    round_deadline = 0.0
                    detection_text = "FACE NOT DETECTED"
                    confidence = 0.0
                    streak = 0
                    emotion_history.clear()
                    debug_probabilities = {}
                    predicted_index = None
                    state = "COUNTDOWN"
                    start_time = time.time()
                    countdown_value = 3
                elif action == "MAIN_MENU" or key in (27, ord("q"), ord("Q")):
                    break

            if key in (27, ord("q"), ord("Q")) and state not in ("IDLE", "GAME_OVER"):
                break

            cv2.imshow(WINDOW_NAME, frame)

    except KeyboardInterrupt:
        print("Emotion Challenge interrupted by user.")
    finally:
        stream.stop()
        cv2.destroyWindow(WINDOW_NAME)
        print("Emotion Challenge closed.")

    return True


def main():
    run_emotion_challenge()


if __name__ == "__main__":
    main()
