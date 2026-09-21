"""관측 클래스와 영상 크기. 클래스 순서는 학습과 추론에서 공유한다."""

SIGNAL_CLASSES = ("vehicle_signal", "pedestrian_signal")
ROADMARK_CLASSES = ("white_lane", "yellow_lane", "stop_line")
DEFAULT_IMAGE_HW = (608, 800)
ROADMARK_STRIDE = 4
