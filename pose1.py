import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the live stream mode:
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('pose landmarker result: {}'.format(result))

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

with PoseLandmarker.create_from_options(options) as landmarker:
    # Send live image data to perform pose landmarking.
    # The results are accessible via the `result_callback` provided in
    # the `PoseLandmarkerOptions` object.
    # The pose landmarker must be created with the live stream mode.
    landmarker.detect_async(mp_image, frame_timestamp_ms)
