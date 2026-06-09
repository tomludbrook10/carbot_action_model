# carbot_action_model

The learning side of **carbot**, my self-driving RC car. I drove the car around my house
collecting camera footage + actions, then trained a neural net to look at the images and
predict where to go next — and it learned to navigate around the house on its own.

The model takes a short stack of recent camera frames (plus a depth map from a frozen
Depth-Anything-V2) and predicts a path of upcoming waypoints. It's trained as imitation
learning on the recorded runs, then exported to a TensorRT engine to run on the car.

**Stack:** PyTorch · Depth-Anything-V2 · transformers · imitation learning · TensorRT export

> Heavy stuff (datasets, model weights, the venv) is gitignored — this repo is the code.

## Part of the carbot project

- [carbot_drivetrain](https://github.com/tomludbrook10/carbot_drivetrain) — ESP32 drivetrain firmware
- [carbot_ws](https://github.com/tomludbrook10/carbot_ws) — the ROS2 brain that runs the live driving loop
- [carbot_inference](https://github.com/tomludbrook10/carbot_inference) — real-time camera→waypoints model (TensorRT)
- [carbot_teleoperation](https://github.com/tomludbrook10/carbot_teleoperation) — remote driving + data recording
