# Object Measurement Using Aruco Markers

## How It works

![result](C:\Users\1256276177\Desktop\result.png)

## Overview

This project detects objects in an image and measures their size using Aruco markers as a reference. Users can manually select areas in the image to measure dimensions or detect objects automatically.

## Features

- Uses Aruco markers for size calibration.
- Detects objects using adaptive thresholding and contour detection.
- Supports manual measurement using mouse clicks.
- Displays detected objects with bounding boxes and size annotations.

## Installation

### Prerequisites

Ensure you have Python installed along with the necessary dependencies.

### Install Dependencies

Run the following command to install required packages:

```
pip install opencv-python numpy
```

## Usage

### Run the Object Measurement Script

Use the following command:

```
python measure_mouse_click.py --side 15
```

- `--side 15` specifies the real-world side length (in cm) of the Aruco marker used for calibration.

### How It Works

1. The script reads an image containing an Aruco marker.
2. It detects the marker and calculates the pixel-to-cm ratio.
3. Users can click and drag to manually measure object dimensions.
4. The script also detects and marks objects automatically.

### Example

```
python measure_mouse_click.py --side 15
```

This sets the reference Aruco marker size to 15 cm.

## File Structure

```
project_folder/
│── images/
│   ├── image7.jpg
│   ├── image8.jpg
│── measure_mouse_click.py
│── objectDetector.py
│── README.md
```

## Notes

- Ensure the Aruco marker is clearly visible in the image.
- Adjust the `--side` argument based on the actual size of the marker.
- The script currently uses `DICT_5X5_100` Aruco markers.

## License

This project is open-source and available under the MIT License.
