# Real-Time Grid Color Detector (Python / OpenCV)

This is a Python/OpenCV application that performs real-time detection of a 9x9 grid on a piece of paper using a live video stream (e.g., an IP camera).

The application applies a bird's-eye view correction to the detected grid and identifies the dominant color within each individual cell (Red, Green, Blue, etc.). It uses a temporal stability check to filter out noise from hand movements or shadows and saves the final state of the grid to a text file upon exit.

[Insert a GIF or screenshot of the project in action here. An image showing the "Camera Feed" and "Grid Analysis" windows side-by-side is ideal.]

## 🚀 Project Definition and Purpose

The primary goal of this project is to create an interactive system that bridges a physical grid with a digital environment. Even as the camera moves, the system automatically finds the grid, corrects its perspective, and analyzes the colors within its cells.

## ✨ Key Features

This project successfully meets all of the following requirements:

* **Real-Time Bird's-Eye View:** Instantly flattens (warps) the skewed grid perspective from a live video feed.
* **Automatic Grid Detection:** Automatically detects the predefined 9x9 grid structure on the paper and highlights it with a green bounding box.
* **Individual Cell Identification:** Identifies and isolates all 81 individual cells within the flattened grid for analysis.
* **Dominant Color Detection:** Detects the dominant color in each cell (based on HSV color space), identifying colors like Red, Green, Blue, Yellow, White, Black, etc.
* **Visual Feedback Overlay:** Overlays the detected color name (e.g., "RED") onto the corresponding cell in the "Grid Analysis" output window.
* **Stability Filtering:** Filters out transient changes (like a hand waving or a brief shadow) and only registers persistent color changes.
* **Save to File:** Saves the final detected color layout of the entire grid to a `.txt` file when the program quits.

## 🛠️ Technology Stack

* **Python 3.x**
* **OpenCV-Python** (Görüntü işleme kütüphanesi)
* **NumPy** (Matris işlemleri için)

---

## 🔧 Setup and Installation

This project uses Python and requires the OpenCV and NumPy libraries. It is highly recommended to use a virtual environment.

### 1. Create `requirements.txt`

Proje klasöründe (`.py` dosyanın yanında) `requirements.txt` adında bir dosya oluştur ve içine şunları ekle:

```txt
opencv-python
numpy
```

2. Clone & Install Dependencies
Clone this GitHub repository:

Bash

git clone [URL_OF_THIS_REPOSITORY]
cd [PROJECT_FOLDER_NAME]
Create a virtual environment:

Bash

python -m venv .venv
Activate the virtual environment:

Windows (PowerShell/CMD):

Bash

.\.venv\Scripts\activate
Linux/macOS:

Bash

source .venv/bin/activate
Install the required libraries from the requirements.txt file:

Bash

pip install -r requirements.txt

🖥️ How to Use
1. Physical Setup
Grid: Draw a 9x9 grid on a standard (e.g., A4) piece of paper. The grid should cover as much of the paper as possible (close to the edges).

Contrast: This is critical for automatic detection. Place the white paper on a dark, non-reflective surface (e.g., a dark green, black, or dark wood desk).

Lighting: Ensure good, diffuse lighting to prevent strong shadows from your hand or the camera.

2. Software Configuration
IP Camera: Install and run an IP camera app on your phone (e.g., "DroidCam" or "IP Webcam").

Video Source: Open your main Python file (e.g., ScanGrid.py) and update the VIDEO_SOURCE variable with the IP address from your phone's app:

Python

# ...
# !!! EDIT THIS !!!
VIDEO_SOURCE = "http://192.168.1.128:4747/video" # Set your camera's URL here
# ...
Run: With your virtual environment activated, run the script from your terminal:

Bash

python ScanGrid.py  # (or whatever your main .py file is named)
Controls
Kamera Akisi (Otomatik Algilama) (Camera Feed (Auto Detect)): Shows the raw camera feed with the green detection border overlaid.

Grid Analizi (Düzleştirilmiş) (Grid Analysis (Flattened)): Shows the flattened, bird's-eye view of the grid with the detected color names written on each cell.

Debug - Threshold: A black-and-white debug window showing what the detection algorithm "sees" to find the paper.

q or ESC: While focused on one of the app's windows, press 'q' or 'ESC' to quit the program and save the final grid state.

📄 Output
When the program is closed (via 'q' or 'ESC'), it saves a file named grid_sonuc.txt to the project's root folder (next to your .py file).

This file contains a neatly formatted table of the last known stable color for all 81 cells.
