# Robot Video Processing Pipeline
## Step 1: Read the Video
The robot recordings are stored in a special compressed format called **Zarr** (similar to a ZIP file for video frames).  
Code is used to read these files frame by frame, extracting:
- The **colour image**
- The **depth image** (which shows how far away each object is)
---
## Step 2: Find the Interesting Moments
Instead of sending every single frame to the AI (which would be very slow), the system detects moments when the robot pauses between actions.
Examples:
- The brief stop between **reaching** and **grasping**
This is done by:
- Measuring motion in each frame using **optical flow**
These pauses act as **natural boundaries between actions**.
---
## Step 3: Ask the AI What is Happening
A few representative frames from each action segment are sent to an AI model, along with a text description of the task.
The AI:
- Analyzes the images
- Returns a label (e.g., `"grasp"` or `"place"`)
- Provides a **confidence score**
### Models Tested:
- `openchat:7b`
- `llama3.3`
- `qwen3-vl` *(the only model capable of actually seeing the images)*
---
## Step 4: Save the Results
All results are saved in a structured **JSON file**, a simple text format that is easy for other programs to read.
### Example Output (Task: *Put apple in bowl*)
The system correctly identified three actions:
- **Reach** (0–19s)
- **Grasp** (28–57s)
- **Place** (67–76s)
