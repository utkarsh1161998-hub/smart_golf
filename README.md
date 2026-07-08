import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open video file or webcam stream
video_path = "golf_swing.mp4"  # Replace with your video file
cap = cv2.VideoCapture(video_path)

# Trajectory buffer to draw the wrist "contrail"
wrist_trajectory = []

print("Processing golf swing video... Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for consistent processing speed
    frame = cv2.resize(frame, (800, 600))
    
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Get coordinates for the left wrist (lead wrist for a right-handed golfer)
        # MediaPipe landmarks return normalized coordinates (0.0 to 1.0)
        lwrist_x = int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * frame.shape[1])
        lwrist_y = int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * frame.shape[0])
        
        # Append to trajectory history
        wrist_trajectory.append((lwrist_x, lwrist_y))
        
        # Simple Logic to detect the "Top of the Swing" 
        # (The point where the wrist reaches its minimum Y value / highest point on screen)
        if len(wrist_trajectory) > 10:
            y_coords = [p[1] for p in wrist_trajectory]
            highest_point_idx = np.argmin(y_coords)
            cv2.circle(frame, wrist_trajectory[highest_point_idx], 10, (0, 0, 255), -1)
            cv2.putText(frame, "TOP OF SWING", (wrist_trajectory[highest_point_idx][0] + 15, wrist_trajectory[highest_point_idx][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw wrist path contrail
        for i in range(1, len(wrist_trajectory)):
            cv2.line(frame, wrist_trajectory[i - 1], wrist_trajectory[i], (0, 255, 0), 2)

        # Draw the standard body skeleton overlay
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

    # Display result
    cv2.imshow('Smart Golf - Swing Analyzer', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import math
import matplotlib.pyplot as plt

def simulate_ball_flight(ball_speed_mph, launch_angle_deg, drag_coefficient=0.25):
    # Convert miles per hour to meters per second
    v0 = ball_speed_mph * 0.44704
    theta = math.radians(launch_angle_deg)
    
    # Constants
    g = 9.81           # Gravity (m/s^2)
    dt = 0.01          # Time step (seconds)
    
    # Initial state vectors
    x, y = 0.0, 0.0
    vx = v0 * math.cos(theta)
    vy = v0 * math.sin(theta)
    
    # Tracking coordinates
    x_trajectory = [x]
    y_trajectory = [y]
    
    while y >= 0:
        v = math.sqrt(vx**2 + vy**2)
        
        # Calculate deceleration due to air resistance (Acceleration = Force / mass)
        # Simplified mass-proportional deceleration factor for demonstrative math
        ax = -drag_coefficient * v * vx
        ay = -g - (drag_coefficient * v * vy)
        
        # Update velocities
        vx += ax * dt
        vy += ay * dt
        
        # Update positions
        x += vx * dt
        y += vy * dt
        
        if y >= 0:
            x_trajectory.append(x)
            y_trajectory.append(y)
            
    # Convert total distance back to yards for standard golf metrics
    total_carry_yards = x_trajectory[-1] * 1.09361
    print(f"Total Carry Distance: {total_carry_yards:.1f} Yards")
    
    return x_trajectory, y_trajectory

# Example parameters: Driver shot at 150 mph ball speed, 12-degree launch angle
x_coords, y_coords = simulate_ball_flight(ball_speed_mph=150, launch_angle_deg=12)

# Plotting the trajectory graph
plt.plot(x_coords, y_coords, label="Ball Path")
plt.title("Simulated Golf Ball Flight")
plt.xlabel("Distance (Meters)")
plt.ylabel("Height (Meters)")
plt.grid(True)
plt.show()
