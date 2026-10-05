#!/usr/bin/env python3
"""
Ball-on-Beam Mechanical System Animation with Random Trajectory
Shows the complete mechanical system: beam, connecting rod, crank, and ball motion
Using B_P_Traj.py for physics-based trajectory planning
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import sys
import os
from scipy.optimize import fsolve

# Add B_P_Traj.py to path
sys.path.append('C:\\Users\\m6nassar\\Desktop\\MTE380 Project')
from B_P_Traj import Trajectory, Constant_Acceleration_Velocity, Constant_Velocity, CA_CV_CA, CA_CA, CV_CA_CA_CA, CA_CV_CA_CA, CA_CA_CV_CA, CV_CA, CV_CA_CA, CA_CA_CA

# Physics constants
g = 9.81  # m/s²
max_angle_deg = 15  # Maximum beam angle in degrees
max_angle = np.radians(max_angle_deg)  # Convert to radians

# Calculate physics parameters as specified
a_max = g * np.sin(np.radians(15))  # g*sin(15°) = 2.539 m/s²
a_min = -a_max  # g*sin(-15°) = -2.539 m/s²
v_max = np.sqrt(2 * a_max * 0.1)  # sqrt(2*a_max*20cm) = 0.713 m/s (using 0.1m = 10cm as distance)
v_min = -v_max

# System geometry
beam_half_length = 0.1  # 10 cm beam half-length

# Mechanical system dimensions (in meters) - Using Test Kinematics parameters
crank_radius = 0.06      # 6 cm crank radius (l11)
connecting_rod_length = 0.08  # 8 cm connecting rod length (l12)
crank_offset_h = 0.06    # 6 cm vertical distance: crank center is h below beam pivot
crank_offset_w = 0.08    # 8 cm horizontal distance: crank center is w to the right of beam pivot

# Calculate absolute positions
beam_pivot_x = 0.0       # Beam pivot at origin horizontally
beam_pivot_y = 0.08      # Beam pivot height (8 cm above bottom)
crank_center_x = beam_pivot_x + crank_offset_w   # Crank center w to the right of pivot
crank_center_y = beam_pivot_y - crank_offset_h   # Crank center h below pivot

# Global arrays for trajectory data
plx = np.zeros(30)
pltt = np.zeros(30)
posx = np.zeros(30)
velx = np.zeros(30)
accx = np.zeros(30)

def init_trajectory_arrays():
    """Initialize the trajectory arrays"""
    global plx, pltt, posx, velx, accx
    plx = np.zeros(30)
    pltt = np.zeros(30)
    posx = np.zeros(30)
    velx = np.zeros(30)
    accx = np.zeros(30)

def inverse_kinematics(theta, l11, l12, w, h, l):
    """
    Inverse kinematics: Given beam angle θ, find crank angle θ11 and rod angle θ12
    Using analytical approach from slide 7
    
    Equation: (l₁₂² - l₁₁² - h² - w² - l²)/√(w²+h²) = 
             2l(sinα sinθ - cosα cosθ) + 
             2l₁₁(-sinα sinθ₁₁ + cosα cosθ₁₁) - 
             (2ll₁₁/√(w²+h²))cos(θ-θ₁₁)
    
    Where: sinα = h/√(w²+h²), cosα = w/√(w²+h²)
    Solve for θ₁₁ given θ
    """
    # Calculate α from geometric parameters
    alpha_denom = np.sqrt(w**2 + h**2)
    sin_alpha = h / alpha_denom
    cos_alpha = w / alpha_denom
    
    # Left side of equation (constant)
    lhs = (l12**2 - l11**2 - h**2 - w**2 - l**2) / alpha_denom
    
    # Known terms involving θ
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Define equation to solve for θ₁₁
    def equation_for_theta_11(theta_11):
        sin_theta_11 = np.sin(theta_11)
        cos_theta_11 = np.cos(theta_11)
        
        # Right side terms
        term1 = 2 * l * (sin_alpha * sin_theta - cos_alpha * cos_theta)
        term2 = 2 * l11 * (-sin_alpha * sin_theta_11 + cos_alpha * cos_theta_11)
        term3 = -(2 * l * l11 / alpha_denom) * np.cos(theta - theta_11)
        
        rhs = term1 + term2 + term3
        
        return lhs - rhs
    
    theta_11_guess = 0.0
    
    try:
        theta_11 = fsolve(equation_for_theta_11, theta_11_guess)[0]
        
        # Calculate θ12 (connecting rod angle)
        crank_end_x = w + l11 * np.cos(theta_11)
        crank_end_y = -h + l11 * np.sin(theta_11)
        beam_connection_x = l * np.cos(theta)
        beam_connection_y = l * np.sin(theta)
        
        dx_rod = beam_connection_x - crank_end_x
        dy_rod = beam_connection_y - crank_end_y
        theta_12 = np.arctan2(dy_rod, dx_rod)
        
        return theta_11, theta_12
    except:
        return 0.0, 0.0

def forward_kinematics(theta_11, l11, l12, w, h, l):
    """
    Forward kinematics: Given crank angle θ11, find beam angle θ
    Using analytical approach from slide 7 - same equation as inverse kinematics
    
    Equation: (l₁₂² - l₁₁² - h² - w² - l²)/√(w²+h²) = 
             2l(sinα sinθ - cosα cosθ) + 
             2l₁₁(-sinα sinθ₁₁ + cosα cosθ₁₁) - 
             (2ll₁₁/√(w²+h²))cos(θ-θ₁₁)
    
    Where: sinα = h/√(w²+h²), cosα = w/√(w²+h²)
    Solve for θ given θ₁₁
    """
    # Calculate α from geometric parameters
    alpha_denom = np.sqrt(w**2 + h**2)
    sin_alpha = h / alpha_denom
    cos_alpha = w / alpha_denom
    
    # Left side of equation (constant)
    lhs = (l12**2 - l11**2 - h**2 - w**2 - l**2) / alpha_denom
    
    # Known terms involving θ₁₁
    sin_theta_11 = np.sin(theta_11)
    cos_theta_11 = np.cos(theta_11)
    
    # Define equation to solve for θ
    def equation_for_theta(theta):
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Right side terms
        term1 = 2 * l * (sin_alpha * sin_theta - cos_alpha * cos_theta)
        term2 = 2 * l11 * (-sin_alpha * sin_theta_11 + cos_alpha * cos_theta_11)
        term3 = -(2 * l * l11 / alpha_denom) * np.cos(theta - theta_11)
        
        rhs = term1 + term2 + term3
        
        return lhs - rhs
    
    # Solve for θ using numerical method
    theta_guess = 0.0
    
    try:
        theta = fsolve(equation_for_theta, theta_guess)[0]
        return theta
    except:
        return 0.0

def generate_random_trajectory():
    """Generate a single random trajectory using B_P_Traj.py"""
    max_attempts = 50  # Limit attempts to avoid infinite loop

    for attempt in range(max_attempts):
        # Random start position and velocity within beam limits
        # Using the correct ±10 cm range as specified
        xb = random.uniform(-0.08, 0.08)  # ±8 cm (within ±10 cm beam half-length)
        vbx = random.uniform(-0.05, 0.05)   # ±5 cm/s (reasonable velocity)
        
        # Random destination position and velocity
        xd = random.uniform(-0.08, 0.08)  # ±8 cm (within ±10 cm beam half-length)
        vdx = random.uniform(-0.05, 0.05)   # ±5 cm/s (reasonable velocity)
        
        print(f"=== RANDOM TRAJECTORY GENERATION (Attempt {attempt+1}) ===")
        print(f"Start: xb = {xb:.3f} m ({xb*100:.1f} cm), vbx = {vbx:.3f} m/s ({vbx*100:.1f} cm/s)")
        print(f"Destination: xd = {xd:.3f} m ({xd*100:.1f} cm), vdx = {vdx:.3f} m/s ({vdx*100:.1f} cm/s)")
        
        # Call B_P_Traj trajectory function
        result_xb, result_xd, result_vbx, result_vdx, estop = Trajectory(
            xb, xd, vbx, vdx, a_max, a_min, v_max, v_min, 
            plx, pltt, posx, velx, accx
        )
        
        # Check if trajectory stays within beam bounds
        min_pos = min(plx)
        max_pos = max(plx)
        beam_limit = 0.1  # 10 cm limit (full beam half-length)
        
        if abs(min_pos) <= beam_limit and abs(max_pos) <= beam_limit:
            print(f"✓ Valid trajectory found! Position range: {min_pos*100:.1f} to {max_pos*100:.1f} cm")
            break
        else:
            print(f"✗ Trajectory out of bounds: {min_pos*100:.1f} to {max_pos*100:.1f} cm, retrying...")
            if attempt < max_attempts - 1:
                init_trajectory_arrays()  # Reset arrays only if we're actually retrying
    else:
        print("⚠ Warning: Could not generate valid trajectory within bounds, using last attempt (may exceed beam limits)")
    
    return {
        'start_pos': xb, 'start_vel': vbx,
        'dest_pos': xd, 'dest_vel': vdx,
        'final_pos': result_xb, 'final_vel': result_vdx,
        'estop': estop,
        'time': pltt.copy(), 'position': plx.copy(), 'velocity': velx.copy(), 'acceleration': accx.copy()
    }

class MechanicalSystemAnimation:
    def __init__(self, trajectory_data):
        self.trajectory = trajectory_data
        self.current_frame = 0
        self.total_frames = len(trajectory_data['time'])
        
        # Ball properties
        self.ball_radius = 0.008  # 8 mm ball radius
        
        # Setup figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_xlim(-0.15, 0.15)  # 15 cm on each side
        self.ax.set_ylim(-0.08, 0.12)  # From -8 cm to +12 cm
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('Horizontal Position (m)')
        self.ax.set_ylabel('Vertical Position (m)')
        self.ax.set_title('Ball-on-Beam Mechanical System Animation\nBeam, Connecting Rod, Crank, and Ball Motion', fontweight='bold')
        
        # Initialize drawing elements
        self.beam_line, = self.ax.plot([], [], 'b-', linewidth=4, label='Beam')
        self.connecting_rod, = self.ax.plot([], [], 'g-', linewidth=3, label='Connecting Rod')
        self.crank, = self.ax.plot([], [], 'r-', linewidth=3, label='Crank')
        self.ball_circle = plt.Circle((0, 0), self.ball_radius, color='orange', zorder=5)
        self.ax.add_patch(self.ball_circle)
        
        # Pivot points
        self.beam_pivot = plt.Circle((beam_pivot_x, beam_pivot_y), 0.003, color='black', zorder=6)
        self.crank_center = plt.Circle((crank_center_x, crank_center_y), 0.003, color='black', zorder=6)
        self.ax.add_patch(self.beam_pivot)
        self.ax.add_patch(self.crank_center)
        
        # Text displays
        self.time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, fontsize=10, 
                                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        self.physics_text = self.ax.text(0.02, 0.85, '', transform=self.ax.transAxes, fontsize=9,
                                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Legend
        self.ax.legend(loc='upper right')
        
        # Calculate crank angle for each time step
        self.crank_angles = self.calculate_crank_angles()
        
    def calculate_crank_angles(self):
        """Calculate crank angles from beam angles using inverse kinematics from Test Kinematics"""
        crank_angles = []
        
        # Convert mechanical parameters to cm for consistency with Test Kinematics
        l11_cm = crank_radius * 100      # 6 cm
        l12_cm = connecting_rod_length * 100  # 8 cm  
        l_cm = beam_half_length * 100    # 10 cm
        w_cm = crank_offset_w * 100      # 8 cm
        h_cm = crank_offset_h * 100      # 6 cm
        
        for i in range(len(self.trajectory['acceleration'])):
            # Beam angle from acceleration - CORRECTED SIGN
            # Positive acceleration means ball accelerates to the right, so beam tilts DOWN on the right
            # This requires a NEGATIVE beam angle (right side down)
            acc = self.trajectory['acceleration'][i]
            beam_angle = -np.arcsin(np.clip(acc / g, -1, 1))  # Negative sign to correct inclination
            
            # Use proper inverse kinematics to find crank angle
            theta_11, theta_12 = inverse_kinematics(beam_angle, l11_cm, l12_cm, w_cm, h_cm, l_cm)
            
            crank_angles.append(theta_11)
            
        return crank_angles
    
    def update_frame(self, frame):
        """Update animation frame"""
        if frame >= len(self.trajectory['time']):
            frame = len(self.trajectory['time']) - 1
            
        self.current_frame = frame
        
        # Get current trajectory data
        current_time = self.trajectory['time'][frame]
        ball_pos = self.trajectory['position'][frame]
        ball_vel = self.trajectory['velocity'][frame]
        acceleration = self.trajectory['acceleration'][frame]
        crank_angle = self.crank_angles[frame]
        
        # Calculate beam angle from acceleration - CORRECTED SIGN
        # Positive acceleration means ball accelerates to the right, so beam tilts DOWN on the right
        # This requires a NEGATIVE beam angle (right side down)
        beam_angle = -np.arcsin(np.clip(acceleration / g, -1, 1))
        beam_angle_deg = np.degrees(beam_angle)
        
        # Draw crank
        crank_tip_x = crank_center_x + crank_radius * np.cos(crank_angle)
        crank_tip_y = crank_center_y + crank_radius * np.sin(crank_angle)
        self.crank.set_data([crank_center_x, crank_tip_x], [crank_center_y, crank_tip_y])
        
        # Calculate right end of beam (where connecting rod attaches)
        beam_right_end_x = beam_pivot_x + beam_half_length * np.cos(beam_angle)
        beam_right_end_y = beam_pivot_y + beam_half_length * np.sin(beam_angle)
        
        # Draw connecting rod (from crank tip to beam right end)
        self.connecting_rod.set_data([crank_tip_x, beam_right_end_x], 
                                   [crank_tip_y, beam_right_end_y])
        
        # Draw beam (rotated around pivot)
        beam_half_len = beam_half_length
        beam_left_x = beam_pivot_x - beam_half_len * np.cos(beam_angle)
        beam_left_y = beam_pivot_y - beam_half_len * np.sin(beam_angle)
        beam_right_x = beam_pivot_x + beam_half_len * np.cos(beam_angle)
        beam_right_y = beam_pivot_y + beam_half_len * np.sin(beam_angle)
        self.beam_line.set_data([beam_left_x, beam_right_x], [beam_left_y, beam_right_y])
        
        # Update ball position (on the beam) - CORRECTED DIRECTION
        # When acceleration is positive, beam tilts so right side goes down
        # Ball should move toward the lower side (down the incline)
        # Position along beam is measured from center, positive = right side
        ball_x_on_beam = ball_pos  # This is the position along the beam from center
        ball_x = beam_pivot_x + ball_x_on_beam * np.cos(beam_angle)  # Global X position  
        ball_y = beam_pivot_y + ball_x_on_beam * np.sin(beam_angle)  # Global Y position
        self.ball_circle.set_center((ball_x, ball_y))
        
        # Update text displays
        time_str = f'Time: {current_time:.3f} s\nFrame: {frame+1}/{len(self.trajectory["time"])}'
        self.time_text.set_text(time_str)
        
        physics_str = f'Ball pos on beam: {ball_pos*100:.1f} cm\n'
        physics_str += f'Ball global pos: ({ball_x*100:.1f}, {ball_y*100:.1f}) cm\n'
        physics_str += f'Ball vel: {ball_vel*100:.1f} cm/s\n'
        physics_str += f'Acceleration: {acceleration:.2f} m/s²\n'
        physics_str += f'Beam angle: {beam_angle_deg:.1f}°\n'
        physics_str += f'Crank angle: {np.degrees(crank_angle):.1f}°'
        self.physics_text.set_text(physics_str)
        
        return [self.beam_line, self.connecting_rod, self.crank, self.ball_circle, 
                self.time_text, self.physics_text]
    
    def animate(self):
        """Start the animation"""
        # Calculate frame interval for real-time playback
        total_time = self.trajectory['time'][-1] - self.trajectory['time'][0]
        interval = (total_time / len(self.trajectory['time'])) * 1000  # ms per frame
        interval = max(50, min(interval, 200))  # Clamp between 50-200 ms
        
        self.ani = animation.FuncAnimation(
            self.fig, self.update_frame, frames=len(self.trajectory['time']),
            interval=interval, blit=False, repeat=True
        )
        
        plt.show()

def main():
    """Generate and visualize one random trajectory with full mechanical system"""
    print("Ball-on-Beam Mechanical System Animation")
    print("Using B_P_Traj.py for physics-based trajectory planning")
    print("=" * 60)
    print(f"Physics Setup:")
    print(f"Max beam angle: ±{max_angle_deg:.0f}°")
    print(f"Acceleration limits: ±{a_max:.3f} m/s²")
    print(f"Velocity limits: ±{v_max:.3f} m/s")
    print(f"Beam half-length: {beam_half_length*100:.0f} cm")
    print()
    print(f"Mechanical System (Test Kinematics Parameters):")
    print(f"Crank radius (l11): {crank_radius*100:.1f} cm")
    print(f"Connecting rod length (l12): {connecting_rod_length*100:.1f} cm")
    print(f"Beam half-length (l): {beam_half_length*100:.1f} cm")
    print(f"Crank offset from beam pivot: w={crank_offset_w*100:.1f} cm right, h={crank_offset_h*100:.1f} cm below")
    print(f"Beam pivot position: ({beam_pivot_x*100:.1f}, {beam_pivot_y*100:.1f}) cm")
    print(f"Crank center position: ({crank_center_x*100:.1f}, {crank_center_y*100:.1f}) cm")
    print()
    
    # Initialize arrays for this trajectory
    init_trajectory_arrays()
    
    # Generate trajectory
    traj = generate_random_trajectory()
    
    # Print results
    print("Trajectory result:")
    print(f"Final position: {traj['final_pos']:.4f} m (target: {traj['dest_pos']:.4f} m)")
    print(f"Final velocity: {traj['final_vel']:.4f} m/s (target: {traj['dest_vel']:.4f} m/s)")
    print(f"Emergency stop: {traj['estop']}")
    print(f"Position error: {abs(traj['final_pos'] - traj['dest_pos'])*1000:.2f} mm")
    print(f"Velocity error: {abs(traj['final_vel'] - traj['dest_vel'])*100:.2f} cm/s")
    
    # Trajectory statistics
    time_span = traj['time'][-1] - traj['time'][0]
    pos_range = (min(traj['position'])*100, max(traj['position'])*100)
    vel_range = (min(traj['velocity'])*100, max(traj['velocity'])*100)
    acc_range = (min(traj['acceleration']), max(traj['acceleration']))
    beam_angles = [-np.degrees(np.arcsin(np.clip(acc/g, -1, 1))) for acc in traj['acceleration']]  # Corrected sign
    angle_range = (min(beam_angles), max(beam_angles))
    
    print(f"Trajectory: {len(traj['position'])} points over {time_span:.2f} seconds")
    print(f"Position range: {pos_range[0]:.1f} to {pos_range[1]:.1f} cm")
    print(f"Velocity range: {vel_range[0]:.1f} to {vel_range[1]:.1f} cm/s")
    print(f"Acceleration range: {acc_range[0]:.2f} to {acc_range[1]:.2f} m/s²")
    print(f"Beam angle range: {angle_range[0]:.1f}° to {angle_range[1]:.1f}°")
    print()
    
    print("Starting mechanical system animation...")
    print("Animation shows: beam (blue), connecting rod (green), crank (red), ball (orange)")
    print("Close the animation window to exit.")
    
    # Create and start animation
    animator = MechanicalSystemAnimation(traj)
    animator.animate()
    
    print("=" * 60)
    print("Mechanical system animation complete!")

if __name__ == "__main__":
    main()
