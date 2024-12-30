import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
from toppra import parametrizer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

### SETUP WITH CONSTRAINTS ON JOINT VELOCITIES, ACCELERATIONS AND TORQUES.
### WE CAN SET WAYPOINTS IN JOINT SPACE, INTERPOLATES AND SOLVES THE TOPP PROBLEM.
### THEN WE PLOT THE CARTESIAN RESULTANT PATH USING FORWARD KINEMATICS.

ta.setup_logging("INFO")

# Define physical parameters
L1, L2, L3 = 1, 1, 1  # Link lengths
m1, m2, m3 = 1, 1, 1  # Link masses
I1, I2, I3 = 0.1, 0.1, 0.1  # Moments of inertia
g = 9.81  # Gravitational acceleration
dof = 3

# Forward kinematics for 3R planar robot
def forward_kinematics_3R(qs, L1=1, L2=1, L3=1):
    x0, y0 = 0, 0  # Base (fixed)
    
    # First joint
    x1 = L1 * np.cos(qs[0])
    y1 = L1 * np.sin(qs[0])
    
    # Second joint
    x2 = x1 + L2 * np.cos(qs[0] + qs[1])
    y2 = y1 + L2 * np.sin(qs[0] + qs[1])
    
    # End effector (third joint)
    x3 = x2 + L3 * np.cos(qs[0] + qs[1] + qs[2])
    y3 = y2 + L3 * np.sin(qs[0] + qs[1] + qs[2])
    
    return np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])

def inverse_dynamics_3R(q, qd, qdd):
    """Compute the inverse dynamics for a 3R planar robot."""
    
    # Extract joint angles, velocities, and accelerations
    q1, q2, q3 = q
    q1d, q2d, q3d = qd
    q1dd, q2dd, q3dd = qdd
    
    # Mass matrix M(q)
    M11 = I1 + I2 + I3 + m2 * L1**2 + m3 * (L1**2 + L2**2 + 2 * L1 * L2 * np.cos(q2)) + m3 * L3**2
    M12 = I2 + I3 + m3 * (L2**2 + L1 * L2 * np.cos(q2)) + m3 * L3**2
    M13 = I3 + m3 * L3**2
    M21 = M12
    M22 = I2 + I3 + m3 * L2**2 + m3 * L3**2
    M23 = I3 + m3 * L3**2
    M31 = M13
    M32 = M23
    M33 = I3 + m3 * L3**2
    M = np.array([[M11, M12, M13],
                  [M21, M22, M23],
                  [M31, M32, M33]])
    
    # Coriolis and centrifugal matrix C(q, qd)
    C11 = -m3 * L1 * L2 * np.sin(q2) * q2d - m3 * L1 * L2 * np.sin(q2) * (q2d + q3d)
    C12 = -m3 * L1 * L2 * np.sin(q2) * q3d
    C13 = 0
    C21 = m3 * L1 * L2 * np.sin(q2) * q1d
    C22 = 0
    C23 = 0
    C31 = 0
    C32 = 0
    C33 = 0
    C = np.array([[C11, C12, C13],
                  [C21, C22, C23],
                  [C31, C32, C33]])
    
    # Gravity vector G(q)
    G1 = (m1 * L1 / 2 + m2 * L1 + m3 * L1) * g * np.cos(q1) + m2 * L2 / 2 * g * np.cos(q1 + q2) + m3 * L3 * g * np.cos(q1 + q2 + q3)
    G2 = m2 * L2 / 2 * g * np.cos(q1 + q2) + m3 * L3 * g * np.cos(q1 + q2 + q3)
    G3 = m3 * L3 * g * np.cos(q1 + q2 + q3)
    G = np.array([G1, G2, G3])
    
    # Inverse dynamics: tau = M(q) * qdd + C(q, qd) * qd + G(q)
    tau = np.dot(M, qdd) + np.dot(C, qd) + G
    
    return tau

import numpy as np

def inverse_kinematics_3R(x, y, L1, L2, L3, phi):
    """
    Calculate the inverse kinematics for a 3R planar robot.
    x, y: End-effector position.
    phi: End-effector orientation (angle in radians).
    L1, L2, L3: Lengths of the robot's links.
    """

    # Solve for q1 and q2 (same as 2R robot, but with the end-effector position corrected)
    xe = x - L3 * np.cos(phi)  # x position of the wrist (end of second link)
    ye = y - L3 * np.sin(phi)  # y position of the wrist

    # Inverse kinematics for 2R planar robot (the first two links)
    c2 = (xe**2 + ye**2 - L1**2 - L2**2) / (2 * L1 * L2)
    #s2 = np.sqrt(1 - c2**2)  # positive elbow configuration (elbow up)
    s2 = -np.sqrt(1 - c2**2)  # negative elbow configuration (elbow down)
    
    q2 = np.arctan2(s2, c2)
    q1 = np.arctan2(ye, xe) - np.arctan2(L2 * s2, L1 + L2 * c2)

    # Solve for q3
    q3 = phi - q1 - q2

    return q1, q2, q3

N_samples=3 # number of waypoints
n = 2 # number of joints
dof=3 # task dof

#ss = np.linspace(0, 1, N_samples)

#way_pts_cartesian = np.array([[1.5, 1],[1.5,-1]])
#way_pts_cartesian = np.array([[2, 0],[1.5, 0.125], [1, 1]])


ss = np.linspace(0, 1, 100)
ss_L = np.linspace(0, 0.5, 100)

#CUSTOM CUBIC PATH
def generate_cubic(): # Esempio 1.1 FEDE
    points= []
    for i in range(len(ss)):
        x = 2-ss[i]
        y = ss[i]**3
        points.append([x, y])
    return points

#LINEAR PATH
def generate_vertical_line(x0, y0):# Esempio 2.1 FEDE
    points= []
    for i in range(len(ss)): 
        x = x0 
        y = y0 - ss[i]
        points.append([x, y])

    return points

def generate_vertical_line_reversed(x0, y0):# Esempio 2.1 FEDE
    points= []
    ss_reversed = np.linspace(0, 0.7, 100)
    for i in range(len(ss)): 
        x = x0 
        y = y0 + ss_reversed[i]   
        points.append([x, y])

    return points

def generate_vertical_question(x0, y0):# Esempio 2.1 FEDE
    points= []
    ss_vertical_question = np.linspace(0, 0.3, 100)
    for i in range(len(ss)): 
        x = x0 
        y = y0 - ss_vertical_question[i]
        points.append([x, y])

    return points

def generate_horizontal_line(x0, y0):# Esempio 2.1 FEDE
    points= []
    for i in range(len(ss_L)): 
        x = x0 + ss_L[i]
        y = y0 
        points.append([x, y])

    return points

#CIRCULAR PATH
def generate_circle(x0, y0, radius):
    points = []
    #x0 = 1
    #y0 = 0.8
    #radius = 0.3
    theta = np.linspace(0, 2*np.pi, 100)
    for i in range(len(theta)):
        x = x0 + radius * np.cos(theta[i])
        y = y0 + radius * np.sin(theta[i])
        points.append([x, y])

    return points 

# ELLIPSOID
def generate_ellipse(x0, y0, a, b):
    points = []
    theta = np.linspace(0, 2*np.pi, 100)  # Angle parameter from 0 to 2π

    for i in range(len(theta)):
        x = x0 + a * np.cos(theta[i])  # Parametrization for the x-coordinate
        y = y0 + b * np.sin(theta[i])  # Parametrization for the y-coordinate
        points.append([x, y])

    return points

# LINEAR WITH SLOPE
def generate_line_with_slope(slope, intercept, x_start, x_end):
    points = []
    x_values = np.linspace(x_start, x_end, 100)
    for x in x_values:
        y = slope * x + intercept  # y = mx + c (line equation)
        points.append([x, y])
    return points

# QUESTION MARK
def generate_question_mark(x0, y0, a, b):
    points = []
    
    # Elliptical arc (top part) evolving clockwise and starting at (x0, y0)
    theta_arc = np.linspace(np.pi, 0, 100)  # Angle range for counterclockwise arc (bottom-left to bottom-right)
    for theta in theta_arc:
        x = x0 + a * (1 + np.cos(theta))  # Shifted to start at x0
        y = y0 + b * np.sin(theta)       # Aligned with y0
        points.append([x, y])
    
    return np.array(points)


waypoints_fist_1 = generate_line_with_slope(slope=1, intercept=2, x_start=-0.8, x_end=-0.5)
waypoints_line1 = generate_vertical_line(x0 = -0.5  , y0 = 1.5) #finisce in (0,0.5)
intra1 = generate_line_with_slope(slope=3.5, intercept=2.25, x_start=-0.5, x_end=-0.3)
waypoints_second_1 = generate_line_with_slope(slope=1, intercept=1.5, x_start=-0.3, x_end=0.0)
waypoints_line2 = generate_vertical_line(x0 = 0.0, y0 = 1.5)
intra2 = generate_line_with_slope(slope=0.625, intercept=0.5, x_start=0.0, x_end=0.8)
waypoints_circle = generate_ellipse(x0=0.5, y0=1, a=0.3, b=0.5)
intra3 = generate_line_with_slope(slope=2.5, intercept=-1, x_start=0.8, x_end=1)
waypoints_line3 = generate_vertical_line(x0= 1.0, y0=1.5)
waypoints_line4 = generate_horizontal_line(x0=1.0 , y0=0.5)
intra4 = generate_vertical_line_reversed(x0=1.5, y0= 0.5)
waypoints_question = generate_question_mark(x0=1.5, y0=1.2, a=0.3, b=0.3)
waypoints_slope_question = generate_line_with_slope(slope=2/3, intercept=-0.2, x_start=2.1, x_end=1.8)
waypoints_vertical_question = generate_vertical_question(x0=1.8, y0=1)
intra5 = generate_line_with_slope(slope=-2, intercept=4.3, x_start=1.8, x_end=1.875)
waypoints_circle_question = generate_circle(x0=1.8, y0=0.55, radius=0.075)



path_first_1 = ta.SplineInterpolator(ss, waypoints_fist_1)
path1 = ta.SplineInterpolator(ss, waypoints_line1)
path_intra1 = ta.SplineInterpolator(ss, intra1)
path_second_1 = ta.SplineInterpolator(ss, waypoints_second_1)
path2 = ta.SplineInterpolator(ss, waypoints_line2)
path_intra2 = ta.SplineInterpolator(ss, intra2)
path3 = ta.SplineInterpolator(ss, waypoints_circle)
path_intra3 = ta.SplineInterpolator(ss, intra3)
path4 = ta.SplineInterpolator(ss, waypoints_line3)
path5 = ta.SplineInterpolator(ss_L, waypoints_line4)
path_intra4 = ta.SplineInterpolator(ss, intra4)
path6 = ta.SplineInterpolator(ss, waypoints_question)
path7 = ta.SplineInterpolator(ss, waypoints_slope_question)
path8 = ta.SplineInterpolator(ss, waypoints_vertical_question)
path_intra5 = ta.SplineInterpolator(ss, intra5)
path9 = ta.SplineInterpolator(ss, waypoints_circle_question)

### INSTANCE FIRST PIECE FIRST 1
cart_instance_first_1 = algo.TOPPRA([], path_first_1)
cart_traj_first_1 = cart_instance_first_1.compute_trajectory(0,0)

# path sampling
ts_cart = np.linspace(0, cart_traj_first_1.duration, 100)
ps_sample_first_1 = cart_traj_first_1(ts_cart)  # cartesian path samples

joint_way_pts_first_1 = []
for i in range(len(ps_sample_first_1)):
    q1, q2, q3 = inverse_kinematics_3R(ps_sample_first_1[i,0], ps_sample_first_1[i, 1], L1, L2, L3, phi=0.5)
    joint_way_pts_first_1.append([q1, q2, q3])


ss_joint_first_1 = np.linspace(0, 1, len(joint_way_pts_first_1))

jnt_path_first_1 = ta.SplineInterpolator(ss, joint_way_pts_first_1)

vlims_first_1 = 10 + np.random.rand(dof) * 20
alims_first_1 = 10 + np.random.rand(dof) * 2

pc_vel_first_1 = constraint.JointVelocityConstraint(vlims_first_1)
pc_acc_first_1 = constraint.JointAccelerationConstraint(alims_first_1)



instance_jnt_first_1 = algo.TOPPRA([pc_vel_first_1, pc_acc_first_1], jnt_path_first_1) #, parametrizer="ParametrizeConstAccel"
jnt_traj_first_1 = instance_jnt_first_1.compute_trajectory(0,0)

#print("DURATION OF OPTIMAL TRAJECTORY IS:", jnt_traj1.duration)

ts_sample_first_1 = np.linspace(0, jnt_traj_first_1.duration, 100)
qs_sample_first_1 = jnt_traj_first_1(ts_sample_first_1)

### INSTANCE PATH 1
cart_instance1 = algo.TOPPRA([], path1)
cart_traj1 = cart_instance1.compute_trajectory(0,0)

# path sampling
ts_cart = np.linspace(0, cart_traj1.duration, 100)
ps_sample1 = cart_traj1(ts_cart)  # cartesian path samples

joint_way_pts1 = []
for i in range(len(ps_sample1)):
    q1, q2, q3 = inverse_kinematics_3R(ps_sample1[i,0], ps_sample1[i, 1], L1, L2, L3, phi=0.5)
    joint_way_pts1.append([q1, q2, q3])


ss_joint1 = np.linspace(0, 1, len(joint_way_pts1))

jnt_path1 = ta.SplineInterpolator(ss, joint_way_pts1)

vlims1 = 10 + np.random.rand(dof) * 20
alims1 = 10 + np.random.rand(dof) * 2

pc_vel1 = constraint.JointVelocityConstraint(vlims1)
pc_acc1 = constraint.JointAccelerationConstraint(alims1)



instance_jnt1 = algo.TOPPRA([pc_vel1, pc_acc1], jnt_path1) #, parametrizer="ParametrizeConstAccel"
jnt_traj1 = instance_jnt1.compute_trajectory(0,0)

#print("DURATION OF OPTIMAL TRAJECTORY IS:", jnt_traj1.duration)

ts_sample1 = np.linspace(0, jnt_traj1.duration, 100)
qs_sample1 = jnt_traj1(ts_sample1)


### INSTANCE INTRA PATH 1
cart_intra1 = algo.TOPPRA([], path_intra1)
cart_intra1 = cart_intra1.compute_trajectory(0,0)

# path sampling
ts_intra1 = np.linspace(0, cart_intra1.duration, 100)
ps_intra1 = cart_intra1(ts_intra1)  # cartesian path samples

joint_way_intra1 = []
for i in range(len(ps_intra1)):
    q1, q2, q3 = inverse_kinematics_3R(ps_intra1[i,0], ps_intra1[i, 1], L1, L2, L3, phi=0.5)
    joint_way_intra1.append([q1, q2, q3])


ss_intra1 = np.linspace(0, 1, len(joint_way_intra1))

jnt_intra1 = ta.SplineInterpolator(ss, joint_way_intra1)

vlims_intra1 = 10 + np.random.rand(dof) * 20
alims_intra1 = 10 + np.random.rand(dof) * 2

pc_vel_intra1 = constraint.JointVelocityConstraint(vlims_intra1)
pc_acc_intra1 = constraint.JointAccelerationConstraint(alims_intra1)



instance_intra1 = algo.TOPPRA([pc_vel_intra1, pc_acc_intra1], jnt_intra1) #, parametrizer="ParametrizeConstAccel"
jnt_intra1 = instance_intra1.compute_trajectory(0,0)

#print("DURATION OF OPTIMAL TRAJECTORY IS:", jnt_traj1.duration)

ts_sample_intra1 = np.linspace(0, jnt_intra1.duration, 100)
qs_sample_intra1 = jnt_intra1(ts_sample_intra1)

### INSTANCE FIRST PIECE SECOND 1
cart_instance_second_1 = algo.TOPPRA([], path_second_1)
cart_traj_second_1 = cart_instance_second_1.compute_trajectory(0,0)

# path sampling
ts_cart = np.linspace(0, cart_traj_second_1.duration, 100)
ps_sample_second_1 = cart_traj_second_1(ts_cart)  # cartesian path samples

joint_way_pts_second_1 = []
for i in range(len(ps_sample_second_1)):
    q1, q2, q3 = inverse_kinematics_3R(ps_sample_second_1[i,0], ps_sample_second_1[i, 1], L1, L2, L3, phi=0.5)
    joint_way_pts_second_1.append([q1, q2, q3])


ss_joint_second_1 = np.linspace(0, 1, len(joint_way_pts_second_1))

jnt_path_second_1 = ta.SplineInterpolator(ss, joint_way_pts_second_1)

vlims_second_1 = 10 + np.random.rand(dof) * 20
alims_second_1 = 10 + np.random.rand(dof) * 2

pc_vel_second_1 = constraint.JointVelocityConstraint(vlims_second_1)
pc_acc_second_1 = constraint.JointAccelerationConstraint(alims_second_1)



instance_jnt_second_1 = algo.TOPPRA([pc_vel_second_1, pc_acc_second_1], jnt_path_second_1) #, parametrizer="ParametrizeConstAccel"
jnt_traj_second_1 = instance_jnt_second_1.compute_trajectory(0,0)

#print("DURATION OF OPTIMAL TRAJECTORY IS:", jnt_traj1.duration)

ts_sample_second_1 = np.linspace(0, jnt_traj_second_1.duration, 100)
qs_sample_second_1 = jnt_traj_second_1(ts_sample_second_1)

### INSTANCE PATH 2
cart_instance2 = algo.TOPPRA([], path2)
cart_traj2 = cart_instance2.compute_trajectory(0,0)

# path sampling
ts_cart2 = np.linspace(0, cart_traj2.duration, 100)
ps_sample2 = cart_traj2(ts_cart2)  # cartesian path samples

joint_way_pts2 = []
for i in range(len(ps_sample2)):
    q1, q2, q3 = inverse_kinematics_3R(ps_sample2[i,0], ps_sample2[i, 1], L1, L2, L3, phi = 0.5)
    joint_way_pts2.append([q1, q2, q3])


ss_joint2 = np.linspace(0, 1, len(joint_way_pts2))

jnt_path2 = ta.SplineInterpolator(ss, joint_way_pts2)

vlims2 = 10 + np.random.rand(dof) * 20
alims2 = 10 + np.random.rand(dof) * 2

pc_vel2 = constraint.JointVelocityConstraint(vlims2)
pc_acc2 = constraint.JointAccelerationConstraint(alims2)



instance_jnt2 = algo.TOPPRA([pc_vel2, pc_acc2], jnt_path2) #, parametrizer="ParametrizeConstAccel"
jnt_traj2 = instance_jnt2.compute_trajectory(0,0)

#print("DURATION OF OPTIMAL TRAJECTORY IS:", jnt_traj1.duration)

ts_sample2 = np.linspace(0, jnt_traj2.duration, 100)
qs_sample2 = jnt_traj2(ts_sample2)

### INSTANCE INTRA PATH 2
cart_intra2 = algo.TOPPRA([], path_intra2)
cart_intra2 = cart_intra2.compute_trajectory(0,0)

# path sampling
ts_intra2 = np.linspace(0, cart_intra2.duration, 100)
ps_intra2 = cart_intra2(ts_intra2)  # cartesian path samples

joint_way_intra2 = []
for i in range(len(ps_intra2)):
    q1, q2, q3 = inverse_kinematics_3R(ps_intra2[i,0], ps_intra2[i, 1], L1, L2, L3, phi=0.5)
    joint_way_intra2.append([q1, q2, q3])


ss_intra2 = np.linspace(0, 1, len(joint_way_intra2))

jnt_intra2 = ta.SplineInterpolator(ss, joint_way_intra2)

vlims_intra2 = 10 + np.random.rand(dof) * 20
alims_intra2 = 10 + np.random.rand(dof) * 2

pc_vel_intra2 = constraint.JointVelocityConstraint(vlims_intra2)
pc_acc_intra2 = constraint.JointAccelerationConstraint(alims_intra2)



instance_intra2 = algo.TOPPRA([pc_vel_intra2, pc_acc_intra2], jnt_intra2) #, parametrizer="ParametrizeConstAccel"
jnt_intra2 = instance_intra2.compute_trajectory(0,0)

#print("DURATION OF OPTIMAL TRAJECTORY IS:", jnt_traj1.duration)

ts_sample_intra2 = np.linspace(0, jnt_intra2.duration, 100)
qs_sample_intra2 = jnt_intra2(ts_sample_intra2)


### INSTANCE PATH 3
cart_instance3 = algo.TOPPRA([], path3)
cart_traj3 = cart_instance3.compute_trajectory(0,0)

# path sampling
ts_cart3 = np.linspace(0, cart_traj3.duration, 100)
ps_sample3 = cart_traj3(ts_cart3)  # cartesian path samples

joint_way_pts3 = []
for i in range(len(ps_sample3)):
    q1, q2, q3 = inverse_kinematics_3R(ps_sample3[i,0], ps_sample3[i, 1], L1, L2, L3, phi=0.5)
    joint_way_pts3.append([q1, q2, q3])


ss_joint3 = np.linspace(0, 1, len(joint_way_pts3))

jnt_path3 = ta.SplineInterpolator(ss, joint_way_pts3)

vlims3 = 10 + np.random.rand(dof) * 20
alims3 = 10 + np.random.rand(dof) * 2

pc_vel3 = constraint.JointVelocityConstraint(vlims3)
pc_acc3 = constraint.JointAccelerationConstraint(alims3)



instance_jnt3 = algo.TOPPRA([pc_vel3, pc_acc3], jnt_path3) #, parametrizer="ParametrizeConstAccel"
jnt_traj3 = instance_jnt3.compute_trajectory(0,0)

#print("DURATION OF OPTIMAL TRAJECTORY IS:", jnt_traj1.duration)

ts_sample3 = np.linspace(0, jnt_traj3.duration, 100)
qs_sample3 = jnt_traj3(ts_sample3)

### INSTANCE INTRA PATH 3
cart_intra3 = algo.TOPPRA([], path_intra3)
cart_intra3 = cart_intra3.compute_trajectory(0,0)

# path sampling
ts_intra3= np.linspace(0, cart_intra3.duration, 100)
ps_intra3 = cart_intra3(ts_intra3)  # cartesian path samples

joint_way_intra3 = []
for i in range(len(ps_intra3)):
    q1, q2, q3 = inverse_kinematics_3R(ps_intra3[i,0], ps_intra3[i, 1], L1, L2, L3, phi=0.5)
    joint_way_intra3.append([q1, q2, q3])


ss_intra3 = np.linspace(0, 1, len(joint_way_intra3))

jnt_intra3 = ta.SplineInterpolator(ss, joint_way_intra3)

vlims_intra3 = 10 + np.random.rand(dof) * 20
alims_intra3 = 10 + np.random.rand(dof) * 2

pc_vel_intra3 = constraint.JointVelocityConstraint(vlims_intra3)
pc_acc_intra3 = constraint.JointAccelerationConstraint(alims_intra3)



instance_intra3 = algo.TOPPRA([pc_vel_intra3, pc_acc_intra3], jnt_intra3) #, parametrizer="ParametrizeConstAccel"
jnt_intra3 = instance_intra3.compute_trajectory(0,0)

#print("DURATION OF OPTIMAL TRAJECTORY IS:", jnt_traj1.duration)

ts_sample_intra3 = np.linspace(0, jnt_intra3.duration, 100)
qs_sample_intra3 = jnt_intra3(ts_sample_intra3)

### INSTANCE PATH 4
cart_instance4 = algo.TOPPRA([], path4)
cart_traj4 = cart_instance4.compute_trajectory(0,0)

# path sampling
ts_cart4 = np.linspace(0, cart_traj4.duration, 100)
ps_sample4 = cart_traj4(ts_cart4)  # cartesian path samples

joint_way_pts4 = []
for i in range(len(ps_sample4)):
    q1, q2, q3 = inverse_kinematics_3R(ps_sample4[i,0], ps_sample4[i, 1], L1, L2, L3, phi=0.5)
    joint_way_pts4.append([q1, q2, q3])


ss_joint4 = np.linspace(0, 1, len(joint_way_pts4))

jnt_path4 = ta.SplineInterpolator(ss, joint_way_pts4)

vlims4 = 10 + np.random.rand(dof) * 20
alims4 = 10 + np.random.rand(dof) * 2

pc_vel4 = constraint.JointVelocityConstraint(vlims4)
pc_acc4 = constraint.JointAccelerationConstraint(alims4)



instance_jnt4 = algo.TOPPRA([pc_vel4, pc_acc4], jnt_path4) #, parametrizer="ParametrizeConstAccel"
jnt_traj4 = instance_jnt4.compute_trajectory(0,0)

#print("DURATION OF OPTIMAL TRAJECTORY IS:", jnt_traj1.duration)

ts_sample4 = np.linspace(0, jnt_traj4.duration, 100)
qs_sample4 = jnt_traj4(ts_sample4)

### INSTANCE PATH 5
cart_instance5 = algo.TOPPRA([], path5)
cart_traj5 = cart_instance5.compute_trajectory(0,0)

# path sampling
ts_cart5 = np.linspace(0, cart_traj5.duration, 100)
ps_sample5 = cart_traj5(ts_cart5)  # cartesian path samples

joint_way_pts5 = []
for i in range(len(ps_sample5)):
    q1, q2, q3 = inverse_kinematics_3R(ps_sample5[i,0], ps_sample5[i, 1], L1, L2, L3, phi = 0.5)
    joint_way_pts5.append([q1, q2, q3])


ss_joint5 = np.linspace(0, 1, len(joint_way_pts5))

jnt_path5 = ta.SplineInterpolator(ss, joint_way_pts5)

vlims5 = 10 + np.random.rand(dof) * 20
alims5 = 10 + np.random.rand(dof) * 2

pc_vel5 = constraint.JointVelocityConstraint(vlims5)
pc_acc5 = constraint.JointAccelerationConstraint(alims5)



instance_jnt5 = algo.TOPPRA([pc_vel5, pc_acc5], jnt_path5) #, parametrizer="ParametrizeConstAccel"
jnt_traj5 = instance_jnt5.compute_trajectory(0,0)

#print("DURATION OF OPTIMAL TRAJECTORY IS:", jnt_traj1.duration)

ts_sample5 = np.linspace(0, jnt_traj5.duration, 100)
qs_sample5 = jnt_traj5(ts_sample5)

### INSTANCE INTRA PATH 4
cart_intra4 = algo.TOPPRA([], path_intra4)
cart_intra4 = cart_intra4.compute_trajectory(0,0)

# path sampling
ts_intra4 = np.linspace(0, cart_intra4.duration, 100)
ps_intra4 = cart_intra4(ts_intra4)  # cartesian path samples

joint_way_intra4 = []
for i in range(len(ps_intra4)):
    q1, q2, q3 = inverse_kinematics_3R(ps_intra4[i,0], ps_intra4[i, 1], L1, L2, L3, phi=0.5)
    joint_way_intra4.append([q1, q2, q3])


ss_intra4 = np.linspace(0, 1, len(joint_way_intra4))

jnt_intra4 = ta.SplineInterpolator(ss, joint_way_intra4)

vlims_intra4 = 10 + np.random.rand(dof) * 20
alims_intra4 = 10 + np.random.rand(dof) * 2

pc_vel_intra4 = constraint.JointVelocityConstraint(vlims_intra4)
pc_acc_intra4 = constraint.JointAccelerationConstraint(alims_intra4)



instance_intra4 = algo.TOPPRA([pc_vel_intra4, pc_acc_intra4], jnt_intra4) #, parametrizer="ParametrizeConstAccel"
jnt_intra4 = instance_intra4.compute_trajectory(0,0)

#print("DURATION OF OPTIMAL TRAJECTORY IS:", jnt_traj1.duration)

ts_sample_intra4 = np.linspace(0, jnt_intra4.duration, 100)
qs_sample_intra4 = jnt_intra4(ts_sample_intra4)

### INSTANCE PATH 6
cart_instance6 = algo.TOPPRA([], path6)
cart_traj6 = cart_instance6.compute_trajectory(0,0)

# path sampling
ts_cart = np.linspace(0, cart_traj6.duration, 100)
ps_sample6 = cart_traj6(ts_cart)  # cartesian path samples

joint_way_pts6 = []
for i in range(len(ps_sample6)):
    q1, q2, q3 = inverse_kinematics_3R(ps_sample6[i,0], ps_sample6[i, 1], L1, L2, L3, phi=0.5)
    joint_way_pts6.append([q1, q2, q3])


ss_joint6 = np.linspace(0, 1, len(joint_way_pts6))

jnt_path6 = ta.SplineInterpolator(ss, joint_way_pts6)

vlims6 = 10 + np.random.rand(dof) * 20
alims6 = 10 + np.random.rand(dof) * 2

pc_vel6 = constraint.JointVelocityConstraint(vlims6)
pc_acc6 = constraint.JointAccelerationConstraint(alims6)



instance_jnt6 = algo.TOPPRA([pc_vel6, pc_acc6], jnt_path6) #, parametrizer="ParametrizeConstAccel"
jnt_traj6 = instance_jnt6.compute_trajectory(0,0)

#print("DURATION OF OPTIMAL TRAJECTORY IS:", jnt_traj1.duration)

ts_sample6 = np.linspace(0, jnt_traj6.duration, 100)
qs_sample6 = jnt_traj6(ts_sample6)

### INSTANCE PATH 7
cart_instance7 = algo.TOPPRA([], path7)
cart_traj7 = cart_instance7.compute_trajectory(0,0)

# path sampling
ts_cart = np.linspace(0, cart_traj7.duration, 100)
ps_sample7 = cart_traj7(ts_cart)  # cartesian path samples

joint_way_pts7 = []
for i in range(len(ps_sample7)):
    q1, q2, q3 = inverse_kinematics_3R(ps_sample7[i,0], ps_sample7[i, 1], L1, L2, L3, phi=0.5)
    joint_way_pts7.append([q1, q2, q3])


ss_joint7 = np.linspace(0, 1, len(joint_way_pts7))

jnt_path7 = ta.SplineInterpolator(ss, joint_way_pts7)

vlims7 = 10 + np.random.rand(dof) * 20
alims7 = 10 + np.random.rand(dof) * 2

pc_vel7 = constraint.JointVelocityConstraint(vlims1)
pc_acc7 = constraint.JointAccelerationConstraint(alims1)



instance_jnt7 = algo.TOPPRA([pc_vel7, pc_acc7], jnt_path7) #, parametrizer="ParametrizeConstAccel"
jnt_traj7 = instance_jnt7.compute_trajectory(0,0)

#print("DURATION OF OPTIMAL TRAJECTORY IS:", jnt_traj1.duration)

ts_sample7 = np.linspace(0, jnt_traj7.duration, 100)
qs_sample7 = jnt_traj7(ts_sample7)

### INSTANCE PATH 8
cart_instance8 = algo.TOPPRA([], path8)
cart_traj8 = cart_instance8.compute_trajectory(0,0)

# path sampling
ts_cart = np.linspace(0, cart_traj8.duration, 100)
ps_sample8 = cart_traj8(ts_cart)  # cartesian path samples

joint_way_pts8 = []
for i in range(len(ps_sample8)):
    q1, q2, q3 = inverse_kinematics_3R(ps_sample8[i,0], ps_sample8[i, 1], L1, L2, L3, phi=0.5)
    joint_way_pts8.append([q1, q2, q3])


ss_joint8 = np.linspace(0, 1, len(joint_way_pts8))

jnt_path8 = ta.SplineInterpolator(ss, joint_way_pts8)

vlims8 = 10 + np.random.rand(dof) * 20
alims8 = 10 + np.random.rand(dof) * 2

pc_vel8 = constraint.JointVelocityConstraint(vlims8)
pc_acc8 = constraint.JointAccelerationConstraint(alims8)



instance_jnt8 = algo.TOPPRA([pc_vel8, pc_acc8], jnt_path8) #, parametrizer="ParametrizeConstAccel"
jnt_traj8 = instance_jnt8.compute_trajectory(0,0)

#print("DURATION OF OPTIMAL TRAJECTORY IS:", jnt_traj1.duration)

ts_sample8 = np.linspace(0, jnt_traj8.duration, 100)
qs_sample8 = jnt_traj8(ts_sample8)

### INSTANCE INTRA PATH 5
cart_intra5 = algo.TOPPRA([], path_intra5)
cart_intra5 = cart_intra5.compute_trajectory(0,0)

# path sampling
ts_intra5 = np.linspace(0, cart_intra5.duration, 100)
ps_intra5 = cart_intra5(ts_intra5)  # cartesian path samples

joint_way_intra5 = []
for i in range(len(ps_intra5)):
    q1, q2, q3 = inverse_kinematics_3R(ps_intra5[i,0], ps_intra5[i, 1], L1, L2, L3, phi=0.5)
    joint_way_intra5.append([q1, q2, q3])


ss_intra5 = np.linspace(0, 1, len(joint_way_intra5))

jnt_intra5 = ta.SplineInterpolator(ss, joint_way_intra5)

vlims_intra5 = 10 + np.random.rand(dof) * 20
alims_intra5 = 10 + np.random.rand(dof) * 2

pc_vel_intra5 = constraint.JointVelocityConstraint(vlims_intra5)
pc_acc_intra5 = constraint.JointAccelerationConstraint(alims_intra5)



instance_intra5 = algo.TOPPRA([pc_vel_intra5, pc_acc_intra5], jnt_intra5) #, parametrizer="ParametrizeConstAccel"
jnt_intra5 = instance_intra5.compute_trajectory(0,0)

#print("DURATION OF OPTIMAL TRAJECTORY IS:", jnt_traj1.duration)

ts_sample_intra5 = np.linspace(0, jnt_intra5.duration, 100)
qs_sample_intra5 = jnt_intra5(ts_sample_intra5)

### INSTANCE PATH 9
cart_instance9 = algo.TOPPRA([], path9)
cart_traj9 = cart_instance9.compute_trajectory(0,0)

# path sampling
ts_cart = np.linspace(0, cart_traj9.duration, 100)
ps_sample9 = cart_traj9(ts_cart)  # cartesian path samples

joint_way_pts9 = []
for i in range(len(ps_sample9)):
    q1, q2, q3 = inverse_kinematics_3R(ps_sample9[i,0], ps_sample9[i, 1], L1, L2, L3, phi=0.5)
    joint_way_pts9.append([q1, q2, q3])


ss_joint9 = np.linspace(0, 1, len(joint_way_pts9))

jnt_path9 = ta.SplineInterpolator(ss, joint_way_pts9)

vlims9 = 10 + np.random.rand(dof) * 20
alims9 = 10 + np.random.rand(dof) * 2

pc_vel9 = constraint.JointVelocityConstraint(vlims9)
pc_acc9 = constraint.JointAccelerationConstraint(alims9)



instance_jnt9 = algo.TOPPRA([pc_vel9, pc_acc9], jnt_path9) #, parametrizer="ParametrizeConstAccel"
jnt_traj9 = instance_jnt9.compute_trajectory(0,0)

#print("DURATION OF OPTIMAL TRAJECTORY IS:", jnt_traj1.duration)

ts_sample9 = np.linspace(0, jnt_traj9.duration, 100)
qs_sample9 = jnt_traj9(ts_sample9)


## TOTAL TRAJECTORY TIME
tot_time = jnt_traj_first_1.duration + jnt_traj1.duration + jnt_intra1.duration 
+ jnt_traj_second_1.duration + jnt_traj2.duration + jnt_intra2.duration
+ jnt_traj3.duration + jnt_intra3.duration + jnt_traj4.duration
+ jnt_traj5.duration + jnt_intra4.duration + jnt_traj6.duration
+ jnt_traj7.duration + jnt_traj8.duration + jnt_intra5.duration, jnt_traj9.duration


print("TOTAL TRAJECTORY TIME IS: ", tot_time)

## CONCATENATE
qs_samples = [qs_sample_first_1, qs_sample1, qs_sample_intra1, qs_sample_second_1, qs_sample2, 
              qs_sample_intra2, qs_sample3, qs_sample_intra3, 
              qs_sample4, qs_sample5, qs_sample_intra4, 
              qs_sample6, qs_sample7, qs_sample8, qs_sample_intra5, qs_sample9]

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-3, 3)  # Adjust limits based on your robot's size
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
#ax.grid()

# Line representing the robot arm
line, = ax.plot([], [], 'o-', lw=2)

# Function to plot end-effector path for each segment separately
def plot_end_effector_path(qs_sample, color='r--'):
    end_effector_path_x = []
    end_effector_path_y = []
    for qs in qs_sample:
        positions = forward_kinematics_3R(qs)
        end_effector_position = positions[-1, :]  # End effector is the last point
        end_effector_path_x.append(end_effector_position[0])
        end_effector_path_y.append(end_effector_position[1])
    # Plot the path for this segment
    ax.plot(end_effector_path_x, end_effector_path_y, color, lw=1)

# Plot the end-effector paths for each trajectory separately
plot_end_effector_path(qs_sample_first_1, 'r')
plot_end_effector_path(qs_sample1, 'r')  # Path for trajectory 1
plot_end_effector_path(qs_sample_second_1, 'r')  # Path for trajectory 1
plot_end_effector_path(qs_sample2, 'r')  # Path for trajectory 2
plot_end_effector_path(qs_sample3, 'r')  # Path for trajectory 3
plot_end_effector_path(qs_sample4, 'r')  # Path for trajectory 4
plot_end_effector_path(qs_sample5, 'r')  # Path for trajectory 5
plot_end_effector_path(qs_sample6, 'r')  # Path for trajectory 6
plot_end_effector_path(qs_sample7, 'r')  # Path for trajectory 7
plot_end_effector_path(qs_sample8, 'r')  # Path for trajectory 8
plot_end_effector_path(qs_sample9, 'r')  # Path for trajectory 8

# Initialization function for FuncAnimation
def init():
    line.set_data([], [])
    return line,

# Animation function
def animate(i):
    # Determine which path we're currently on
    total_frames = sum([len(qs_sample) for qs_sample in qs_samples])
    path_indices = [(path_idx, frame_idx) for path_idx, qs_sample in enumerate(qs_samples) for frame_idx in range(len(qs_sample))]
    
    path_idx, frame_idx = path_indices[i]  # Get the path and frame

    # Get the joint angles for the current frame
    qs = qs_samples[path_idx][frame_idx]

    # Compute forward kinematics for this set of joint angles
    positions = forward_kinematics_3R(qs)

    # Unpack x and y coordinates for the robot
    x_data = positions[:, 0]
    y_data = positions[:, 1]

    # Update the robot arm's line plot
    line.set_data(x_data, y_data)
    return line,

# Total number of frames (all paths combined)
total_frames = sum([len(qs_sample) for qs_sample in qs_samples])

# Create the animation
anim = FuncAnimation(fig, animate, init_func=init, frames=total_frames, interval=20, blit=True)

# Show the animation
plt.show()