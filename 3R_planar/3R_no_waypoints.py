import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
from toppra import parametrizer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from mpl_toolkits.mplot3d import Axes3D

### SETUP WITH CONSTRAINTS ON JOINT VELOCITIES, ACCELERATIONS AND TORQUES.
### WE CAN SET WAYPOINTS IN JOINT SPACE, INTERPOLATES AND SOLVES THE TOPP PROBLEM.
### THEN WE PLOT THE CARTESIAN RESULTANT PATH USING FORWARD KINEMATICS.

ta.setup_logging("INFO")

# Define physical parameters
L1, L2, L3 = 1, 1, 1  # Link lengths
m1, m2, m3 = 1, 1, 1  # Link masses
I1, I2, I3 = 0.333, 0.333, 0.333  # Moments of inertia
d1, d2, d3 = 0.5, 0.5, 0.5
g = 0  # Set to 0 for no gravity test
dof = 3

# Forward kinematics for 3R planar robot
def forward_kinematics_3R(qs):
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
    
    M11 = I1 + I2 + I3 + m1*d1**2 + m2*(L1**2 + d2**2 + 2*L1*d2*np.cos(q2)) + m3*(L1**2 + L2**2 + d3**2 + 2*L1*L2*np.cos(q2)
          + 2*L1*d3*np.cos(q2+q3) + 2*L2*d3*np.cos(q3))
    M12 = I2 + I3 + m2*(d2**2 + L1*d2*np.cos(q2)) + m3*(L2**2 + d3**2 + L1*L2*np.cos(q2) + L1*d3*np.cos(q2+q3) + L2*d3*np.cos(q3))
    M13 = I3 + m3*(d3**2 + L1*d3*np.cos(q2+q3) + L2*d3*np.cos(q3))
    M21 = M12
    M22 = I2 + I3 + m2*d2**2 + m3*(L2**2 + d3**2 + 2*L2*d3*np.cos(q3))
    M23 = I3 + m3*(d3**2 + L2*d3*np.cos(q3))
    M31 = M13
    M32 = M23
    M33 = I3 + m3*d3**2
    M = np.array([[M11, M12, M13],
                  [M21, M22, M23],
                  [M31, M32, M33]])
    
    C11 = -m2*L1*d2*np.sin(q2)*q2d - m3*(L1*d3*np.sin(q2+q3)*(q2d+q3d) + L1*d2*np.sin(q2)*q2d + L2*d3*np.sin(q3)*q3d)
    C12 = -m2*L1*d2*np.sin(q2)*(q1d+q2d) - m3*(L1*d3*np.sin(q2+q3)*(q1d+q2d+q3d) + L1*d2*np.sin(q2)*(q1d+q2d) + L2*d3*np.sin(q3)*q3d)
    C13 = -m3*(L1*d3*np.sin(q2+q3)*(q1d+q2d+q3d) + L2*d3*np.sin(q3)*(q1d+q2d+q3d))
    C21 = m2*L1*d2*np.sin(q2)*q1d + m3*(L1*d3*np.sin(q2+q3)*q1d + L1*d2*np.sin(q2)*q1d + L2*d3*np.sin(q3)*q3d)
    C22 = -m3*L2*d3*np.sin(q3)*q3d
    C23 = -m3*L2*d3*np.sin(q3)*(q2d+q3d)
    C31 = m3*(L1*d3*np.sin(q2+q3)*q1d + L2*d3*np.sin(q3)*q1d)
    C32 = m3*L2*d3*np.sin(q3)*q2d
    C33 = 0
    C = np.array([[C11, C12, C13],
                  [C21, C22, C23],
                  [C31, C32, C33]])
    
    G1 = (m1*d1 + m2*L1 + m3*L1)*g*np.cos(q1) + m2*d2*g*np.cos(q1+q2) + m3*(L2*np.cos(q1+q2) + d3**np.cos(q1+q2+q3))*g
    G2 = (m2*d2 + m3*L2)*g*np.cos(q1+q2) + m3*d3*g*np.cos(q1+q2+q3)
    G3 = m3*d3*g*np.cos(q1+q2+q3)
    G = np.array([G1, G2, G3])
    
    tau = np.dot(M, qdd) + np.dot(C, qd) + G
    
    return tau

################################################################################
# We generate a path with some random waypoints.

def generate_new_problem(seed=4):
    # Parameters
    N_samples = 2

    # way_pts = np.array([[0, 0, 0], [np.pi/2, 0, 0]])  # TEST 1
    # way_pts = np.array([[0, 0, 0], [3*np.pi/4, -np.pi/4,0]])  # TEST 2
    # way_pts = np.array([[0, 0, 0], [3*np.pi / 4, -np.pi/4, -np.pi/2]])  # TEST 3
    # way_pts = np.array([[0, 0, 0], [2*np.pi, 0, 0]])  # TEST 4
    way_pts = np.array([[0, 0, 0], [np.pi, 0, 0]])  # TEST 5

    return (
        np.linspace(0, 1, N_samples), #ss
        way_pts,
    )

ss, way_pts = generate_new_problem()

################################################################################
# Define the geometric path and two constraints.

path = ta.SplineInterpolator(ss, way_pts)

tau_max = np.array([[-40, 40], [-35, 35], [-30, 30]])  # Example torque limits for each joint

fs_coef = np.array([0.0, 0.0, 0.0])

# Define the joint torque constraint using the inverse dynamics function
pc_tau = constraint.JointTorqueConstraint(
    inverse_dynamics_3R, tau_max, fs_coef, discretization_scheme=constraint.DiscretizationType.Interpolation
)


instance = algo.TOPPRA([pc_tau], path, parametrizer="ParametrizeConstAccel") 
jnt_traj = instance.compute_trajectory(0,0)

ts_sample = np.linspace(0, jnt_traj.duration, 100)
qs_sample = jnt_traj(ts_sample)
qds_sample = jnt_traj(ts_sample, 1)
qdds_sample = jnt_traj(ts_sample, 2)
fig, axs = plt.subplots(3, 1, sharex=True)
for i in range(path.dof):
    # plot the i-th joint trajectory
    axs[0].plot(ts_sample, qs_sample[:, i], c="C{:d}".format(i),   label='q'+ str(i+1))
    axs[1].plot(ts_sample, qds_sample[:, i], c="C{:d}".format(i),  label='$\dot{q}$'+ str(i+1))
    axs[2].plot(ts_sample, qdds_sample[:, i], c="C{:d}".format(i), label='$\ddot{q}$'+ str(i+1))
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
axs[2].set_xlabel("Time (s)")
axs[0].set_ylabel("Position (rad)")
axs[1].set_ylabel("Velocity (rad/s)")
axs[2].set_ylabel("Acceleration (rad/s\u00b2)")
plt.show()

#PLOTTING THE PATH IN JOINT SPACE IN 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot

ax.plot(qs_sample[:, 0], qs_sample[:, 1], qs_sample[:, 2], label='Time-Optimal Path')
ax.scatter(way_pts[:, 0], way_pts[:, 1], way_pts[:, 2], color='red', label='Waypoints')
ax.set_xlabel("Joint 1 Position")
ax.set_ylabel("Joint 2 Position")
ax.set_zlabel("Joint 3 Position")

plt.axis('equal')  # Ensure equal scaling for both axes
plt.legend()
plt.title("Optimized Joint Space Path")
plt.grid()
plt.show()

#PLOTTING TORQUE
torque = []
    
for q_, qd_, qdd_ in zip(qs_sample, qds_sample, qdds_sample):
    torque.append(inverse_dynamics_3R(q_, qd_, qdd_) + fs_coef * np.sign(qd_))
torque = np.array(torque)

# Plotting
fig, axs = plt.subplots(dof, 1, figsize=(8, 10))  # Adjust size to fit 3 subplots

for i in range(dof):
    axs[i].plot(ts_sample, torque[:, i]) #label=f"Joint {i+1} Torque"
    axs[i].plot([ts_sample[0], ts_sample[-1]], [tau_max[i], tau_max[i]])
    axs[i].plot([ts_sample[0], ts_sample[-1]], [-tau_max[i], -tau_max[i]])
    
    # Set labels and title for each subplot
    axs[i].set_ylabel(f"Torque $(Nm)$ - Joint {i+1}")
    axs[i].set_title(f"Torque Profile for Joint {i+1}")
    # axs[i].legend(loc='upper right')

plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()

instance.compute_feasible_sets() # plots the feasible sets, I can inspect also the controllable and reachable set
instance.inspect() # plots the s-s_dot^2 phase plane with the optimal trajectory

##### PLOTTING ANIMATION OF THE ROBOT AND THE RESULTING CARTESIAN PATH
# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-4, 4)  # Adjust limits based on your robot's size
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.grid()

# Line representing the robot arm
line, = ax.plot([], [], 'o-', lw=2)

# Compute the end effector path before animation starts
end_effector_path = []
for qs in qs_sample:
    positions = forward_kinematics_3R(qs)
    end_effector_position = positions[-1, :]  # End effector is the last point
    end_effector_path.append(end_effector_position)

# Convert the path to x and y coordinates for plotting
path_x, path_y = zip(*end_effector_path)

# Plot the path before the animation starts
ax.plot(path_x, path_y, 'r--', lw=1)  # Plot path as a red dashed line

# Initialization function for FuncAnimation
def init():
    line.set_data([], [])
    return line,

# Animation function
def animate(i):
    # Get joint positions at time step i
    qs = qs_sample[i, :]  # (q1, q2, q3) at time i
    positions = forward_kinematics_3R(qs)
    
    # Unpack x and y coordinates for the robot
    x_data = positions[:, 0]
    y_data = positions[:, 1]
    
    # Update the robot arm's line plot
    line.set_data(x_data, y_data)
    
    return line,

# Create the animation
anim = FuncAnimation(fig, animate, init_func=init, frames=len(ts_sample), interval=50, blit=True)

# Show the animation
plt.show()