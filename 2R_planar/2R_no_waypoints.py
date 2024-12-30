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
L1, L2 = 1, 1  # Link lengths
m1, m2 = 1, 1  # Link masses
I1, I2 = 0.333, 0.333  # Moments of inertia
g = 0  # Set to 0 for no gravity test
dof = 2
d1, d2 = 0.5, 0.5


def forward_kinematics_2R(qs):
    x0, y0 = 0, 0  # Base (fixed)
    
    # First joint
    x1 = L1 * np.cos(qs[0])
    y1 = L1 * np.sin(qs[0])
    
    # Second joint
    x2 = x1 + L2 * np.cos(qs[0] + qs[1])
    y2 = y1 + L2 * np.sin(qs[0] + qs[1])
    
    
    return np.array([[x0, y0], [x1, y1], [x2, y2]])


def inverse_dynamics_2R(q, qd, qdd):
    """Compute the inverse dynamics for a 2R planar robot."""
    
    # Extract joint angles, velocities, and accelerations
    q1, q2 = q
    q1d, q2d = qd
    q1dd, q2dd = qdd
    
    # Mass matrix M(q)
    M11 = I1 + m1*d1**2 + I2 + m2*d2**2 + m2*L1**2 + 2*m2*L1*d2 * np.cos(q2)
    M12 = I2 + m2*d2**2 + m2*L1*d2*np.cos(q2)
    M21 = M12
    M22 = I2 + m2*d2**2
    M = np.array([[M11, M12],
                  [M21, M22]])
    
    # Coriolis and centrifugal matrix C(q, qd)
    C11 = -m2 * L1 *d2 * np.sin(q2) * q2d
    C12 = -m2 * L1 * d2 * np.sin(q2) * (q1d+q2d)
    C21 = m2 * L1 * d2 * np.sin(q2) * q1d
    C22 = 0
    C = np.array([[C11, C12],
                  [C21, C22]])
    
    # Gravity vector G(q)
    G1 = (m1 * d1 + m2 * L1) * g * np.cos(q1) + m2 * d2 * g * np.cos(q1 + q2) # prima centro di massa era a metà
    G2 = m2 * d2 * g * np.cos(q1 + q2)
    G = np.array([G1, G2])
    
    # Inverse dynamics: tau = M(q) * qdd + C(q, qd) * qd + G(q)
    tau = np.dot(M, qdd) + np.dot(C, qd) + G
    
    return tau

def generate_waypoints(point1, point2, num_waypoints):
    """
    Generates intermediate waypoints between two points along a straight line 
    and returns them as a flattened array.

    Parameters:
    point1 (tuple or list): Coordinates of the first point (x1, y1).
    point2 (tuple or list): Coordinates of the second point (x2, y2).
    num_waypoints (int): Number of waypoints to generate, excluding the initial and final points.

    Returns:
    np.ndarray: Flattened array of intermediate waypoints.
    """
    # Define the line equation by interpolating linearly between the two points, excluding endpoints
    x_values = np.linspace(point1[0], point2[0], num=num_waypoints+2)[1:-1]
    y_values = np.linspace(point1[1], point2[1], num=num_waypoints+2)[1:-1]

    # Stack x and y values to create an array of waypoints and flatten
    waypoints = np.column_stack((x_values, y_values)).flatten()
    
    return waypoints

################################################################################
# We generate a path with some random waypoints.

def generate_new_problem():
    # Parameters
    N_samples = 2
    dof = 2
    np.random.seed()
    #way_pts = np.random.randn(N_samples, dof)

    # way_pts = np.array([[0, 0],[np.pi / 2, 0]]) # TEST 1
    # way_pts = np.array([[0, 0],[3*np.pi/4, np.pi/4]]) # TEST 2
    way_pts = np.array([[0, 0],[3*np.pi/4, -np.pi/4]]) # TEST 3
    # way_pts = np.array([[0, 0],[2 * np.pi, 0]]) # TEST 4
    # way_pts = np.array([[0, 0],[np.pi, 0]]) # TEST 5

    return (
        np.linspace(0, 1, N_samples), #ss 5 = N_samples = N° waypoints
        way_pts, 
    )

ss, way_pts = generate_new_problem()

path = ta.SplineInterpolator(ss, way_pts)

def plot_path(path, way_pts, N_samples=100):
    # Sample the interpolated path at more points for smoothness
    ss_sampled = np.linspace(ss[0], ss[-1], N_samples)
    qs_sampled = path.eval(ss_sampled)

    # Plot the interpolated path
    plt.figure(figsize=(6, 6))
    plt.plot(qs_sampled[:, 0], qs_sampled[:, 1], label="Interpolated Path", color='b')

    # Plot the waypoints
    plt.scatter(way_pts[:, 0], way_pts[:, 1], color='r', label="Waypoints", zorder=5)

    # Labels and grid
    plt.title("Path of the 2R Robot")
    plt.xlabel("Joint 1 Position")
    plt.ylabel("Joint 2 Position")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")  # Ensure aspect ratio is equal

    # Show the plot
    plt.show()

# Call the function to plot the path
plot_path(path, way_pts)

tau_max = np.array([[-20, 20], [-20, 20]])  # Torque limits 
fs_coef = np.array([0.0, 0.0])

pc_tau = constraint.JointTorqueConstraint(
    inverse_dynamics_2R, tau_max, fs_coef, discretization_scheme=constraint.DiscretizationType.Interpolation
)

instance = algo.TOPPRA([pc_tau], path, parametrizer="ParametrizeConstAccel") # ONLY TORQUE

jnt_traj = instance.compute_trajectory(0, 0)

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

#PLOTTING TORQUE
torque = []
    
for q_, qd_, qdd_ in zip(qs_sample, qds_sample, qdds_sample):
    torque.append(inverse_dynamics_2R(q_, qd_, qdd_) + fs_coef * np.sign(qd_))
torque = np.array(torque)

fig, axs = plt.subplots(dof, 1)
for i in range(0, path.dof):
    axs[i].plot(ts_sample, torque[:, i])
    axs[i].plot([ts_sample[0], ts_sample[-1]], [tau_max[i], tau_max[i]], "--")
    axs[i].plot([ts_sample[0], ts_sample[-1]], [-tau_max[i], -tau_max[i]], "--")
  # Set labels and title for each subplot
    axs[i].set_ylabel(f"Torque $(Nm)$ - Joint {i+1}")
    axs[i].set_title(f"Torque Profile for Joint {i+1}")
    # axs[i].legend(loc='upper right')

plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()

#PLOTTING THE PATH IN JOINT SPACE

plt.plot(qs_sample[:, 0], qs_sample[:, 1], label='Time-Optimal Path')
plt.scatter(way_pts[:, 0], way_pts[:, 1], color='red', label='Waypoints')
plt.title('Time-Optimal Path')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.axis('equal')  # Ensure equal scaling for both axes
plt.legend()
plt.grid(True)
plt.show()

# PLOTTING VELOCITY CURVE
instance.compute_feasible_sets() # plots the feasible sets, I can inspect also the controllable and reachable set
instance.inspect() # plots the s-s_dot^2 phase plane with the optimal trajectory

##### PLOTTING ANIMATION OF THE ROBOT AND THE RESULTING CARTESIAN PATH

# Forward kinematics for 2R planar robot
# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-3, 3)  # Adjust limits based on your robot's size
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.grid()

# Line representing the robot arm
line, = ax.plot([], [], 'o-', lw=2)

# Compute the end effector path before animation starts
end_effector_path = []
for qs in qs_sample:
    positions = forward_kinematics_2R(qs)
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
    qs = qs_sample[i, :]  # (q1, q2) at time i
    positions = forward_kinematics_2R(qs)
    
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