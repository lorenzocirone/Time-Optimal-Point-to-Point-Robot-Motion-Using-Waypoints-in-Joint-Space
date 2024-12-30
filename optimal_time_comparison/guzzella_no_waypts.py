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
L1, L2 = 0.4, 0.25  # Link lengths
m1, m2 = 15, 6  # Link masses
I1, I2 = 1.6, 0.43  # Moments of inertia
g = 0  # Set to 0 for no gravity test
dof = 2
d1, d2 = 0.2, 0.125


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

################################################################################
# We generate a path with some random waypoints.
N_samples = 2

way_pts = np.array([[0, 0],[np.pi/3, 0]]) # TEST A0
# way_pts = np.array([[0, 0],[1, 0]]) # TEST A1
#way_pts = np.array([[0, 0],[0.785, -2*np.pi]]) # TEST B0
# way_pts = np.array([[0, 0],[np.pi, -2*np.pi]]) # TEST B1
ss = np.linspace(0,1,N_samples)


path = ta.SplineInterpolator(ss, way_pts)
#ss_sampled = np.linspace(ss[0], ss[-1], 5)
qs_before = path.eval(ss)

# Assuming ss, way_pts, vlims, alims, and path have been defined as in your original code
def plot_path(path, way_pts, N_samples=100):
    # Sample the interpolated path at more points for smoothness
    ss_sampled = np.linspace(ss[0], ss[-1], N_samples)
    qs_sampled = path.eval(ss_sampled)

    # Plot the interpolated path
    plt.figure(figsize=(6, 6))
    plt.plot(qs_sampled[:, 0], qs_sampled[:, 1], label="Interpolated Path", color='b')

    # Plot the waypoints
    plt.scatter(way_pts[:, 0], way_pts[:, 1], color='r', label="Init/Final Conf", zorder=5)

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

#path = ta.UnivariateSplineInterpolator(ss, way_pts)
#print(path.path_interval)

#vlims = np.array([[-10, 10], [-10, 10]])
#alims = 20 * np.ones(dof)


tau_max = np.array([[-25, 25], [-9, 9]])  # Torque limits 
fs_coef = np.array([0, 0.0])

pc_tau = constraint.JointTorqueConstraint(
    inverse_dynamics_2R, tau_max, fs_coef, discretization_scheme=constraint.DiscretizationType.Interpolation
)

################################################################################

instance = algo.TOPPRA([pc_tau], path, parametrizer="ParametrizeConstAccel")
jnt_traj = instance.compute_trajectory(0,0)

################################################################################

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

################################################################################
# Optionally, we can inspect the output.
#X = instance.compute_feasible_sets() # plots the feasible sets, I can inspect also the controllable and reachable set
# print(X)
#instance.inspect() # plots the s-s_dot^2 phase plane with the optimal trajectory

# Better to use this instead of the function inspect() since I cannot change the options
X = instance.compute_feasible_sets()
K = instance.compute_controllable_sets(0, 0)
_, sd_vec, _ = instance.compute_parameterization(0, 0)
X = np.sqrt(X)
K = np.sqrt(K)
plt.plot(X[:, 0], c='green', label="Feasible sets")
plt.plot(X[:, 1], c='green')
plt.plot(K[:, 0], '--', c='red', label="Controllable sets")
plt.plot(K[:, 1], '--', c='red')
plt.plot(sd_vec, label="Velocity profile")
plt.title("Path-position path-velocity plot")
plt.xlabel("Path position")
plt.ylabel("Path velocity square")
plt.legend()
plt.tight_layout()
plt.show()

##### PLOTTING ANIMATION OF THE ROBOT AND THE RESULTING CARTESIAN PATH

# Forward kinematics for 2R planar robot
# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-0.50, 0.75)  # Adjust limits based on your robot's size
ax.set_ylim(-0.5, 0.6)
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

# # Plot the path in a separate figure
# fig_path, ax_path = plt.subplots()
# ax_path.plot(path_x, path_y, 'r--', lw=1)  # Plot path as a red dashed line
# ax_path.set_title('End Effector Path')
# ax_path.set_xlabel('X Position')
# ax_path.set_ylabel('Y Position')
# ax_path.set_aspect('equal')
# ax_path.grid()


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