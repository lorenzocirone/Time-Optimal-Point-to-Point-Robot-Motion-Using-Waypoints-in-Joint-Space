import numpy as np
import matplotlib.pyplot as plt
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
from toppra import SplineInterpolator
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import time

st = time.time()


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
    # Extract joint angles, velocities, and accelerations
    q1, q2 = q
    q1d, q2d = qd
    q1dd, q2dd = qdd
    
    # Mass matrix M(q)
    M11 = I1 + m1*d1**2 + I2 + m2*d2**2 + m2*L1**2 + 2*m2*L1*d2 * np.cos(q2)
    M12 = I2 + m2*d2**2 + m2*L1*L2*np.cos(q2)
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
    G1 = (m1 * d1 + m2 * L1) * g * np.cos(q1) + m2 * d2 * g * np.cos(q1 + q2) 
    G2 = m2 * d2 * g * np.cos(q1 + q2)
    G = np.array([G1, G2])
    
    # Inverse dynamics: tau = M(q) * qdd + C(q, qd) * qd + G(q)
    tau = np.dot(M, qdd) + np.dot(C, qd) + G
    
    return tau


def generate_waypoints(point1, point2, num_waypoints):
    # Define the line equation by interpolating linearly between the two points, excluding endpoints
    x_values = np.linspace(point1[0], point2[0], num=num_waypoints+2)[1:-1]
    y_values = np.linspace(point1[1], point2[1], num=num_waypoints+2)[1:-1]

    # Stack x and y values to create an array of waypoints and flatten
    waypoints = np.column_stack((x_values, y_values)).flatten()
    
    return waypoints

#################### CHOOSE INITIAL AND FINAL CONFIGURATIONS ####################

# q_i = np.array([0, 0])  #### TEST 1
# q_f = np.array([np.pi / 2, 0])


# q_i = np.array([0,0], dtype=float) #### TEST 2
# q_f = np.array([3*np.pi/4, np.pi/4], dtype=float)


# q_i = np.array([0, 0], dtype=float)  #### TEST 3
# q_f = np.array([3*np.pi / 4, -np.pi / 4], dtype=float)


# q_i = np.array([0,0], dtype=float) #### TEST 4
# q_f = np.array([2*np.pi, 0], dtype=float)


q_i = np.array([0,0], dtype=float) ####  TEST 5
q_f = np.array([np.pi, 0], dtype=float)


###################### CHOOSE AND GENERATE WAYPOINTS  ######################
# initial_waypoints = generate_waypoints(q_i, q_f , 1) # 1.1, 2.1, 3.1, 4.1, 5.1
# initial_waypoints = generate_waypoints(q_i, q_f , 2) # 1.2, 2.2, 3.2, 4.2, 5.2
# initial_waypoints = generate_waypoints(q_i, q_f , 3) # 1.3, 2.3, 3.3, 4.3, 5.3
# initial_waypoints = generate_waypoints(q_i, q_f , 4) # 1.4, 2.4, 3.4, 4.4, 5.4
# initial_waypoints = generate_waypoints(q_i, q_f , 5) # 1.5, 2.5, 3.5, 4.5, 5.5
initial_waypoints = generate_waypoints(q_i, q_f, 10) # 1.10, 2.10, 3.10, 4.10, 5.10

####################### TORQUE CONSTRAINTS DEFINITION #######################
tau_max = np.array([[-20, 20], [-20, 20]])  # Torque limits 
fs_coef = np.array([0.0, 0.0])
pc_tau = constraint.JointTorqueConstraint(inverse_dynamics_2R, tau_max, fs_coef, 
                                          discretization_scheme=constraint.DiscretizationType.Interpolation)


all_instances= []
all_trajectories = []
all_durations = []
all_paths = []

points=[]

###################### OBJECTIVE FUNCTION DEFINITION ######################

def compute_trajectory_time(waypoints_flat):
    # Reshape the flat array into 2D array for waypoints (one row per waypoint)
    num_waypoints = waypoints_flat.size // 2  # Assuming 2 joints
    waypoints = waypoints_flat.reshape((num_waypoints, 2))

    # Define the full points set with initial, waypoints, and final configurations
    full_points_set = np.vstack([q_i, waypoints, q_f])
    points.append(full_points_set)
    t_waypoints = np.linspace(0, 1, len(full_points_set))
    path = SplineInterpolator(t_waypoints, full_points_set) #, bc_type='clamped'
    all_paths.append(path)

    # Create the TOPPRA instance and compute the trajectory
    instance = algo.TOPPRA([pc_tau], path, parametrizer="ParametrizeConstAccel") 
    all_instances.append(instance)
    trajectory = instance.compute_trajectory()
    all_trajectories.append(trajectory)
    
    # Penalize invalid trajectory durations
    if trajectory is None or trajectory.duration is None:
        return 1e6
    all_durations.append(trajectory.duration)
    return trajectory.duration

# Objective function for the optimization
def objective_function(waypoints_flat):
    return compute_trajectory_time(waypoints_flat)


##################################### BASIN-HOPPING INITIALIZATION ############################################

minimizer_kwargs = {"method": "SLSQP"} 
result = basinhopping(objective_function, initial_waypoints, niter=10, 
                      minimizer_kwargs=minimizer_kwargs, take_step=None, accept_test=None,
                      callback=None,  disp=True, niter_success=None,
                      seed=None, target_accept_rate=0.5, stepwise_factor=0.9)

# minimizer_kwargs = {"method": "L-BFGS-B"}
# result = basinhopping(objective_function, initial_waypoints, niter=15, 
#                       minimizer_kwargs=minimizer_kwargs, take_step=None, accept_test=None,
#                       callback=None,  disp=True, niter_success=None,
#                       seed=None, target_accept_rate=0.5, stepwise_factor=0.9)

# minimizer_kwargs = {"method": "Nelder-Mead"} 
# result = basinhopping(objective_function, initial_waypoints, niter=5, 
#                       minimizer_kwargs=minimizer_kwargs, take_step=None, accept_test=None,
#                       callback=None,  disp=True, niter_success=None,
#                       seed=None, target_accept_rate=0.5, stepwise_factor=0.9)


###################### SOLVE BASIN-HOPPING INSTANCE ######################

optimized_waypoints_flat = result.x
optimized_waypoints = optimized_waypoints_flat.reshape((-1, 2))

# Compute final trajectory with the optimized waypoints
final_duration = compute_trajectory_time(optimized_waypoints_flat)
iteration_count = result.nit
print(f"Total iterations: {iteration_count}")
print(f"Initial Duration: {all_durations[0]} s")
print(f"Initial Waypoints: {initial_waypoints}")
print(f"Final Duration with Optimized Waypoints: {final_duration} s")
print(f"Optimized Waypoints:\n{optimized_waypoints}")
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')


# Visualize optimized path
t_fine = np.linspace(0, final_duration, 100)
final_path = np.vstack([q_i, optimized_waypoints, q_f])
t_waypoints = np.linspace(0, 1, len(final_path))
path = SplineInterpolator(t_waypoints, final_path)

instance = algo.TOPPRA([pc_tau], path, parametrizer="ParametrizeConstAccel") 
trajectory = instance.compute_trajectory()
qs_sample = trajectory(t_fine)

plt.plot(qs_sample[:, 0], qs_sample[:, 1], label="Optimized Joint Path")
plt.scatter(final_path[:, 0], final_path[:, 1], color='r', label="Start/Waypoints/End")
plt.xlabel("Joint 1 Position")
plt.ylabel("Joint 2 Position")
plt.legend()
plt.title("Optimized Joint Space Path with Multiple Waypoints")
plt.grid()
plt.show()


###################### PLOT ALL THE PATHS ######################

for i in range(iteration_count):
     # Create the full path including the moving waypoint
    full_points_set = points[i]
    t_waypoints = np.linspace(0, 1, len(full_points_set))  # Define time for waypoints (here three points)

    path_joint_1 = ta.SplineInterpolator(t_waypoints, full_points_set[:, 0]) # bc_type='clamped
    path_joint_2 = ta.SplineInterpolator(t_waypoints, full_points_set[:, 1]) # bc_type='clamped

    # Spline Interpolator for the joint space path
    path = ta.SplineInterpolator(t_waypoints, full_points_set) # bc_type='clamped
    all_paths.append(path)

    # Create a time vector for interpolation (smooth path)
    t_fine = np.linspace(0, 1, 100)  # 100 points for smooth interpolation
    
    # Evaluate the spline at finer points
    interpolated_joint_1 = path_joint_1(t_fine)
    interpolated_joint_2 = path_joint_2(t_fine)

    # Combine the joint positions into a single path
    interpolated_path = np.vstack([interpolated_joint_1, interpolated_joint_2]).T

    if i % 100 == 0 or i == iteration_count: # REMOVE TO PLOT ALL THE PATHS

        # Plot the resulting interpolated path
        plt.plot(interpolated_joint_1, interpolated_joint_2, label=f'Path {i}')
        # Plot the waypoints
        plt.scatter(full_points_set[:, 0], full_points_set[:, 1], color='r')

    
#opt_full_points_set = np.array([q_i, optimized_waypoints, q_f])
opt_full_points_set = np.vstack([q_i, optimized_waypoints, q_f])
opt_path_joint_1 = ta.SplineInterpolator(t_waypoints, opt_full_points_set[:, 0]) # bc_type='clamped
opt_path_joint_2 = ta.SplineInterpolator(t_waypoints, opt_full_points_set[:, 1]) # bc_type='clamped

opt_path = ta.SplineInterpolator(t_waypoints, opt_full_points_set) # bc_type='clamped

opt_interpolated_joint_1 = opt_path_joint_1(t_fine)
opt_interpolated_joint_2 = opt_path_joint_2(t_fine)
opt_interpolated_path = np.vstack([opt_interpolated_joint_1, opt_interpolated_joint_2]).T

plt.plot(opt_interpolated_joint_1, opt_interpolated_joint_2, label=f'Optimal Path')
plt.scatter(opt_full_points_set[:, 0], opt_full_points_set[:, 1], color='r')

    

# Plot settings
plt.title('Spline Interpolated Paths for Moving Waypoint')
plt.xlabel('Joint 1 Configuration')
plt.ylabel('Joint 2 Configuration')
plt.grid(True)
plt.legend()
plt.show()

###################### INITIAL TRAJECTORY ######################
ts_init = np.linspace(0, all_trajectories[0].duration, 100)

qs_init = all_trajectories[0](ts_init)
qds_init = all_trajectories[0](ts_init, 1)
qdds_init = all_trajectories[0](ts_init, 2)

#parametrizer.ParametrizeConstAccel.plot_parametrization(min_traj)
fig, axs = plt.subplots(3, 1, sharex=True)
fig.suptitle('INITIAL TRAJECTORY')
for i in range(path.dof):
    # plot the i-th joint trajectory
    axs[0].plot(ts_init, qs_init[:, i], c="C{:d}".format(i),   label='q'+ str(i+1) + ' init')
    axs[1].plot(ts_init, qds_init[:, i], c="C{:d}".format(i),  label='$\dot{q}$'+ str(i+1) +  ' init')
    axs[2].plot(ts_init, qdds_init[:, i], c="C{:d}".format(i), label='$\ddot{q}$'+ str(i+1) +  ' init')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
axs[2].set_xlabel("Time (s)")
axs[0].set_ylabel("Position (rad)")
axs[1].set_ylabel("Velocity (rad/s)")
axs[2].set_ylabel("Acceleration (rad/s\u00b2)")

plt.show()

all_instances[0].compute_feasible_sets()
all_instances[0].inspect()

####################### PLOTTING TORQUE OF INITIAL TRAJECTORY ######################
torque_init = []
    
for q_, qd_, qdd_ in zip(qs_init, qds_init, qdds_init):
    torque_init.append(inverse_dynamics_2R(q_, qd_, qdd_) + fs_coef * np.sign(qd_))
torque = np.array(torque_init)

fig, axs = plt.subplots(dof, 1)
fig.suptitle("JOINT TORQUE OF INITIAL TRAJECTORY")
for i in range(0, path.dof):
    axs[i].plot(ts_init, torque[:, i])
    axs[i].plot([ts_init[0], ts_init[-1]], [tau_max[i], tau_max[i]], "--")
    axs[i].plot([ts_init[0], ts_init[-1]], [-tau_max[i], -tau_max[i]], "--")
  # Set labels and title for each subplot
    axs[i].set_ylabel(f"Torque $(Nm)$ - Joint {i+1}")
    axs[i].set_title(f"Torque Profile for Joint {i+1}")


plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()

################## PLOT ANIMATION OF INITIAL CARTESIAN TRAJECTORY ##################
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
for qs in qs_init:
    positions = forward_kinematics_2R(qs)
    end_effector_position = positions[-1, :]  # End effector is the last point
    end_effector_path.append(end_effector_position)

# Convert the path to x and y coordinates for plotting
path_x, path_y = zip(*end_effector_path)

# Plot the path in a separate figure
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
    qs = qs_init[i, :]  # (q1, q2, q3) at time i
    positions = forward_kinematics_2R(qs)
    
    # Unpack x and y coordinates for the robot
    x_data = positions[:, 0]
    y_data = positions[:, 1]
    
    # Update the robot arm's line plot
    line.set_data(x_data, y_data)
    
    return line,

# Create the animation
anim = FuncAnimation(fig, animate, init_func=init, frames=len(ts_init), interval=50, blit=True)

# Show the animation
plt.title('Trajectory with initial Waypoints')
plt.show()

######################## MIN TIME TRAJECTORY ######################
min_duration = min(all_durations)
min_index = np.argmin(all_durations)

min_traj = all_trajectories[min_index]
ts_sample = np.linspace(0, min_duration, 100)

qs_sample = min_traj(ts_sample)
qds_sample = min_traj(ts_sample, 1)
qdds_sample = min_traj(ts_sample, 2)

#parametrizer.ParametrizeConstAccel.plot_parametrization(min_traj)
fig, axs = plt.subplots(3, 1, sharex=True)
fig.suptitle("OPTIMAL TIME TRAJECTORY")
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

all_instances[min_index].compute_feasible_sets()
all_instances[min_index].inspect()

###################### PLOT TORQUE OF OPTIMAL TIME TRAJECTORY ######################
torque_opt = []
    
for q_, qd_, qdd_ in zip(qs_sample, qds_sample, qdds_sample):
    torque_opt.append(inverse_dynamics_2R(q_, qd_, qdd_) + fs_coef * np.sign(qd_))
torque_opt = np.array(torque)

fig, axs = plt.subplots(dof, 1)
fig.suptitle("JOINT TORQUE OF OPTIMAL TIME TRAJECTORY")
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

################# PLOT ANIMATION OF OPTIMAL CARTESIAN TRAJECTORY #################
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
    qs = qs_sample[i, :]  # (q1, q2, q3) at time i
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
plt.title('OPTIMAL TIME TRAJECTORY')
plt.show()