import numpy as np
import matplotlib.pyplot as plt
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
from toppra import SplineInterpolator
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from mpl_toolkits.mplot3d import Axes3D
import time

st = time.time()

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

def generate_waypoints(point1, point2, num_waypoints):
    # Generate equally spaced points including the endpoints, then exclude them
    x_values = np.linspace(point1[0], point2[0], num=num_waypoints + 2)[1:-1]
    y_values = np.linspace(point1[1], point2[1], num=num_waypoints + 2)[1:-1]
    z_values = np.linspace(point1[2], point2[2], num=num_waypoints + 2)[1:-1]

    # Stack x, y, and z values to create an array of waypoints and flatten
    waypoints = np.column_stack((x_values, y_values, z_values)).flatten()
    
    return waypoints


#################### CHOOSE INITIAL AND FINAL CONFIGURATIONS ####################

q_i = np.array([0, 0, 0])  #### TEST 1
q_f = np.array([np.pi / 2, 0, 0])


# q_i = np.array([0,0,0], dtype=float) #### TEST 2
# q_f = np.array([3*np.pi/4, -np.pi/4,0], dtype=float)


# q_i = np.array([0, 0, 0], dtype=float)  #### TEST 3
# q_f = np.array([3 * np.pi / 4, -np.pi / 4, -np.pi/2], dtype=float)  



# q_i = np.array([0,0,0], dtype=float) #### TEST 4
# q_f = np.array([2*np.pi, 0, 0], dtype=float)


# q_i = np.array([0,0,0], dtype=float) ####  TEST 5
# q_f = np.array([np.pi, 0, 0], dtype=float)


###################### CHOOSE AND GENERATE WAYPOINTS  ######################
initial_waypoints = generate_waypoints(q_i, q_f, 1)  # 1.1, 2.1, 3.1, 4.1, 5.1
# initial_waypoints = generate_waypoints(q_i, q_f , 2) # 1.2, 2.2, 3.2, 4.2, 5.2 
# initial_waypoints = generate_waypoints(q_i, q_f , 3) # 1.3, 2.3, 3.3, 4.3, 5.3
# initial_waypoints = generate_waypoints(q_i, q_f , 4) # 1.4, 2.4, 3.4, 4.4, 5.4 
# initial_waypoints = generate_waypoints(q_i, q_f , 5) # 1.5, 2.5, 3.5, 4.5, 5.5 
# initial_waypoints = generate_waypoints(q_i, q_f, 10) # 1.10, 2.10, 3.10, 4.10, 5.10

####################### TORQUE CONSTRAINTS DEFINITION #######################

tau_max = np.array([[-40, 40], [-35, 35], [-30, 30]])
fs_coef = np.array([0.0, 0.0, 0.0])
pc_tau = constraint.JointTorqueConstraint(inverse_dynamics_3R, tau_max, fs_coef, 
                                          discretization_scheme=constraint.DiscretizationType.Interpolation)


all_instances= []
all_trajectories = []
all_durations = []
all_paths=[]
points=[]


###################### OBJECTIVE FUNCTION DEFINITION ######################

def compute_trajectory_time(waypoints_flat):
    # Reshape the flat array into a 2D array for waypoints (one row per waypoint)
    num_waypoints = waypoints_flat.size // 3  # Assuming 3 joints
    waypoints = waypoints_flat.reshape((num_waypoints, 3))

    # Define the full points set with initial, waypoints, and final configurations
    full_points_set = np.vstack([q_i, waypoints, q_f])
    points.append(full_points_set)
    t_waypoints = np.linspace(0, 1, len(full_points_set))
    path = SplineInterpolator(t_waypoints, full_points_set)
    all_paths.append(path)

    # Create the TOPPRA instance and compute the trajectory
    instance = algo.TOPPRA([pc_tau], path, parametrizer="ParametrizeConstAccel")
    all_instances.append(instance)
    trajectory = instance.compute_trajectory()
    all_trajectories.append(trajectory)
    
    # Penalize invalid trajectory durations
    if trajectory is None or trajectory.duration is None:
        return 1e6  # Large penalty for invalid trajectory
    all_durations.append(trajectory.duration)
    return trajectory.duration

# Objective function for the optimization
def objective_function(waypoints_flat):
    return compute_trajectory_time(waypoints_flat)




##################################### BASIN-HOPPING INITIALIZATION ############################################


minimizer_kwargs = {"method": "SLSQP"}  #all test with 5 waypoints 
result = basinhopping(objective_function, initial_waypoints, niter=10, 
                      minimizer_kwargs=minimizer_kwargs, take_step=None, accept_test=None,
                      callback=None,  disp=True, niter_success=None,
                      seed=None, target_accept_rate=0.5, stepwise_factor=0.9)


# minimizer_kwargs = {"method": "L-BFGS-B"} 
# result = basinhopping(objective_function, initial_waypoints, niter=5, 
#                       minimizer_kwargs=minimizer_kwargs, take_step=None, accept_test=None,
#                       callback=None,  disp=True, niter_success=None,
#                       seed=None, target_accept_rate=0.5, stepwise_factor=0.9)

# minimizer_kwargs = {"method": "Nelder-Mead"} 
# result = basinhopping(objective_function, initial_waypoints, niter=10, 
#                       minimizer_kwargs=minimizer_kwargs, take_step=None, accept_test=None,
#                       callback=None,  disp=True, niter_success=None,
#                       seed=None, target_accept_rate=0.5, stepwise_factor=0.9)

###################### SOLVE BASIN-HOPPING INSTANCE ######################

optimized_waypoints_flat = result.x
iteration_count = result.nit
optimized_waypoints = optimized_waypoints_flat.reshape((-1, 3)) 


min_duration = min(all_durations)
min_index = np.argmin(all_durations)

min_traj = all_trajectories[min_index]
ts_sample = np.linspace(0, min_traj.duration, 100)


# Compute final trajectory with the optimized waypoints
print(f"Total iterations: {iteration_count}")
final_duration = compute_trajectory_time(optimized_waypoints_flat)
print(f"Initial Duration: {all_durations[0]} s")
print(f"Initial Waypoints:\n{initial_waypoints.reshape((-1, 3))}")  # Reshape for display
print(f"Final Duration with Optimized Waypoints: {final_duration} s")
print(f"MIN DURATION: {min_duration} s")
print(f"Optimized Waypoints:\n{optimized_waypoints}")
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')


# Visualize optimized path
t_fine = np.linspace(0, final_duration, 100)
final_path = np.vstack([q_i, optimized_waypoints, q_f]) # change in real_full_point_set
t_waypoints = np.linspace(0, 1, len(final_path))
path = SplineInterpolator(t_waypoints, final_path)

instance = algo.TOPPRA([pc_tau], path, parametrizer="ParametrizeConstAccel")
trajectory = instance.compute_trajectory()
qs_sample = trajectory(t_fine)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
# Plotting the optimized joint path
ax.plot(qs_sample[:, 0], qs_sample[:, 1], qs_sample[:, 2], label="Optimized Joint Path")

# Plotting the start, end, and optimized waypoint
ax.scatter(final_path[:, 0], final_path[:, 1], final_path[:, 2], color='r', label="Start/End/Optimized Waypoint")

# Set axis labels
ax.set_xlabel("Joint 1 Position")
ax.set_ylabel("Joint 2 Position")
ax.set_zlabel("Joint 3 Position")

# Additional plot settings
plt.legend()
plt.title("Optimized Joint Space Path")
plt.grid()
plt.show()

###################### PLOT ALL THE PATHS ######################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # Creating a 3D subplot

for i in range(iteration_count):
     # Create the full path including the moving waypoint
    full_points_set = points[i]
    t_waypoints = np.linspace(0, 1, len(full_points_set))  # Define time for waypoints (here three points)

    path_joint_1 = ta.SplineInterpolator(t_waypoints, full_points_set[:, 0])
    path_joint_2 = ta.SplineInterpolator(t_waypoints, full_points_set[:, 1])
    path_joint_3 = ta.SplineInterpolator(t_waypoints, full_points_set[:, 2])

    # Spline Interpolator for the joint space path
    path = ta.SplineInterpolator(t_waypoints, full_points_set)
    all_paths.append(path)

    # Create a time vector for interpolation (smooth path)
    t_fine = np.linspace(0, 1, 100)  # 100 points for smooth interpolation
    
    # Evaluate the spline at finer points
    interpolated_joint_1 = path_joint_1(t_fine)
    interpolated_joint_2 = path_joint_2(t_fine)
    interpolated_joint_3 = path_joint_3(t_fine)

    # Combine the joint positions into a single path
    interpolated_path = np.vstack([interpolated_joint_1, interpolated_joint_2, interpolated_joint_3]).T

    if i % 1000 == 0 or i == iteration_count: # REMOVE TO PLOT ALL THE PATHS

        # Plot the resulting interpolated path
        ax.plot(interpolated_joint_1, interpolated_joint_2, interpolated_joint_3, label=f'Path {i}')
        # Plot the waypoints
        ax.scatter(full_points_set[:, 0], full_points_set[:, 1], full_points_set[:, 2], color='r')

    
#opt_full_points_set = np.array([q_i, optimized_waypoint, q_f])
opt_full_points_set = np.vstack([q_i, optimized_waypoints, q_f])
opt_path_joint_1 = ta.SplineInterpolator(t_waypoints, opt_full_points_set[:, 0])
opt_path_joint_2 = ta.SplineInterpolator(t_waypoints, opt_full_points_set[:, 1])
opt_path_joint_3 = ta.SplineInterpolator(t_waypoints, opt_full_points_set[:, 2])

opt_path = ta.SplineInterpolator(t_waypoints, opt_full_points_set)

opt_interpolated_joint_1 = opt_path_joint_1(t_fine)
opt_interpolated_joint_2 = opt_path_joint_2(t_fine)
opt_interpolated_joint_3 = opt_path_joint_3(t_fine)
opt_interpolated_path = np.vstack([opt_interpolated_joint_1, opt_interpolated_joint_2, opt_interpolated_joint_3]).T

ax.plot(opt_interpolated_joint_1, opt_interpolated_joint_2, opt_interpolated_joint_3, label=f'Optimal Path')
ax.scatter(opt_full_points_set[:, 0], opt_full_points_set[:, 1], opt_full_points_set[:, 2], color='r')

# 3D Plot settings
ax.set_title('Spline Interpolated Paths for Moving Waypoint')
ax.set_xlabel('Joint 1 Configuration')
ax.set_ylabel('Joint 2 Configuration')
ax.set_zlabel('Joint 3 Configuration')
ax.grid(True)
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
    torque_init.append(inverse_dynamics_3R(q_, qd_, qdd_) + fs_coef * np.sign(qd_))
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
    # axs[i].legend(loc='upper right')

plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()

################## PLOT ANIMATION OF INITIAL CARTESIAN TRAJECTORY ##################
# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-3.5, 3.5)  # Adjust limits based on your robot's size
ax.set_ylim(-3.5, 3.5)
ax.set_aspect('equal')
ax.grid()

# Line representing the robot arm
line, = ax.plot([], [], 'o-', lw=2)

# Compute the end effector path before animation starts
end_effector_path = []
for qs in qs_init:
    positions = forward_kinematics_3R(qs)
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
    positions = forward_kinematics_3R(qs)
    
    # Unpack x and y coordinates for the robot
    x_data = positions[:, 0]
    y_data = positions[:, 1]
    
    # Update the robot arm's line plot
    line.set_data(x_data, y_data)
    
    return line,

# Create the animation
anim = FuncAnimation(fig, animate, init_func=init, frames=len(ts_init), interval=50, blit=True)

# Show the animation
plt.title('Trajectory with initial Waypoint')
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
    torque_opt.append(inverse_dynamics_3R(q_, qd_, qdd_) + fs_coef * np.sign(qd_))
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
ax.set_xlim(-3.5, 3.5)  # Adjust limits based on your robot's size
ax.set_ylim(-3.5, 3.5)
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
plt.title('OPTIMAL TIME TRAJECTORY')
plt.show()

