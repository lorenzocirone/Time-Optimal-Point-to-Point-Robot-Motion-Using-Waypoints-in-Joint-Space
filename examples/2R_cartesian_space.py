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
g = 9.81  # Set to 0 for no gravity test
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

def inverse_kinematics_2R(x, y, L1, L2):
    """Calculate the inverse kinematics for a 2R planar robot."""
    # q1 = np.arctan2(y, x)
    # d = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    # q2 = np.arccos(np.clip(d, -1, 1))  # Ensure value is within the domain of arccos

    c2=(x**2+y**2-L1**2-L2**2)/(2*L1*L2)
    s2=-np.abs(np.sqrt(1-c2**2))
    #s2=-np.abs(np.sqrt(1-c2**2))
    # s1=(((L1+L2*c2)*x)+L2*s2*y)/(x^2+y^2)
    # c1=(((L1+L2*c2)*y)+L2*s2*x)/(x^2+y^2)
    q1=np.arctan2(y,x)-np.arctan2(L2*s2,L1+L2*c2)
    q2=np.arctan2(s2,c2)

    return q1, q2


#N_samples=3 # number of waypoints
n = 2 # number of joints
dof=2 # task dof

#ss = np.linspace(0, 1, N_samples)

#way_pts_cartesian = np.array([[1.5, 1],[1.5,-1]])
#way_pts_cartesian = np.array([[2, 0],[1.5, 0.125], [1, 1]])


#ss = np.linspace(0, 0.9, 100) # Software Fede Lambda max = 0.9 Esempio 1.1
#ss = np.linspace(0, 2, 100) # Software Fede Lambda max = 0.9 Esempio 2.1
ss = np.linspace(0, 1, 100)
# CUSTOM CUBIC PATH
def generate_custom_cubic(): # Esempio 1.1 FEDE
    points= []
    for i in range(len(ss)):
        x = 2-ss[i]
        y = ss[i]**3
        points.append([x, y])
    return points

# LINEAR VERTICAL PATH 

def generate_vertical_line(x0, y0):# Esempio 2.1 FEDE
    points= []
    for i in range(len(ss)): 
        x = x0
        y = y0 - ss[i]
        points.append([x, y])

    return points

def generate_vertical_line_updown(x0, y0):# Esempio 2.1 FEDE
    points= []
    for i in range(len(ss)): 
        x = x0
        y = y0 - ss[i]
        points.append([x, y])

    return points

def generate_vertical_line_downup(x0, y0):# Esempio 2.1 FEDE
    points= []
    for i in range(len(ss)): 
        x = x0
        y = y0 + ss[i]
        points.append([x, y])

    return points

# LINEAR HORIZONTAL PATH
def generate_horizontal_line(x0, y0):
    points= []
    for i in range(len(ss)): 
        x = x0- ss[i]
        y = y0
        points.append([x, y])

    return points

def generate_horizontal_line_dxsx(x0, y0):
    points= []
    for i in range(len(ss)): 
        x = x0- ss[i]
        y = y0
        points.append([x, y])

    return points

def generate_horizontal_line_sxdx(x0, y0):
    points= []
    for i in range(len(ss)): 
        x = ss[i] + x0 
        y = y0
        points.append([x, y])

    return points

def generate_horizontal_L(x0, y0):
    ss_L = np.linspace(0, 0.5, 100)
    points= []
    for i in range(len(ss_L)): 
        x = ss_L[i] + x0 
        y = y0
        points.append([x, y])

    return points

def generate_horizontal_L_dxsx(x0, y0):
    ss_L = np.linspace(0, 0.5, 100)
    points= []
    for i in range(len(ss_L)): 
        x =  x0 - ss_L[i]
        y = y0
        points.append([x, y])

    return points


def generate_for_star(x0, y0):
    ss_L = np.linspace(0, 0.35, 100)
    points= []
    for i in range(len(ss_L)): 
        x = ss_L[i] + x0 
        y = y0
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


# CIRCULAR PATH
def generate_circle(x0, y0, radius):
    points = []
    #x0 = 1.2
    #y0 = 1.2
    #radius = 0.5
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

### OTHER PATHS
def generate_square():
    tmp1 = generate_horizontal_line_sxdx(x0 = 0.5, y0=1)
    tmp2 = generate_vertical_line_updown(x0 = 1.5, y0 = 1)
    tmp3 = generate_horizontal_line_dxsx(x0 = 1.5, y0 = 0)
    tmp4 = generate_vertical_line_downup(x0 = 0.5, y0 = 0)
    way_pts_cartesian = np.concatenate([tmp1, tmp2, tmp3, tmp4])
    return way_pts_cartesian

def generate_triangle():
    tmp1 = generate_line_with_slope(slope=2, intercept=0.5, x_start=0, x_end=0.5)
    tmp2 = generate_line_with_slope(slope=-2, intercept=2.5, x_start=0.5, x_end=1)
    tmp3 = generate_horizontal_line_dxsx(x0 = 1, y0 = 0.5)
    way_pts_cartesian = np.concatenate([tmp1, tmp2, tmp3])
    return way_pts_cartesian

def generate_hexagon():
    tmp1 = generate_line_with_slope(slope=1, intercept=1, x_start=0, x_end=0.5)
    tmp2 = generate_horizontal_L(x0=0.5 , y0=1.5)
    tmp3 = generate_line_with_slope(slope=-1, intercept=2.5, x_start=1, x_end=1.5)
    tmp4 = generate_line_with_slope(slope=1, intercept=-0.5, x_start=1.5, x_end=1)
    tmp5 = generate_horizontal_L_dxsx(x0 = 1, y0= 0.5)
    tmp6 = generate_line_with_slope(slope=-1, intercept=1, x_start=0.5, x_end=0)
    way_pts_cartesian = np.concatenate([tmp1, tmp2, tmp3, tmp4, tmp5, tmp6])
    return way_pts_cartesian

def generate_star():
    tmp1 = generate_for_star(x0=0 , y0=1.05)
    tmp2 = generate_line_with_slope(slope=3, intercept=0, x_start=0.35, x_end=0.5)
    tmp3 = generate_line_with_slope(slope=-3, intercept=3, x_start=0.5, x_end=0.65)
    tmp4 = generate_for_star(x0=0.65 , y0=1.05)
    tmp5 = generate_line_with_slope(slope=0.7, intercept=0.35, x_start=1, x_end=0.7)
    tmp6 = generate_line_with_slope(slope=-3, intercept=2.94, x_start=0.7, x_end=0.83)
    #tmp7 = generate_line_with_slope(slope=-0.7, intercept=1.031, x_start=0.83, x_end=0.5)
    tmp7 = generate_line_with_slope(slope=-0.7, intercept=1.031, x_start=0.83, x_end=0.5)
    tmp8 = generate_line_with_slope(slope=0.7, intercept=0.33, x_start=0.5, x_end=0.17)
    tmp9 = generate_line_with_slope(slope=3, intercept=-0.06, x_start=0.17, x_end=0.3)
    tmp10 = generate_line_with_slope(slope=-0.7, intercept=1.05, x_start=0.3, x_end=0.0)

    way_pts_cartesian = np.concatenate([tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10])
    return way_pts_cartesian
    

# def generate_110L():
#     tmp1 = generate_vertical_line(x0 = -1  , y0 = 1.5) #finisce in (-1,0.5)
#     intra1 = generate_line_with_slope(slope=2.5, intercept=3, x_start=-1, x_end=-0.6)
#     tmp2 = generate_vertical_line(x0 = -0.6, y0 = 1.5)
#     tmp3 = generate_ellipse(x0 = -0.1, y0 = 1, a = 0.3 , b= 0.5)
#     tmp4 = generate_vertical_line(x0= 0.5, y0=1.5)
#     tmp5 = generate_horizontal_L(x0=0.5 , y0=0.5)
#     way_pts_cartesian = np.concatenate([tmp1, tmp2, tmp3, tmp4, tmp5])
#     return way_pts_cartesian

### PRIMITIVE PATHS
#way_pts_cartesian = generate_custom_cubic() # Esempio 1.1 Fede
#way_pts_cartesian = generate_vertical_line(x0= 1.5, y0=1) # Esempio2.1 Fede
#way_pts_cartesian = generate_horizontal_line(x0= 1.5, y0=1)
#way_pts_cartesian = generate_horizontal_line_sxdx(x0= 0.5, y0=1)
#way_pts_cartesian = generate_vertical_line_downup(x0 = 0.5, y0 = 0.5)
# #way_pts_cartesian = generate_ellipse(1.2, 1.2, 0.3, 0.7)

### COMPOSED PATHS
#way_pts_cartesian = generate_circle(x0 = 1, y0 = 1, radius = 0.5)
way_pts_cartesian = generate_square()
#way_pts_cartesian = generate_triangle()
#way_pts_cartesian = generate_hexagon()
#way_pts_cartesian = generate_star()
#way_pts_cartesian = generate_110L()

### ATTEMPT
#way_pts_cartesian = np.array([[1.5, 1],[1.8, 0],[1.5,-1]])

#way_pts_cartesian = generate_line_with_slope(slope=1, intercept=1, x_start=0, x_end=0.5)
#way_pts_cartesian = generate_horizontal_L(x0=0.5 , y0=1.5)
#way_pts_cartesian = generate_line_with_slope(slope=3, intercept=0, x_start=0.35, x_end=0.5)
#way_pts_cartesian = generate_line_with_slope(slope=-3, intercept=3, x_start=0.5, x_end=0.65)

#way_pts_cartesian = generate_horizontal_L_dxsx(x0 = 1, y0= 0.5)

#way_pts_cartesian = generate_horizontal_line_dxsx(x0 = 1, y0 = 0.5)


way_pts_cartesian = np.array(way_pts_cartesian)

#print("Cartesian waypoints:\n", way_pts_cartesian)

#ss = np.linspace(0,0.9,100) # 1.1 Fede Esempio 
#ss = np.linspace(0,0.9,100) # Esempio 2.1 Fede

ss = np.linspace(0,1,400) ### CHANGE NUMBER OF SS DEPENDING ON THE GEOMETRIC PATH PERFORMED
cart_path = ta.SplineInterpolator(ss, way_pts_cartesian)

instance_cart = algo.TOPPRA([], cart_path)#, parametrizer="ParametrizeConstAccel"
cart_traj = instance_cart.compute_trajectory(0,0) 


# path sampling
ts_cart = np.linspace(0, cart_traj.duration, 400) ### CHANGE NUMBER OF TS DEPENDING ON THE GEOMETRIC PATH PERFORMED
ps_sample = cart_traj(ts_cart)  # cartesian path samples

#print("Cartesian position are:\n", ps_sample)
#print(ps_sample[:,1].size)


joint_way_pts = []
for i in range(len(ps_sample)):
    q1, q2 = inverse_kinematics_2R(ps_sample[i,0], ps_sample[i, 1], L1, L2)
    joint_way_pts.append([q1, q2])

print("Joint waypoints are:\n" ,len(joint_way_pts))

ss_joint = np.linspace(0, 1, len(joint_way_pts))

jnt_path = ta.SplineInterpolator(ss, joint_way_pts)

vlims = [5,5]
alims = 10 + np.random.rand(dof) * 2 

pc_vel = constraint.JointVelocityConstraint(vlims)
pc_acc = constraint.JointAccelerationConstraint(alims) 

tau_max = np.array([[-30, 30], [-30, 30]])  # Example torque limits for each joint

# Friction coefficients (example values)
fs_coef = np.random.rand(2) * 10

# Define the joint torque constraint using the inverse dynamics function
pc_tau = constraint.JointTorqueConstraint(
    inverse_dynamics_2R, tau_max, fs_coef, discretization_scheme=constraint.DiscretizationType.Interpolation
)

print(pc_vel)
print(pc_acc)

instance_jnt = algo.TOPPRA([pc_vel, pc_tau], jnt_path) #, parametrizer="ParametrizeConstAccel"
jnt_traj = instance_jnt.compute_trajectory(0,0)

print("DURATION OF OPTIMAL TRAJECTORY IS:", jnt_traj.duration)

ts_sample = np.linspace(0, jnt_traj.duration, 100)
qs_sample = jnt_traj(ts_sample)
#print("q are:\n", qs_sample)
qds_sample = jnt_traj(ts_sample, 1)
qdds_sample = jnt_traj(ts_sample, 2)

fig, axs = plt.subplots(3, 1, sharex=True)

for i in range(jnt_path.dof):
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
for i in range(0, jnt_path.dof):
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


instance_jnt.compute_feasible_sets()
instance_jnt.inspect()




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