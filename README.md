Robotic manipulators are indispensable in modern industrial environments, perform-
ing tasks that demand precision, speed, and adaptability.
Within these real-world workspaces, particularly in production plants, robots often
operate free from constraints such as obstacles, allowing them to traverse their entire
workspace to complete tasks.
A fundamental category of robotic motion involves Point-to-Point (P2P) tasks, where
the robot transition happens between two configurations in joint space without a
predefined path. Unlike Cartesian paths, where the motion is explicitly defined
in workspace coordinates, P2P tasks provide greater flexibility in choosing the
joint-space trajectory.
This research addresses the challenge of finding the time-optimal trajectory by
leveraging joint-space waypoints to guide the robot’s motion. Starting from an initial
configuration to a final configuration, waypoints are strategically introduced along
the joint-space path.
These waypoints provide a framework for optimizing the trajectory’s time parame-
terization, with the ultimate goal of achieving the shortest possible execution time.
The process is iterative, adjusting waypoints to further reduce total movement time.