import numpy as np
from numpy.typing import ArrayLike
from typing import List, Union
import plotly.express as ex
import plotly.graph_objects as go
from copy import deepcopy

from .target import Target


__all__ = ['AutopilotTarget', 'Straight', 'Turn']


M_180_PI = 180 / np.pi
M_PI_180 = np.pi / 180


class Straight:
    def __init__(self, distance_m: float):
        self.distance_m = distance_m
        self.p0 = np.zeros(3)
        self.p1 = np.zeros(3)
        self.progress = 0
    
    def __repr__(self):
        return repr(dict(p0=list(self.p0), p1=list(self.p1), distance_m=self.distance_m,
                         progress=self.progress))


class Turn:
    def __init__(self, radius_m: float, turn_deg: float, left: bool = True):
        self.radius_m = radius_m
        self.turn_deg = turn_deg
        self.left = left
        self.center = np.zeros(3)
        self.alpha0 = 0
        self.alpha1 = 0
        self.progress = 0
    
    def __repr__(self):
        return repr(dict(center=list(self.center), alpha0=self.alpha0, alpha1=self.alpha1,
                         turn_deg=self.turn_deg, radius_m=self.radius_m, left=self.left,
                         progress=self.progress))



# fill in coordinates of segments of a path
#
# path is provided by the user as a sequence of abstract segments: straights and arcs;
# the next immediate requirements is to fill in their specific coordinates: (1) the
# start and end of a straight and (2) the center of a turn arc and its start and end
# angles
#
# pos - initial position
# heading - initial heading
# commands - autopilot path
def _populate_coordinates(pos: ArrayLike, heading: float, commands: List[Union[Straight, Turn]]) -> List[Union[Straight, Turn]]:
    pos = np.asarray(pos)
     
    for command in commands:
        if isinstance(command, Straight):
            command.p0 = pos

            pos = pos + _heading_to_rotation(heading) @ np.array([command.distance_m, 0, 0])
            command.p1 = pos
        
        elif isinstance(command, Turn):
            radius = np.array([command.radius_m, 0, 0])
            turn   = command.turn_deg / 180.0 * np.pi

            if command.left:
                command.alpha0 = heading - np.pi/2
                command.center = pos + _heading_to_rotation(heading + np.pi/2) @ radius
                heading += turn
                command.alpha1 = heading - np.pi/2

            else:
                command.alpha0 = heading + np.pi/2
                command.center = pos + _heading_to_rotation(heading - np.pi/2) @ radius                
                heading -= turn
                command.alpha1 = heading + np.pi/2

            pos = command.center + np.asarray([np.cos(command.alpha1), np.sin(command.alpha1), 0]) * command.radius_m

        else:
            raise Exception("Unsupported command")
        
    return commands


# https://en.wikipedia.org/wiki/Rotation_matrix
def _heading_to_rotation(theta_rad: float):
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    return np.array([[cos_theta, -sin_theta, 0],
                     [sin_theta,  cos_theta, 0],
                     [0,          0,         1]])


def _normalize(x: ArrayLike) -> np.ndarray:
    x = np.asarray(x)
    x_norm = np.linalg.norm(x)
    if x_norm == 0:
        return None
    return x / x_norm


# find a point on the path closest to the given position
#
# delta_dist = how much does the mover move in a single time step
def _project_on_path(position: ArrayLike, commands: List[Union[Straight, Turn]], delta_dist: float) -> np.ndarray:
    # ignore the altitude dimension
    position = np.asarray(position).copy()
    position[2] = 0
    
    d_min = np.inf
    ans   = None

    for command in commands:
        if isinstance(command, Straight):
            dp = command.p1 - command.p0
            t  = np.dot(dp, position - command.p0) / np.dot(dp, dp)
            t  = np.clip(t, 0, 1)
            p  = command.p0 + t * dp
            d  = np.linalg.norm(p - position)
            x  = _normalize(dp)
            y  = _normalize(position - p)
            
            # mover is already on the path
            if y is None:
                y = np.array([x[1], -x[0], x[2]])

            u = delta_dist / np.linalg.norm(dp)
            q = p + u * dp

        elif isinstance(command, Turn):
            # find direction vector from center of the arc to the queried position,
            # ignore difference in altitude
            pos = np.copy(position)
            pos[2] = command.center[2]
            direction = pos - command.center

            # find the angle alpha on the arc closest to the current position of the mover
            cos_alpha = np.dot(direction, np.array([1, 0, 0])) / np.linalg.norm(direction)
            alpha = np.arccos(cos_alpha)

            if direction[1] < 0:
                alpha = -alpha

            a0, a1 = np.sort([command.alpha0, command.alpha1])
            alpha = np.clip(alpha, a0, a1)

            # now find the point on the arc corresponding to the angle alpha; r goes
            # from center of the turn to position
            r = _heading_to_rotation(alpha) @ np.array([command.radius_m, 0, 0])
            p = command.center + r
            d = np.linalg.norm(p - position)

            # y axis of a coordinate system tangential to the arc in point p
            y = _normalize(position - p)

            # x axis of that coordinate system; x is perpendicular to y and
            # points in the direction of movement
            if command.left:
                x = _normalize([-r[1], r[0], 0])
            else:
                x = _normalize([r[1], -r[0], 0])
            
            # the next point for the mover to aim at
            beta = delta_dist / command.radius_m
            if not command.left:
                beta = -beta
            r = _heading_to_rotation(alpha + beta) @ np.array([command.radius_m, 0, 0])
            q = command.center + r


        else:
            raise Exception("Unsupported command")

        if d < d_min:
            d_min = d
            ans = d, p, q, x, y

    return ans



def _plot_path(commands: List[Union[Straight, Turn]], position: ArrayLike = None, a: float = 1, delta_dist: float = 1):
    x, y = [], []
    pts_x, pts_y = [], []

    for command in commands:
        if isinstance(command, Straight):
            x.extend([command.p0[0], command.p1[0]])
            y.extend([command.p0[1], command.p1[1]])

            pts_x.extend([command.p0[0], command.p1[0]])
            pts_y.extend([command.p0[1], command.p1[1]])
        
        elif isinstance(command, Turn):
            alpha = np.linspace(command.alpha0, command.alpha1, 50)

            x.extend(list(np.cos(alpha) * command.radius_m + command.center[0]))
            y.extend(list(np.sin(alpha) * command.radius_m + command.center[1]))

        else:
            raise Exception("Unsupported command")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, name='path'))
    fig.add_trace(go.Scatter(x=pts_x, y=pts_y, mode='markers', name='waypoints'))

    # if the current position of the mover is provide, plot it against its hyperbolic
    # trajectory
    if position is not None:
        d, p, q, x_unit, y_unit = _project_on_path(position, commands, delta_dist)

        # center of the coordinate system, shifted back relative to the current
        # position of the mover by distance x = a/d (derived from the hyperbole
        # equation y = a/x)
        O = p - x_unit * a/d

        # points on the hyperbole
        x = np.linspace(a/(d*3), a, 10)
        x = np.sort(np.concatenate((x, a/x)))
        y = a/x
        points = np.array((x, y)).T
        points = O + points @ np.array((x_unit, y_unit))

        fig.add_trace(go.Scatter(x=points[:,0], y=points[:,1], mode='lines', name='trajectory'))

        # plot axes scaled by the max size of the hyperbole
        x_axis = O + x_unit * d * 3
        y_axis = O + y_unit * d * 3
        fig.add_trace(go.Scatter(x=[y_axis[0], O[0], x_axis[0]], y=[y_axis[1], O[1], x_axis[1]], mode='lines', name='axes'))
        fig.add_trace(go.Scatter(x=[position[0]], y=[position[1]], mode='markers', name='current position'))
        
        # the next point on the arc where the mover is going
        fig.add_trace(go.Scatter(x=[q[0]], y=[q[1]], mode='markers', name='next position'))

    fig.update_layout(height=700)

    return fig





class AutopilotTarget(Target):
    def __init__(self, commands: List[Union[Straight, Turn]], max_turn_rate_deg_s: float = 5,
                 mass_kg: float = 1000, thrust_N: float = 2000, drag_coefficient: float = 0.05,
                 noise_intensity: float = .01, integration_steps_count: int = 20,
                 seed: int = None):
        
        self.initial_position = np.array([0, 0, 0])
        self.initial_heading  = 0
        self.commands = _populate_coordinates(self.initial_position, self.initial_heading, deepcopy(commands))
        
        self.noise_intensity = noise_intensity
        self.mass_kg = mass_kg
        self.thrust_N = thrust_N
        self.drag_coefficient = drag_coefficient
        self.integration_steps_count = integration_steps_count
        self.max_turn_rate_deg_s = max_turn_rate_deg_s

        self.seed = seed
        
        if self.seed is None:
            self.name = f"autopilot"
        else:
            self.name = f"autopilot_{self.seed}"


    def true_states(self, T: Union[float, ArrayLike] = 1, n: int = 400, seed: int = None, debug: bool = False) -> np.ndarray:
        if seed is None:
            seed = self.seed
        
        if isinstance(T, (int, float)):
            time = np.arange(0, n, T)
        else:
            T = 1 # TODO better take the most frequent value in np.diff(time)
            time = np.array(T)
        

        rnd = np.random.default_rng(seed=seed)
        dt = T / self.integration_steps_count
        mover = ForceWaypointMover(self.initial_position, [1, 0, 0], self.mass_kg, self.thrust_N,
                                   self.drag_coefficient, self.max_turn_rate_deg_s,
                                   np.sqrt(self.noise_intensity * dt))

        segment_index = 0

        trace = []
        for _ in time:
            if segment_index == len(self.commands):
                break

            segment = self.commands[segment_index]
            
            progress, path_pos, next_target = _project_on_segment(mover.position, segment, np.linalg.norm(mover.velocity) * T)
            segment.progress = max(segment.progress, progress)

            if segment.progress >= 1:
                segment_index += 1
                continue

            if debug:
                trace.append(np.concatenate(([segment_index, segment.progress], mover.position, path_pos, mover.position - path_pos, mover.velocity)))
            else:
                trace.append(np.concatenate((mover.position, mover.velocity)))


            for _ in range(self.integration_steps_count):
                waypoint_reached = mover.update(dt, next_target, rnd)
                if waypoint_reached:
                    break # TODO approximation: should continue movement in the last established direction

            # TODO check if reached the end of the path                    

        return np.asarray(trace)




def _project_on_segment(position: ArrayLike, segment: Union[Straight, Turn], delta_distance: float) -> float:
    # ignore the altitude dimension
    position = np.asarray(position).copy()
    position[2] = 0
    
    if isinstance(segment, Straight):
        dp = segment.p1 - segment.p0
        t  = np.dot(dp, position - segment.p0) / np.dot(dp, dp)
        p  = segment.p0 + t * dp

        u = t + delta_distance / np.linalg.norm(dp)
        q = segment.p0 + u * dp

    elif isinstance(segment, Turn):
        # find direction vector from center of the arc to the queried position,
        # ignore difference in altitude
        direction = position - segment.center
        direction[2] = 0

        # find the angle alpha on the arc closest to the current position of the mover
        cos_alpha = np.dot(direction, np.array([1, 0, 0])) / np.linalg.norm(direction)
        alpha = np.arccos(cos_alpha)

        # if displacement in Y is negative, change the direction of rotation; this
        # is because arccos is symmetrical and maps [-1, 1] -> [0, 180]
        if direction[1] < 0:
            alpha = -alpha

        # TODO normalize by 2*pi but shift the [a0, a1] interval such that
        #      its mid-point is in x=pi

        s = alpha / (2 * np.pi)
        s0, s1 = np.sort([segment.alpha0, segment.alpha1]) / (2 * np.pi)
        s = np.clip(s, s0, s1)

        alpha = s * 2 * np.pi
        p = segment.center + _heading_to_rotation(alpha) @ np.array([segment.radius_m, 0, 0])

        t = (alpha - segment.alpha0) / (segment.alpha1 - segment.alpha0)

        # the next point for the mover to aim at
        beta = delta_distance / segment.radius_m
        if not segment.left:
            beta = -beta
        
        q = segment.center + _heading_to_rotation(alpha + beta) @ np.array([segment.radius_m, 0, 0])

    return t, p, q



class ForceWaypointMover:
    def __init__(self, initial_position: ArrayLike, initial_velocity: ArrayLike,
                 mass: float, thrust_force: float, drag_coefficient: float,
                 max_turn_rate_deg_s: float = 5, noise_intensity: float = 1.0):
        """
        Initialize the air vehicle mover.
        
        :param initial_position: Initial 3D position vector (x, y, z)
        :param initial_velocity: Initial 3D velocity vector (vx, vy, vz)
        :param mass: Mass of the vehicle in kg
        :param thrust_force: Maximum thrust force in Newtons
        :param drag_coefficient: Drag coefficient (dimensionless)
        """
        self.position = np.asarray(initial_position, dtype=float)
        self.velocity = np.asarray(initial_velocity, dtype=float)
        self.mass = mass
        self.thrust_force = thrust_force
        self.drag_coefficient = drag_coefficient
        self.noise_intensity = noise_intensity
        self.max_turn_rate = max_turn_rate_deg_s / 180 * np.pi # radians / s
        
        self.previous_waypoint = np.asarray(initial_position, dtype=float)

        self.cos_theta = None
                
    def update(self, dt: float, target: ArrayLike, rnd: np.random.Generator = np.random):
        """
        Update the vehicle's position and velocity for one time step.

        :param dt: Time step in seconds
        :param requested_heading: Desired heading as a unit vector (x, y, z)
        :param randomness_scale: Scale for random acceleration, affects how much the velocity changes randomly
        """
        
        # Calculate direction towards target
        direction_to_target = target - self.position
        distance_to_target = np.linalg.norm(direction_to_target)
        
        # Normalize direction vector
        if distance_to_target == 0:
            return True
        
        direction_to_target /= distance_to_target

        velocity_norm      = np.linalg.norm(self.velocity)
        velocity_direction = self.velocity / velocity_norm

        cos_theta = np.dot(velocity_direction, direction_to_target)
        theta = np.arccos(cos_theta)

        if theta < self.max_turn_rate * dt:
            self.velocity = direction_to_target * velocity_norm
            velocity_direction = direction_to_target
        
        else:
            theta = self.max_turn_rate * dt
            cos_theta = np.cos(theta)

            axis = np.cross(velocity_direction, direction_to_target)
            self.velocity = self.velocity * cos_theta + np.cross(axis, self.velocity) * np.sin(theta) + \
                            axis * np.dot(axis, self.velocity) * (1 - cos_theta)

            velocity_direction = self.velocity / velocity_norm


        # Calculate drag force
        drag_force = -0.5 * self.drag_coefficient * velocity_norm * self.velocity
        
        # Random acceleration (normally distributed)
        random_acceleration = rnd.normal(0, self.noise_intensity, 3)
        
        # Net acceleration
        net_acceleration = (self.thrust_force * velocity_direction / self.mass + 
                            drag_force / self.mass + 
                            random_acceleration)
        
        # Update velocity, trying to move towards the target
        self.velocity += net_acceleration * dt

        # Update position
        self.position += self.velocity * dt
        
        # the normal vector for the plane which defines the end of this leg
        waypoint_plane_normal = target - self.previous_waypoint
        waypoint_plane_normal_norm = np.linalg.norm(waypoint_plane_normal)

        # vector from the current position to the target
        to_target = target - self.position
        to_target_norm = np.linalg.norm(to_target)

        # if either norm is zero, cosinus will be zero as well
        self.cos_theta = np.dot(waypoint_plane_normal, to_target)
        if waypoint_plane_normal_norm > 0 and to_target_norm > 0:
            self.cos_theta = self.cos_theta / waypoint_plane_normal_norm / to_target_norm

        # if cosinus between these two vectors is negative, we moved past the target
        if self.cos_theta < 0:
            self.previous_waypoint = target.copy()
            return True
        
        return False




# --- old code ---









def _project(p, l0, l1):
    p, l0, l1 = np.asarray(p), np.asarray(l0), np.asarray(l1)
    line = l1 - l0
    
    np_dot_line = np.dot(line, line)
    if np_dot_line == 0:
        return l0

    # L = dot(p-l0, line) / np.linarg.norm(line) is the length of the projection of p-l0 onto line
    # X = line / np.linarg(line) is the direction unit vector of line
    # P = l0 + X*L is the projection of point 'p' onto line
    # norm used twice is collapsed into dot(line, line)
    return l0 + np.dot(p - l0, line) / np_dot_line * line





def _generate_waypoints(commands: List[Union[Straight, Turn]]):
    current_wp = np.array([0, 0, 0])
    current_heading = 0

    waypoints = [current_wp]
    
    for command in commands:
        if isinstance(command, Straight):
            current_wp = current_wp + np.array([command.distance_m, 0, 0]) @ _heading_to_rotation(current_heading)
            waypoints.append(current_wp)
        
        elif isinstance(command, Turn):
            if command.left:
                center = current_wp + np.array([command.radius_m, 0, 0]) @ _heading_to_rotation(current_heading - np.pi/2)
                alpha  = current_heading - np.pi/2 + np.arange(0, command.turn_deg / 180.0 * np.pi, 0.1)
                heading_change = command.turn_deg / 180.0 * np.pi
            else:
                center = current_wp + np.array([command.radius_m, 0, 0]) @ _heading_to_rotation(current_heading + np.pi/2)
                alpha  = current_heading + np.pi/2 - np.arange(0, command.turn_deg / 180.0 * np.pi, 0.1)
                heading_change = -command.turn_deg / 180.0 * np.pi
            
            turn_wps = np.array((command.radius_m * np.cos(alpha),
                                 command.radius_m * np.sin(alpha),
                                 np.zeros_like(alpha))).T
            turn_wps += center
            waypoints.extend(list(turn_wps))

            current_wp = waypoints[-1]
            current_heading += heading_change

        else:
            raise Exception("Unsupported command")
    
    return waypoints






