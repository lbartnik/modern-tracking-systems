import numpy as np
from numpy.typing import ArrayLike
from typing import List, Union
from copy import deepcopy

from .target import Target


__all__ = ['AutopilotTarget', 'Straight', 'Turn']


M_180_PI = 180 / np.pi
M_PI_180 = np.pi / 180


class Straight:
    def __init__(self, distance_m: float):
        self.distance_m = distance_m


class Turn:
    def __init__(self, radius_m: float, turn_deg: float, left: bool = True):
        self.radius_m = radius_m
        self.turn_deg = turn_deg
        self.left = left



class AutopilotTarget(Target):
    def __init__(self, commands: List[Union[Straight, Turn]], mass_kg: float = 1500,
                 thrust_N: float = 2000, drag_coefficient: float = 0.05,
                 noise_intensity: float = 1, integration_steps_count: int = 20,
                 seed: int = None):
        
        self.waypoints = _generate_waypoints(commands)
        
        self.noise_intensity = noise_intensity
        self.mass_kg = mass_kg
        self.thrust_N = thrust_N
        self.drag_coefficient = drag_coefficient
        self.integration_steps_count = integration_steps_count

        self.seed = seed
        
        if self.seed is None:
            self.name = f"autopilot"
        else:
            self.name = f"autopilot_{self.seed}"


    def true_states(self, T: Union[float, ArrayLike] = 1, n: int = 400, seed: int = None, only_noise: bool = False) -> np.ndarray:
        if seed is None:
            seed = self.seed
        
        if isinstance(T, (int, float)):
            time = np.arange(0, n, T)
        else:
            T = 1 # TODO better take the most frequent value in np.diff(time)
            time = np.array(T)
        

        rnd = np.random.default_rng(seed=seed)
        dt = T / self.integration_steps_count
        mover = ForceWaypointMover([0, 0, 0], [1, 0, 0], self.mass_kg, self.thrust_N, self.drag_coefficient, np.sqrt(self.noise_intensity * dt))

        waypoints = deepcopy(self.waypoints)
        prev_wp = next_wp = waypoints.pop(0)

        trace = []
        for _ in time:
            for _ in range(self.integration_steps_count):
                waypoint_reached = mover.update(dt, next_wp, rnd)
                if waypoint_reached:
                    break
            
            if waypoint_reached and len(waypoints) > 0:
                prev_wp = next_wp
                next_wp = waypoints.pop(0)
                
            if only_noise:
                path_pos = _project(mover.position, next_wp, prev_wp)
                trace.append(np.concatenate((mover.position, path_pos, mover.position - path_pos)))
            else:
                trace.append(np.concatenate((mover.position, mover.velocity)))

            if waypoint_reached and len(waypoints) == 0:
                break
                    

        return np.asarray(trace)




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


# https://en.wikipedia.org/wiki/Rotation_matrix
def _heading_to_rotation(theta_rad: float):
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    return np.array([[cos_theta, -sin_theta, 0],
                     [sin_theta,  cos_theta, 0],
                     [0,          0,         1]])


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


class ForceWaypointMover:
    def __init__(self, initial_position: ArrayLike, initial_velocity: ArrayLike,
                 mass: float, thrust_force: float, drag_coefficient: float, noise_intensity: float = 1.0):
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
        
        # Calculate drag force
        drag_force = -0.5 * self.drag_coefficient * velocity_norm * self.velocity
        
        # Random acceleration (normally distributed)
        random_acceleration = rnd.normal(0, self.noise_intensity, 3)
        
        # Net acceleration
        net_acceleration = (self.thrust_force * direction_to_target / self.mass + 
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

    def simulate(self, waypoints, dt=.1, t_max=200, seed=0):
        rnd = np.random.default_rng(seed=seed)

        trace = []
        t     = 0

        for wp in map(np.asarray, waypoints):
            # iterate until you you move past the plane whose normal is defined by the
            # previous position and the waypoint
            while t <= t_max:
                reached_target = self.update(dt, wp, rnd=rnd)
                trace.append(np.concatenate(((t, self.cos_theta), self.position.copy(), self.velocity.copy())))
                t += dt
                
                if reached_target:
                    break
            
            if t > t_max:
                break

        return np.asarray(trace)
