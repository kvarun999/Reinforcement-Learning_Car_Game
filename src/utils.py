import math
from src.utils import line_intersection

class Car:
    def __init__(self, config: dict):
        env_params = config["env_params"]
        self.config = config # Store config for cast_rays
        self.x = 0
        self.y = 0
        self.angle = 0
        self.velocity = 0

        # Required by logic but missing in some config versions
        self.max_velocity = env_params["max_velocity"]
        self.acceleration_rate = env_params.get("acceleration_rate", 0.5)
        self.turn_rate = env_params.get("turn_rate", 5.0)
        self.friction = env_params.get("friction", 0.98)

    def accelerate(self):
        self.velocity += self.acceleration_rate
        self.velocity = min(self.velocity, self.max_velocity)

    def brake(self):
        self.velocity -= self.acceleration_rate * 1.5
        self.velocity = max(self.velocity, 0)

    def turn(self, direction: int):
        if self.velocity > 0.1:
            self.angle += direction * self.turn_rate
        self.angle %= 360

    def update(self):
        self.velocity *= self.friction
        if self.velocity < 0.01:
            self.velocity = 0

        radians = math.radians(self.angle)
        self.x += math.cos(radians) * self.velocity
        self.y += math.sin(radians) * self.velocity

    def cast_rays(self, walls):
        """Calculates normalized LIDAR distances."""
        rays = []
        num_rays = self.config["env_params"]["num_rays"]
        max_dist = self.config["env_params"]["max_ray_distance"]
        spread = self.config["env_params"].get("ray_spread", 180)
        
        start_angle = self.angle - spread / 2
        angle_step = spread / (num_rays - 1) if num_rays > 1 else 0

        for i in range(num_rays):
            ray_angle = math.radians(start_angle + i * angle_step)
            end_x = self.x + math.cos(ray_angle) * max_dist
            end_y = self.y + math.sin(ray_angle) * max_dist
            
            closest_dist = max_dist
            for wall_start, wall_end in walls:
                hit = line_intersection((self.x, self.y), (end_x, end_y), wall_start, wall_end)
                if hit:
                    dist = math.sqrt((hit[0] - self.x)**2 + (hit[1] - self.y)**2)
                    closest_dist = min(closest_dist, dist)
            
            rays.append(closest_dist / max_dist) # Normalize
        return rays