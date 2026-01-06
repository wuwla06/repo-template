import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import warnings

warnings.filterwarnings('ignore')


class BilliardBall:
    def __init__(self, x, y, radius, color, speed, angle):
        self.radius = radius
        self.color = color
        self.x = x
        self.y = y
        self.vx = speed * np.cos(np.radians(angle))
        self.vy = speed * np.sin(np.radians(angle))
        self.trajectory = [(x, y)]
        self.stopped = False
        self.mu = 0.2

    def update_position(self, dt):
        if self.stopped:
            return
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.trajectory.append((self.x, self.y))

    def apply_friction(self, dt):
        if self.stopped:
            return
        speed = np.sqrt(self.vx ** 2 + self.vy ** 2)
        if speed < 0.02:
            self.vx = 0
            self.vy = 0
            self.stopped = True
            return
        factor = np.exp(-self.mu * dt)
        self.vx *= factor
        self.vy *= factor

    def check_wall_collision(self, table_width, table_height):
        if self.stopped:
            return
        if self.x - self.radius <= 0:
            self.x = self.radius
            self.vx = abs(self.vx)
        elif self.x + self.radius >= table_width:
            self.x = table_width - self.radius
            self.vx = -abs(self.vx)
        if self.y - self.radius <= 0:
            self.y = self.radius
            self.vy = abs(self.vy)
        elif self.y + self.radius >= table_height:
            self.y = table_height - self.radius
            self.vy = -abs(self.vy)


class BilliardTable:
    def __init__(self, width=10, height=6):
        self.width = width
        self.height = height
        self.balls = []
        self.colors = plt.cm.rainbow(np.linspace(0, 1, 16))

    def setup_balls_triangle(self, num_balls, speeds, angles):
        self.balls = []
        ball_radius = 0.3
        if num_balls == 1:
            x = self.width / 2
            y = self.height / 2
            self.balls.append(BilliardBall(x, y, ball_radius, self.colors[0], speeds[0], angles[0]))
            return
        rows = 1
        while rows * (rows + 1) // 2 < num_balls:
            rows += 1
        start_x = self.width * 0.7
        start_y = self.height / 2
        ball_index = 0
        for row in range(rows):
            y_offset = row * ball_radius * 2
            balls_in_row = row + 1
            x_offset = -row * ball_radius
            for col in range(balls_in_row):
                if ball_index >= num_balls:
                    break
                x = start_x + x_offset + col * ball_radius * 2
                y = start_y + y_offset - row * ball_radius
                if x + ball_radius > self.width - 1:
                    x = self.width - ball_radius - 1
                if y + ball_radius > self.height - 1:
                    y = self.height - ball_radius - 1
                if y - ball_radius < 1:
                    y = ball_radius + 1
                ball = BilliardBall(x, y, ball_radius, self.colors[ball_index % len(self.colors)], speeds[ball_index],
                                    angles[ball_index])
                self.balls.append(ball)
                ball_index += 1

    def handle_ball_collisions(self):
        for i in range(len(self.balls)):
            if self.balls[i].stopped:
                continue
            for j in range(i + 1, len(self.balls)):
                if self.balls[j].stopped:
                    continue
                dx = self.balls[j].x - self.balls[i].x
                dy = self.balls[j].y - self.balls[i].y
                distance = np.sqrt(dx ** 2 + dy ** 2)
                if distance < 2 * self.balls[i].radius:
                    if distance > 0:
                        nx = dx / distance
                        ny = dy / distance
                    else:
                        nx, ny = 1, 0
                    v1n = self.balls[i].vx * nx + self.balls[i].vy * ny
                    v2n = self.balls[j].vx * nx + self.balls[j].vy * ny
                    v1n_new = v2n
                    v2n_new = v1n
                    self.balls[i].vx += (v1n_new - v1n) * nx
                    self.balls[i].vy += (v1n_new - v1n) * ny
                    self.balls[j].vx += (v2n_new - v2n) * nx
                    self.balls[j].vy += (v2n_new - v2n) * ny
                    overlap = 2 * self.balls[i].radius - distance
                    if distance > 0:
                        self.balls[i].x -= overlap * nx * 0.5
                        self.balls[i].y -= overlap * ny * 0.5
                        self.balls[j].x += overlap * nx * 0.5
                        self.balls[j].y += overlap * ny * 0.5

    def update(self, dt):
        for ball in self.balls:
            ball.update_position(dt)
            ball.apply_friction(dt)
            ball.check_wall_collision(self.width, self.height)
        self.handle_ball_collisions()

    def all_balls_stopped(self):
        return all(ball.stopped for ball in self.balls)

    def get_total_path_length(self):
        total = 0
        for i, ball in enumerate(self.balls):
            path = 0
            for j in range(1, len(ball.trajectory)):
                x1, y1 = ball.trajectory[j - 1]
                x2, y2 = ball.trajectory[j]
                path += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            total += path
        return total


class BilliardAnimation:
    def __init__(self, table):
        self.table = table
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.time = 0
        self.animation_running = True
        self.setup_plot()

    def setup_plot(self):
        self.ax.set_xlim(-0.5, self.table.width + 0.5)
        self.ax.set_ylim(-0.5, self.table.height + 0.5)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('darkgreen')
        self.ax.set_title('Бильярд: Моделирование движения шаров')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        border = plt.Rectangle((0, 0), self.table.width, self.table.height, fill=False, edgecolor='brown', linewidth=3)
        self.ax.add_patch(border)
        self.ball_patches = []
        for ball in self.table.balls:
            patch = Circle((ball.x, ball.y), ball.radius, color=ball.color, zorder=10)
            self.ax.add_patch(patch)
            self.ball_patches.append(patch)
        self.trajectory_lines = []
        for ball in self.table.balls:
            line, = self.ax.plot([], [], '-', alpha=0.5, linewidth=1)
            self.trajectory_lines.append(line)
        self.time_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def update(self, frame):
        if not self.animation_running:
            return self.ball_patches + self.trajectory_lines + [self.time_text]
        dt = 0.05
        self.table.update(dt)
        self.time += dt
        for i, (ball, patch) in enumerate(zip(self.table.balls, self.ball_patches)):
            patch.center = (ball.x, ball.y)
            if len(ball.trajectory) > 1:
                xs, ys = zip(*ball.trajectory)
                self.trajectory_lines[i].set_data(xs, ys)
        moving_balls = sum(1 for ball in self.table.balls if not ball.stopped)
        self.time_text.set_text(f'Время: {self.time:.1f} сек\nДвижущихся шаров: {moving_balls}/{len(self.table.balls)}')
        if self.table.all_balls_stopped():
            self.animation_running = False
            self.show_results()
        return self.ball_patches + self.trajectory_lines + [self.time_text]

    def show_results(self):
        print("\n" + "=" * 50)
        print("ИТОГИ МОДЕЛИРОВАНИЯ")
        print("=" * 50)
        total_path = self.table.get_total_path_length()
        for i, ball in enumerate(self.table.balls):
            path = 0
            for j in range(1, len(ball.trajectory)):
                x1, y1 = ball.trajectory[j - 1]
                x2, y2 = ball.trajectory[j]
                path += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            print(f"Шар {i + 1}: путь = {path:.2f} единиц")
        print(f"\nОБЩИЙ ПУТЬ ВСЕХ ШАРОВ: {total_path:.2f} единиц")
        print("=" * 50)


def get_valid_input(prompt, input_type=float, min_val=None, max_val=None):
    while True:
        try:
            value = input_type(input(prompt))
            if min_val is not None and value < min_val:
                print(f"Значение должно быть не меньше {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Значение должно быть не больше {max_val}")
                continue
            return value
        except ValueError:
            print("Ошибка ввода. Попробуйте снова.")


def main():
    print("МОДЕЛИРОВАНИЕ БИЛЬЯРДА")
    print("=" * 50)
    num_balls = get_valid_input("Количество шаров (1-15): ", int, 1, 15)
    speeds = []
    angles = []
    print("\nВведите параметры для каждого шара:")
    for i in range(num_balls):
        print(f"\nШар {i + 1}:")
        speed = get_valid_input("  Скорость (0.1-10): ", float, 0.1, 10)
        angle = get_valid_input("  Направление в градусах (0-360): ", float, 0, 360)
        speeds.append(speed)
        angles.append(angle)
    table = BilliardTable()
    table.setup_balls_triangle(num_balls, speeds, angles)
    anim = BilliardAnimation(table)
    animation = FuncAnimation(anim.fig, anim.update, frames=1000, interval=50, blit=True, repeat=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()