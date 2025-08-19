import pygame
import random
import math

# Konstanta
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 300
GROUND_HEIGHT = 230

# Inisialisasi Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

# Warna
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class Dino:
    def __init__(self):
        self.image = pygame.Surface((40, 40))
        self.image.fill((0, 200, 0))  # Hijau
        self.rect = self.image.get_rect()
        self.rect.x = 50
        self.rect.y = GROUND_HEIGHT - 40

        self.velocity_y = 0
        self.is_jumping = False
        self.is_ducking = False
        self.duck_timer = 0  # Timer untuk unduck otomatis

    def jump(self):
        if not self.is_jumping and not self.is_ducking:
            self.velocity_y = -12
            self.is_jumping = True

    def duck(self):
        if not self.is_jumping:
            self.is_ducking = True
            self.duck_timer = 15  # Duck selama 15 frames
            self.image = pygame.Surface((40, 20))
            self.image.fill((0, 150, 0))
            self.rect.height = 20
            self.rect.y = GROUND_HEIGHT - 20

    def unduck(self):
        self.is_ducking = False
        self.duck_timer = 0
        self.image = pygame.Surface((40, 40))
        self.image.fill((0, 200, 0))
        self.rect.height = 40
        self.rect.y = GROUND_HEIGHT - 40

    def update(self):
        # Handle duck timer - unduck otomatis setelah beberapa frame
        if self.is_ducking:
            self.duck_timer -= 1
            if self.duck_timer <= 0:
                self.unduck()

        # Gravitasi
        self.velocity_y += 1
        self.rect.y += self.velocity_y

        if self.rect.y >= GROUND_HEIGHT - self.rect.height:
            self.rect.y = GROUND_HEIGHT - self.rect.height
            self.velocity_y = 0
            self.is_jumping = False


class Obstacle:
    def __init__(self, game_speed=8):
        self.type = random.choice(['cactus', 'bird'])
        self.speed = game_speed

        if self.type == 'cactus':
            self.image = pygame.Surface((20, 40))
            self.image.fill((200, 0, 0))  # Merah
            self.rect = self.image.get_rect()
            self.rect.y = GROUND_HEIGHT - 40
        else:
            self.image = pygame.Surface((30, 20))
            self.image.fill((0, 0, 200))  # Biru (burung)
            self.rect = self.image.get_rect()
            # Burung bisa di 2 ketinggian berbeda
            bird_heights = [GROUND_HEIGHT - 80, GROUND_HEIGHT - 100]
            self.rect.y = random.choice(bird_heights)

        self.rect.x = SCREEN_WIDTH + random.randint(20, 50)
        self.passed = False

    def update(self):
        self.rect.x -= self.speed

    def is_off_screen(self):
        return self.rect.x < -self.rect.width


class DinoGame:
    def __init__(self):
        self.dino = Dino()
        self.obstacles = []
        self.score = 0
        self.frame = 0
        self.done = False
        self.game_speed = 6  # Mulai lebih lambat
        self.spawn_timer = 0
        self.min_spawn_distance = 120
        self.max_spawn_distance = 200

    def reset(self):
        self.__init__()
        return self.get_state()

    def get_state(self):
        """State yang lebih informatif untuk Q-learning"""
        if self.obstacles:
            obs = self.obstacles[0]  # Obstacle terdekat
            distance = obs.rect.x - self.dino.rect.x

            # Normalisasi distance ke range 0-15 (lebih detail)
            distance_discrete = min(max(int(distance / 20), 0), 15)

            obstacle_type = 0 if obs.type == 'cactus' else 1

            # State dino yang lebih detail
            dino_state = 0  # ground
            if self.dino.is_jumping:
                if self.dino.velocity_y < -6:
                    dino_state = 1  # jumping up
                elif self.dino.velocity_y > 6:
                    dino_state = 2  # falling down
                else:
                    dino_state = 3  # mid jump
            elif self.dino.is_ducking:
                dino_state = 4  # ducking

            # Tambahkan informasi obstacle kedua jika ada
            second_obs_info = 0
            if len(self.obstacles) > 1:
                second_obs = self.obstacles[1]
                second_distance = second_obs.rect.x - self.dino.rect.x
                if second_distance < 300:  # Hanya jika cukup dekat
                    second_obs_info = 1 if second_obs.type == 'bird' else 2

            return (distance_discrete, obstacle_type, dino_state, second_obs_info)
        else:
            return (15, 0, 0, 0)  # No obstacle, default state

    def step(self, action):
        reward = 0
        self.frame += 1

        # Action: 0 = stay, 1 = jump, 2 = duck
        if action == 1:
            self.dino.jump()
        elif action == 2:
            self.dino.duck()
        # Untuk action 0 (stay), tidak perlu unduck paksa karena ada timer

        self.dino.update()

        # Spawn obstacle dengan timing yang lebih baik
        self.spawn_timer += 1
        spawn_interval = random.randint(70, 120)  # Variasi yang lebih besar

        if self.spawn_timer >= spawn_interval:
            # Pastikan jarak minimum dari obstacle terakhir
            if not self.obstacles or (self.obstacles[-1].rect.x < SCREEN_WIDTH - self.min_spawn_distance):
                self.obstacles.append(Obstacle(self.game_speed))
                self.spawn_timer = 0

        # Update obstacles
        for obs in self.obstacles[:]:
            obs.update()

            # Check collision dengan toleransi yang lebih baik
            if obs.rect.colliderect(self.dino.rect):
                # Collision yang lebih presisi
                overlap_x = min(obs.rect.right, self.dino.rect.right) - max(obs.rect.left, self.dino.rect.left)
                overlap_y = min(obs.rect.bottom, self.dino.rect.bottom) - max(obs.rect.top, self.dino.rect.top)

                if overlap_x > 5 and overlap_y > 5:  # Toleransi collision
                    reward = -100  # Penalty untuk collision
                    self.done = True
                    break

            # Check jika obstacle sudah dilewati
            if not obs.passed and obs.rect.x + obs.rect.width < self.dino.rect.x:
                obs.passed = True
                reward += 15  # Reward untuk melewati obstacle
                self.score += 1

        # Remove obstacles yang sudah off screen
        self.obstacles = [obs for obs in self.obstacles if not obs.is_off_screen()]

        # Reward untuk bertahan hidup
        if not self.done:
            reward += 0.2

        # Intelligent action rewards
        if self.obstacles:
            closest_obs = self.obstacles[0]
            distance = closest_obs.rect.x - self.dino.rect.x

            # Reward yang lebih nuanced berdasarkan timing
            if 50 < distance < 150:  # Sweet spot untuk action
                if closest_obs.type == 'cactus' and action == 1:  # Jump untuk cactus
                    if 80 < distance < 120:
                        reward += 5  # Perfect timing
                    else:
                        reward += 2  # Good timing
                elif closest_obs.type == 'bird' and action == 2:  # Duck untuk bird
                    if 70 < distance < 110:
                        reward += 5  # Perfect timing
                    else:
                        reward += 2  # Good timing
                elif action == 0:  # Stay saat aman
                    if distance > 130:
                        reward += 1  # Good to wait

            # Penalty untuk action yang salah pada jarak kritis
            elif 30 < distance < 80:  # Danger zone
                if closest_obs.type == 'cactus' and action == 2:
                    reward -= 3  # Duck saat ada cactus dekat
                elif closest_obs.type == 'bird' and action == 1:
                    reward -= 3  # Jump saat ada bird dekat

        # Progressive difficulty
        if self.score > 0 and self.score % 5 == 0:
            self.game_speed = min(12, 6 + self.score // 5)  # Increase speed gradually

        # Bonus untuk score tinggi
        if self.score >= 10:
            reward += 2  # Bonus untuk mencapai 10+
        if self.score >= 20:
            reward += 3  # Bonus ekstra untuk 20+

        return self.get_state(), reward, self.done

    def render(self):
        screen.fill(WHITE)

        # Gambar ground line
        pygame.draw.line(screen, BLACK, (0, GROUND_HEIGHT), (SCREEN_WIDTH, GROUND_HEIGHT), 2)

        # Gambar dino
        screen.blit(self.dino.image, self.dino.rect)

        # Gambar obstacles
        for obs in self.obstacles:
            screen.blit(obs.image, obs.rect)

        # Tampilkan score dan info
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, BLACK)
        screen.blit(score_text, (10, 10))

        # Tampilkan speed
        speed_text = font.render(f"Speed: {self.game_speed}", True, BLACK)
        screen.blit(speed_text, (10, 50))

        # Tampilkan state untuk debugging
        state = self.get_state()
        state_text = font.render(f"State: {state}", True, BLACK)
        screen.blit(state_text, (10, 90))

        pygame.display.update()

    def close(self):
        pygame.quit()