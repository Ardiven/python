import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
import math


# Data structures
@dataclass
class Item:
    id: int
    name: str
    weight: float  # kg
    destination: str
    dimension_category: int  # 0=kecil, 1=menengah, 2=besar
    volume: float  # m3


@dataclass
class Truck:
    id: int
    plate_number: str
    fuel_ratio: float  # liter/km
    max_capacity: float  # kg
    box_volume: float  # m3
    current_load: float = 0
    current_volume: float = 0
    assigned_items: list = None

    def __post_init__(self):
        if self.assigned_items is None:
            self.assigned_items = []


# Optimized Environment
class OptimizedTruckLoadingEnvironment:
    def __init__(self, fixed_items=None):
        # City distances (km) from origin
        self.city_distances = {
            'Jakarta': 0,
            'Bandung': 150,
            'Semarang': 450,
            'Surabaya': 800,
            'Yogyakarta': 560,
            'Solo': 520,
            'Malang': 850
        }

        # Dimension pricing multiplier
        self.dimension_multipliers = {0: 1.0, 1: 1.5, 2: 2.0}

        # Fuel cost per liter
        self.fuel_cost_per_liter = 10000  # IDR

        # Base price per kg per km
        self.base_price_per_kg_km = 500  # IDR

        # Reward scaling factors
        self.profit_scale = 1e-6  # Scale profits to reasonable reward range
        self.penalty_scale = 0.1

        self.fixed_items = fixed_items
        self.reset()

    def reset(self):
        """Reset environment for new episode"""
        # Initialize trucks
        self.trucks = [
            Truck(0, "B1234AB", 0.15, 5000, 25.0),
            Truck(1, "B5678CD", 0.15, 5000, 25.0),
            Truck(2, "B9012EF", 0.15, 5000, 25.0),
            Truck(3, "B3456GH", 0.15, 5000, 25.0)
        ]

        # Generate random items for the day
        self.items = self._generate_daily_items()
        self.current_item_idx = 0
        self.total_profit = 0
        self.total_revenue = 0
        self.total_cost = 0
        self.done = False
        self.items_processed = 0
        self.items_assigned = 0
        self.items_skipped = 0

        return self._get_state()

    def _generate_daily_items(self):
        """Generate or use fixed items for daily optimization"""
        if self.fixed_items is not None:
            return self.fixed_items

        cities = list(self.city_distances.keys())[1:]  # Exclude Jakarta (origin)
        items = []

        np.random.seed(42)  # ✅ Set seed tetap jika tidak fixed

        for i in range(50):
            weight = np.random.uniform(200, 800)
            volume = weight * np.random.uniform(0.003, 0.006)
            dimension_cat = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
            destination = np.random.choice(cities)

            items.append(Item(
                id=i,
                name=f"Barang_{i}",
                weight=weight,
                destination=destination,
                dimension_category=dimension_cat,
                volume=volume
            ))

        return items

    def _get_state(self):
        """Get improved state representation"""
        if self.current_item_idx >= len(self.items):
            return np.zeros(36, dtype=np.float32)  # ✅ sesuai state_size

        current_item = self.items[self.current_item_idx]

        state = []

        # Current item features (normalized)
        state.extend([
            current_item.weight / 1000.0,  # 0-1 range
            current_item.volume / 10.0,  # 0-1 range
            current_item.dimension_category / 2.0,  # 0-1 range
            self.city_distances[current_item.destination] / 1000.0,  # 0-1 range
        ])

        # Global statistics
        progress = self.current_item_idx / len(self.items)
        assignment_rate = self.items_assigned / max(1, self.items_processed)

        state.extend([
            progress,
            assignment_rate,
            self.total_profit * self.profit_scale,  # Normalized profit so far
            len([t for t in self.trucks if t.current_load > 0]) / 4.0  # Active trucks ratio
        ])

        # Enhanced truck status for each truck
        for truck in self.trucks:
            load_ratio = truck.current_load / truck.max_capacity
            volume_ratio = truck.current_volume / truck.box_volume

            # Check if item can fit
            can_fit_weight = current_item.weight <= (truck.max_capacity - truck.current_load)
            can_fit_volume = current_item.volume <= (truck.box_volume - truck.current_volume)
            can_fit = can_fit_weight and can_fit_volume

            # Calculate compatibility score
            compatibility = self._calculate_compatibility(current_item, truck)

            # Estimated profit normalized
            estimated_profit = self._calculate_item_profit(current_item, truck) * self.profit_scale

            state.extend([
                load_ratio,
                volume_ratio,
                1.0 - load_ratio,  # Remaining capacity ratio
                1.0 - volume_ratio,  # Remaining volume ratio
                float(can_fit),
                compatibility,
                estimated_profit
            ])

        return np.array(state, dtype=np.float32)

    def _calculate_compatibility(self, item, truck):
        """Calculate how compatible an item is with truck's current load"""
        if not truck.assigned_items:
            return 1.0

        # Check destination compatibility
        destinations = [existing_item.destination for existing_item in truck.assigned_items]
        if item.destination in destinations:
            return 1.0  # Same destination = high compatibility

        # Check distance compatibility (similar distances)
        current_distances = [self.city_distances[dest] for dest in destinations]
        item_distance = self.city_distances[item.destination]
        avg_distance = np.mean(current_distances)

        distance_compatibility = 1.0 - abs(item_distance - avg_distance) / 1000.0
        return max(0.0, distance_compatibility)

    def _calculate_item_profit(self, item, truck):
        """Calculate profit for assigning item to truck"""
        # Revenue calculation
        distance = self.city_distances[item.destination]
        dimension_multiplier = self.dimension_multipliers[item.dimension_category]
        revenue = item.weight * distance * self.base_price_per_kg_km * dimension_multiplier

        # Improved cost calculation
        base_fuel_cost = distance * truck.fuel_ratio * self.fuel_cost_per_liter

        if not truck.assigned_items:
            # First item - full cost
            fuel_cost = base_fuel_cost
        else:
            # Check if we're already going to this destination
            existing_destinations = [existing_item.destination for existing_item in truck.assigned_items]
            if item.destination in existing_destinations:
                # Same destination - minimal additional cost
                fuel_cost = base_fuel_cost * 0.05
            else:
                # Different destination - partial additional cost
                fuel_cost = base_fuel_cost * 0.3

        return revenue - fuel_cost

    def step(self, action):
        """Execute action with improved reward structure"""
        if self.done or self.current_item_idx >= len(self.items):
            return self._get_state(), 0, True, {}

        current_item = self.items[self.current_item_idx]
        reward = 0
        info = {}

        self.items_processed += 1

        if action < 4:  # Assign to truck 0-3
            truck = self.trucks[action]

            # Check if item can fit
            can_fit_weight = current_item.weight <= (truck.max_capacity - truck.current_load)
            can_fit_volume = current_item.volume <= (truck.box_volume - truck.current_volume)

            if can_fit_weight and can_fit_volume:
                # Valid assignment
                truck.current_load += current_item.weight
                truck.current_volume += current_item.volume
                truck.assigned_items.append(current_item)

                # Calculate profit and reward
                item_profit = self._calculate_item_profit(current_item, truck)
                self.total_profit += item_profit

                # Multi-component reward
                base_reward = item_profit * self.profit_scale

                # Bonus for efficiency
                efficiency_bonus = 0
                if truck.current_load / truck.max_capacity > 0.8:  # High utilization
                    efficiency_bonus += 0.2
                if truck.current_volume / truck.box_volume > 0.8:  # High volume utilization
                    efficiency_bonus += 0.2

                # Bonus for destination consolidation
                consolidation_bonus = 0
                destinations = [item.destination for item in truck.assigned_items]
                if len(set(destinations)) < len(destinations):  # Same destination exists
                    consolidation_bonus = 0.3

                reward = base_reward + efficiency_bonus + consolidation_bonus
                self.items_assigned += 1

                info['assignment'] = 'valid'
                info['profit'] = item_profit

            else:
                # Invalid assignment - penalty
                reward = -1.0
                info['assignment'] = 'invalid'
                if not can_fit_weight:
                    info['reason'] = 'weight_exceeded'
                else:
                    info['reason'] = 'volume_exceeded'

        else:  # action == 4: Skip item
            # Dynamic skip penalty based on item value
            item_value = self._calculate_item_profit(current_item, self.trucks[0]) * self.profit_scale
            skip_penalty = -min(0.5, item_value * 0.1)  # Penalty proportional to lost value
            reward = skip_penalty
            self.items_skipped += 1
            info['assignment'] = 'skipped'

        # Move to next item
        self.current_item_idx += 1

        # Check if episode is done
        if self.current_item_idx >= len(self.items):
            self.done = True

            # Final efficiency bonus
            total_capacity_used = sum(truck.current_load for truck in self.trucks)
            total_capacity = sum(truck.max_capacity for truck in self.trucks)
            capacity_utilization = total_capacity_used / total_capacity

            efficiency_bonus = capacity_utilization * 2.0  # Reward for high utilization
            reward += efficiency_bonus

            info['final_efficiency_bonus'] = efficiency_bonus

        next_state = self._get_state()

        info.update({
            'total_profit': self.total_profit,
            'items_assigned': self.items_assigned,
            'items_skipped': self.items_skipped,
            'assignment_rate': self.items_assigned / max(1, self.items_processed)
        })

        return next_state, reward, self.done, info


# Improved Deep Q-Network
class ImprovedDQN(nn.Module):
    def __init__(self, state_size=36, action_size=5, hidden_size=256):
        super(ImprovedDQN, self).__init__()

        # Larger network with batch normalization
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)

        self.dropout = nn.Dropout(0.3)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Handle single samples for batch normalization
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)

        if single_sample:
            x = x.squeeze(0)

        return x


# Experience replay buffer (unchanged)
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    def __init__(self, capacity=50000):  # Increased capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# Improved DQN Agent
class ImprovedDQNAgent:
    def __init__(self, state_size=36, action_size=5, lr=0.0003, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.05):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Neural networks
        self.q_network = ImprovedDQN(state_size, action_size)
        self.target_network = ImprovedDQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-5)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.8)

        # Experience replay
        self.memory = ReplayBuffer()
        self.batch_size = 128  # Increased batch size
        self.update_target_freq = 50  # More frequent updates
        self.learn_step = 0

        # Copy weights to target network
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def act(self, state):
        """Choose action using epsilon-greedy policy with improved exploration"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        self.q_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            q_values = self.q_network(state_tensor)
        self.q_network.train()

        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from memory
        experiences = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences]))
        dones = torch.BoolTensor([e.done for e in experiences])

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Double DQN: use main network to select action, target network to evaluate
        next_actions = self.q_network(next_states).max(1)[1].detach()
        next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss with gradient clipping
        loss = F.huber_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network
        self.learn_step += 1
        if self.learn_step % self.update_target_freq == 0:
            self.update_target_network()

        # Decay epsilon more gradually
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update learning rate
        if self.learn_step % 100 == 0:
            self.scheduler.step()

    def save(self, path="best_agent.pth"):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    @classmethod
    def load(cls, path="best_agent.pth"):
        checkpoint = torch.load(path)
        agent = cls()
        agent.q_network.load_state_dict(checkpoint['q_network'])
        agent.target_network.load_state_dict(checkpoint['target_network'])
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        agent.epsilon = checkpoint.get('epsilon', 0.1)
        return agent


# Improved training function
def train_improved_dqn(episodes=200, load_best=False, fixed_items=None):
    env = OptimizedTruckLoadingEnvironment(fixed_items=fixed_items)
    agent = ImprovedDQNAgent()
    # ✅ Jika load_best, muat model terbaik dari file
    if load_best:
        print("[INFO] Loading agent from best_agent.pth for continued training...")
        state = torch.load("best_agent.pth")
        agent.q_network.load_state_dict(state['q_network'])
        agent.target_network.load_state_dict(state['target_network'])
        agent.optimizer.load_state_dict(state['optimizer'])
        agent.epsilon = max(state.get('epsilon', 0.1), 0.1)  # jangan mulai dari 0

    scores = []
    profits = []
    assignment_rates = []

    # Tracking best performance
    best_profit = 0
    best_agent_state = None

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_info = {}

        while not env.done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)

            # Learn more frequently in early episodes
            if episode < 500 or len(agent.memory) % 4 == 0:
                agent.learn()

            state = next_state
            total_reward += reward
            episode_info = info

        scores.append(total_reward)
        profits.append(episode_info.get('total_profit', 0))
        assignment_rates.append(episode_info.get('assignment_rate', 0))

        # Track best performance
        if episode_info.get('total_profit', 0) > best_profit:
            best_profit = episode_info.get('total_profit', 0)
            best_agent_state = {
                'q_network': agent.q_network.state_dict(),
                'target_network': agent.target_network.state_dict(),
                'optimizer': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon
            }

        if episode % 1 == 0:
            avg_score = np.mean(scores[-100:])
            avg_profit = np.mean(profits[-100:])
            avg_assignment_rate = np.mean(assignment_rates[-100:])
            current_lr = agent.optimizer.param_groups[0]['lr']

            print(f"Episode {episode:4d} | "
                  f"Avg Score: {avg_score:8.2f} | "
                  f"Avg Profit: {avg_profit:12.0f} | "
                  f"Assignment Rate: {avg_assignment_rate:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"LR: {current_lr:.6f}")

    # Load best model
    if best_agent_state:
        agent.q_network.load_state_dict(best_agent_state['q_network'])
        agent.target_network.load_state_dict(best_agent_state['target_network'])
        agent.epsilon = 0  # Set to 0 for evaluation
        agent.save("best_agent.pth")
        print("Best agent saved to 'best_agent.pth'")
        print(f"\nLoaded best model with profit: IDR {best_profit:,.0f}")

    return agent, scores, profits, assignment_rates


# Enhanced evaluation function
def evaluate_improved_agent(agent, episodes=10, fixed_items=None):
    env = OptimizedTruckLoadingEnvironment(fixed_items=fixed_items)
    agent.epsilon = 0  # No exploration during evaluation

    total_profits = []
    assignment_rates = []
    utilization_rates = []

    for episode in range(episodes):
        state = env.reset()
        episode_actions = []

        while not env.done:
            action = agent.act(state)
            episode_actions.append(action)
            state, _, done, info = env.step(action)

        total_profits.append(info['total_profit'])
        assignment_rates.append(info['assignment_rate'])

        # Calculate truck utilization
        total_capacity_used = sum(truck.current_load for truck in env.trucks)
        total_capacity = sum(truck.max_capacity for truck in env.trucks)
        utilization_rates.append(total_capacity_used / total_capacity)

        # Print detailed results for first episode
        if episode >= 0:
            print(f"\n=== Detailed Results for Episode 1 ===")
            print(f"Total Profit: IDR {info['total_profit']:,.0f}")
            print(f"Items Assigned: {info['items_assigned']}/{len(env.items)} ({info['assignment_rate']:.1%})")
            print(f"Items Skipped: {info['items_skipped']}")

            active_trucks = 0
            for i, truck in enumerate(env.trucks):
                if truck.assigned_items:
                    active_trucks += 1
                    print(f"\nTruck {i} ({truck.plate_number}):")
                    print(
                        f"  Load: {truck.current_load:.1f}/{truck.max_capacity} kg ({truck.current_load / truck.max_capacity:.1%})")
                    print(
                        f"  Volume: {truck.current_volume:.1f}/{truck.box_volume} m³ ({truck.current_volume / truck.box_volume:.1%})")
                    print(f"  Items assigned: {len(truck.assigned_items)}")

                    # Group by destination
                    dest_groups = {}
                    for item in truck.assigned_items:
                        if item.destination not in dest_groups:
                            dest_groups[item.destination] = []
                        dest_groups[item.destination].append(item)

                    for dest, items in dest_groups.items():
                        total_weight = sum(item.weight for item in items)
                        total_profit = sum(env._calculate_item_profit(item, truck) for item in items)
                        print(f"    {dest}: {len(items)} items, {total_weight:.1f}kg, IDR {total_profit:,.0f}")

            print(f"\nActive Trucks: {active_trucks}/4")
            print(f"Overall Capacity Utilization: {total_capacity_used / total_capacity:.1%}")

    avg_profit = np.mean(total_profits)
    avg_assignment_rate = np.mean(assignment_rates)
    avg_utilization = np.mean(utilization_rates)

    print(f"\n=== Evaluation Summary ===")
    print(f"Average Profit over {episodes} episodes: IDR {avg_profit:,.0f}")
    print(f"Average Assignment Rate: {avg_assignment_rate:.1%}")
    print(f"Average Capacity Utilization: {avg_utilization:.1%}")
    print(f"Best Profit: IDR {max(total_profits):,.0f}")
    print(f"Worst Profit: IDR {min(total_profits):,.0f}")
    print(f"Profit Std Dev: IDR {np.std(total_profits):,.0f}")

    return total_profits, assignment_rates, utilization_rates
def load_and_run_best_agent(episodes=10):
    print("\nLoading best agent from 'best_agent.pth'...")

    state = torch.load("best_agent.pth")
    agent = ImprovedDQNAgent()
    agent.q_network.load_state_dict(state['q_network'])
    agent.target_network.load_state_dict(state['target_network'])
    agent.optimizer.load_state_dict(state['optimizer'])
    agent.epsilon = 0  # Set ke evaluasi

    print("Evaluating best agent...\n")
    return evaluate_improved_agent(agent, episodes=episodes)



# Main execution
if __name__ == "__main__":
    print("Starting Improved DQN Training for Truck Loading Optimization...")
    print("Improvements:")
    print("- Better reward structure with multi-component rewards")
    print("- Improved state representation with compatibility scores")
    print("- Larger network with batch normalization")
    print("- Better exploration strategy")
    print("- Double DQN with gradient clipping")
    print("- Learning rate scheduling")
    print("- Best model tracking\n")
    import sys

    # Buat fixed item list
    fixed_env = OptimizedTruckLoadingEnvironment()
    fixed_items = fixed_env._generate_daily_items()

    if "--eval" in sys.argv:
        agent = ImprovedDQNAgent.load("best_agent.pth")
        evaluate_improved_agent(agent, episodes=10, fixed_items=fixed_items)
        exit()

    # Train the agent
    trained_agent, training_scores, training_profits, training_assignment_rates = train_improved_dqn(episodes=200, load_best=True, fixed_items=fixed_items)
    print("\nTraining completed! Evaluating trained agent...")

    # Evaluate the trained agent
    evaluation_profits, evaluation_assignment_rates, evaluation_utilization = evaluate_improved_agent(trained_agent, episodes=10, fixed_items=fixed_items)

    # Plot training progress
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(training_scores)
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(training_profits)
    plt.title('Training Profits')
    plt.xlabel('Episode')
    plt.ylabel('Profit (IDR)')
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(training_assignment_rates)
    plt.title('Assignment Rates')
    plt.xlabel('Episode')
    plt.ylabel('Assignment Rate')
    plt.grid(True)

    plt.subplot(2, 3, 4)
    # Moving averages
    window = 50
    scores_ma = np.convolve(training_scores, np.ones(window) / window, mode='valid')
    profits_ma = np.convolve(training_profits, np.ones(window) / window, mode='valid')

    plt.plot(scores_ma, label='Scores MA')
    plt.title('Moving Averages (50 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(profits_ma)
    plt.title('Profit Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Profit (IDR)')
    plt.grid(True)

    plt.subplot(2, 3, 6)
    # Evaluation results
    plt.bar(['Profits', 'Assignment Rate', 'Utilization'],
            [np.mean(evaluation_profits) / 1e9, np.mean(evaluation_assignment_rates) * 100,
             np.mean(evaluation_utilization) * 100],
            color=['blue', 'green', 'orange'])
    plt.title('Evaluation Results')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.show()

    print(f"\nFinal Results:")
    print(f"Final Average Training Profit (last 100): IDR {np.mean(training_profits[-100:]):,.0f}")
    print(f"Best Training Profit: IDR {max(training_profits):,.0f}")
    print(f"Evaluation Average Profit: IDR {np.mean(evaluation_profits):,.0f}")
    print(f"Improvement over original: {((np.mean(evaluation_profits) / 1933245655) - 1) * 100:.1f}%")

    # Jalankan ulang best agent dari file
    load_and_run_best_agent(episodes=10)


