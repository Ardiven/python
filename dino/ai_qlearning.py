import random
import numpy as np
import pickle
import os
import json
from datetime import datetime
from collections import defaultdict


class QLearningAgent:
    def __init__(self, actions, learning_rate=0.15, discount=0.99, epsilon=0.9, epsilon_decay=0.9995):
        # Menggunakan defaultdict untuk Q-table yang lebih efisien
        self.q_table = defaultdict(lambda: [0.0 for _ in actions])
        self.actions = actions
        self.alpha = learning_rate  # Sedikit lebih tinggi untuk learning lebih cepat
        self.gamma = discount  # Lebih tinggi untuk mempertimbangkan reward jangka panjang
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay  # Decay lebih lambat
        self.initial_epsilon = epsilon
        self.min_epsilon = 0.05  # Minimum epsilon lebih tinggi

        # Experience replay buffer
        self.experience_buffer = []
        self.buffer_size = 10000
        self.batch_size = 32

        # Action frequency tracking untuk balanced exploration
        self.action_counts = defaultdict(int)
        self.total_actions = 0

        # Training history
        self.training_history = {
            'episodes_trained': 0,
            'total_runs': 0,
            'best_score': 0,
            'training_sessions': []
        }

    def add_experience(self, state, action, reward, next_state, done):
        """Tambahkan experience ke buffer untuk replay learning"""
        experience = (state, action, reward, next_state, done)

        if len(self.experience_buffer) >= self.buffer_size:
            self.experience_buffer.pop(0)

        self.experience_buffer.append(experience)

    def replay_experience(self):
        """Experience replay untuk stabilitas learning"""
        if len(self.experience_buffer) < self.batch_size:
            return

        # Sample random batch
        batch = random.sample(self.experience_buffer, self.batch_size)

        for state, action, reward, next_state, done in batch:
            if done:
                target = reward
            else:
                target = reward + self.gamma * max(self.get_qs(next_state))

            current_q = self.get_qs(state)[action]
            # Update Q-value dengan learning rate yang dinamis
            dynamic_alpha = self.alpha * (1.0 + abs(reward) / 100.0)  # Higher learning rate for significant rewards
            self.q_table[state][action] = current_q + dynamic_alpha * (target - current_q)

    def get_qs(self, state):
        return self.q_table[state]

    def choose_action(self, state):
        self.total_actions += 1

        # Epsilon-greedy dengan action balancing
        if random.random() < self.epsilon:
            # Exploration dengan bias ke action yang jarang digunakan
            if self.total_actions > 100:  # Setelah beberapa action
                action_probs = []
                for i in range(len(self.actions)):
                    freq = self.action_counts[i] / self.total_actions
                    # Inverse frequency untuk encourage underused actions
                    prob = 1.0 - freq + 0.1  # Minimal 0.1 probability
                    action_probs.append(prob)

                # Normalize probabilities
                total_prob = sum(action_probs)
                action_probs = [p / total_prob for p in action_probs]

                # Weighted random choice
                action = np.random.choice(len(self.actions), p=action_probs)
            else:
                action = random.choice(range(len(self.actions)))
        else:
            # Exploitation
            qs = self.get_qs(state)
            action = int(np.argmax(qs))

        self.action_counts[action] += 1
        return action

    def update_q(self, state, action, reward, next_state, done=False):
        """Update Q-value dengan metode yang lebih sophisticated"""
        # Store experience for replay
        self.add_experience(state, action, reward, next_state, done)

        # Immediate Q-learning update
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.get_qs(next_state))

        current_q = self.get_qs(state)[action]

        # Dynamic learning rate berdasarkan magnitude reward
        dynamic_alpha = self.alpha
        if abs(reward) > 10:  # Significant reward/penalty
            dynamic_alpha *= 1.5
        elif abs(reward) < 1:  # Small reward
            dynamic_alpha *= 0.8

        # TD Error untuk priority
        td_error = target - current_q

        # Update Q-value
        self.q_table[state][action] = current_q + dynamic_alpha * td_error

        # Periodic experience replay
        if len(self.experience_buffer) >= self.batch_size and random.random() < 0.1:
            self.replay_experience()

    def decay_epsilon(self):
        """Epsilon decay yang lebih smooth"""
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.min_epsilon, self.epsilon)

    def adaptive_learning_rate(self, episode):
        """Adaptive learning rate berdasarkan progress"""
        if episode < 100:
            self.alpha = 0.15  # High learning rate awal
        elif episode < 500:
            self.alpha = 0.12  # Moderate
        else:
            self.alpha = 0.08  # Lower untuk fine-tuning

    def get_state_value(self, state):
        """Estimasi nilai state"""
        return max(self.get_qs(state))

    def get_policy_quality(self):
        """Evaluasi kualitas policy"""
        if not self.q_table:
            return 0

        # Average Q-value sebagai indikator kualitas
        all_q_values = []
        for state_values in self.q_table.values():
            all_q_values.extend(state_values)

        return np.mean(all_q_values) if all_q_values else 0

    def save_model(self, filename="dino_ai_enhanced.pkl"):
        """Simpan model dengan informasi tambahan"""
        # Convert defaultdict to regular dict for pickling
        regular_q_table = dict(self.q_table)

        model_data = {
            'q_table': regular_q_table,
            'epsilon': self.epsilon,
            'training_history': self.training_history,
            'hyperparameters': {
                'learning_rate': self.alpha,
                'discount': self.gamma,
                'epsilon_decay': self.epsilon_decay,
                'initial_epsilon': self.initial_epsilon,
                'min_epsilon': self.min_epsilon
            },
            'action_stats': dict(self.action_counts),
            'total_actions': self.total_actions,
            'policy_quality': self.get_policy_quality()
        }

        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✅ Model saved to {filename}")
        print(f"   Q-table size: {len(self.q_table)} states")
        print(f"   Current epsilon: {self.epsilon:.4f}")
        print(f"   Policy quality: {self.get_policy_quality():.3f}")
        print(f"   Total episodes trained: {self.training_history['episodes_trained']}")

    def load_model(self, filename="dino_ai_enhanced.pkl"):
        """Load model dengan backward compatibility"""
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    model_data = pickle.load(f)

                # Load Q-table and convert back to defaultdict
                loaded_q_table = model_data['q_table']
                self.q_table = defaultdict(lambda: [0.0 for _ in self.actions])
                self.q_table.update(loaded_q_table)

                self.epsilon = model_data['epsilon']
                self.training_history = model_data.get('training_history', self.training_history)

                # Load hyperparameters
                if 'hyperparameters' in model_data:
                    hyper = model_data['hyperparameters']
                    self.alpha = hyper.get('learning_rate', self.alpha)
                    self.gamma = hyper.get('discount', self.gamma)
                    self.epsilon_decay = hyper.get('epsilon_decay', self.epsilon_decay)
                    self.initial_epsilon = hyper.get('initial_epsilon', self.initial_epsilon)
                    self.min_epsilon = hyper.get('min_epsilon', self.min_epsilon)

                # Load action stats
                if 'action_stats' in model_data:
                    self.action_counts = defaultdict(int)
                    self.action_counts.update(model_data['action_stats'])
                    self.total_actions = model_data.get('total_actions', 0)

                print(f"✅ Model loaded from {filename}")
                print(f"   Q-table size: {len(self.q_table)} states")
                print(f"   Current epsilon: {self.epsilon:.4f}")
                print(f"   Policy quality: {model_data.get('policy_quality', 0):.3f}")
                print(f"   Total episodes trained: {self.training_history['episodes_trained']}")
                print(f"   Total training runs: {self.training_history['total_runs']}")
                print(f"   Best score achieved: {self.training_history['best_score']}")

                return True
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                print("Starting with fresh model...")
                return False
        else:
            print(f"📁 No existing model found at {filename}")
            print("Starting with fresh model...")
            return False

    def start_new_training_session(self, episodes):
        """Mulai session training baru"""
        session_info = {
            'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'episodes': episodes,
            'start_epsilon': self.epsilon,
            'start_q_table_size': len(self.q_table),
            'start_policy_quality': self.get_policy_quality()
        }
        self.training_history['training_sessions'].append(session_info)
        self.training_history['total_runs'] += 1

    def end_training_session(self, all_rewards):
        """Akhiri session training dan update statistics"""
        if self.training_history['training_sessions']:
            current_session = self.training_history['training_sessions'][-1]
            current_session['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_session['end_epsilon'] = self.epsilon
            current_session['end_q_table_size'] = len(self.q_table)
            current_session['end_policy_quality'] = self.get_policy_quality()
            current_session['average_reward'] = sum(all_rewards) / len(all_rewards) if all_rewards else 0
            current_session['best_reward'] = max(all_rewards) if all_rewards else 0
            current_session['final_100_avg'] = sum(all_rewards[-100:]) / min(100,
                                                                             len(all_rewards)) if all_rewards else 0

            # Update global best score - convert rewards to approximate scores
            max_score = max([reward / 15 for reward in all_rewards]) if all_rewards else 0  # Approximate conversion
            if max_score > self.training_history['best_score']:
                self.training_history['best_score'] = max_score

            self.training_history['episodes_trained'] += len(all_rewards)

    def print_training_summary(self):
        """Print ringkasan training dengan info tambahan"""
        print(f"\n{'=' * 50}")
        print(f"🤖 AI TRAINING SUMMARY")
        print(f"{'=' * 50}")
        print(f"Total Training Runs: {self.training_history['total_runs']}")
        print(f"Total Episodes Trained: {self.training_history['episodes_trained']}")
        print(f"Best Score Ever: {self.training_history['best_score']:.1f}")
        print(f"Current Q-table Size: {len(self.q_table)} states")
        print(f"Current Epsilon: {self.epsilon:.4f}")
        print(f"Policy Quality: {self.get_policy_quality():.3f}")

        # Action distribution
        if self.total_actions > 0:
            print(f"\n🎯 ACTION DISTRIBUTION:")
            action_names = ["STAY", "JUMP", "DUCK"]
            for i, action_name in enumerate(action_names):
                count = self.action_counts[i]
                percentage = (count / self.total_actions) * 100
                print(f"  {action_name}: {percentage:.1f}% ({count} times)")

        print(f"\n📊 RECENT TRAINING SESSIONS:")
        for i, session in enumerate(self.training_history['training_sessions'][-3:], 1):  # Show last 3 sessions
            session_num = len(self.training_history['training_sessions']) - 3 + i
            print(f"  Session {session_num}:")
            print(f"    Episodes: {session.get('episodes', 'N/A')}")
            print(f"    Date: {session.get('start_time', 'N/A')}")
            print(f"    Best Reward: {session.get('best_reward', 'N/A'):.1f}")
            print(f"    Avg Last 100: {session.get('final_100_avg', 'N/A'):.1f}")
            print(f"    Epsilon: {session.get('start_epsilon', 'N/A'):.3f} → {session.get('end_epsilon', 'N/A'):.3f}")
            print(
                f"    Policy Quality: {session.get('start_policy_quality', 0):.3f} → {session.get('end_policy_quality', 0):.3f}")

    def reset_epsilon(self, new_epsilon=None):
        """Reset epsilon dengan nilai yang lebih optimal"""
        if new_epsilon is None:
            # Reset ke nilai yang memungkinkan exploration yang baik
            self.epsilon = max(0.4, self.initial_epsilon * 0.5)
        else:
            self.epsilon = new_epsilon
        print(f"🔄 Epsilon reset to {self.epsilon:.3f}")

    def get_policy_stats(self):
        """Analisis policy yang dipelajari dengan lebih detail"""
        if not self.q_table:
            return "No policy learned yet"

        action_counts = {action: 0 for action in range(len(self.actions))}
        state_count = 0

        for state, q_values in self.q_table.items():
            best_action = np.argmax(q_values)
            action_counts[best_action] += 1
            state_count += 1

        policy_summary = []
        action_names = ["STAY", "JUMP", "DUCK"]

        for action_id, count in action_counts.items():
            percentage = (count / state_count) * 100 if state_count > 0 else 0
            policy_summary.append(f"{action_names[action_id]}: {percentage:.1f}%")

        confidence_score = max(action_counts.values()) / state_count if state_count > 0 else 0

        return f"Policy: {', '.join(policy_summary)} | Confidence: {confidence_score:.2f}"