from ai_qlearning import QLearningAgent
from dino_env import DinoGame
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


def analyze_model(filename="dino_ai_enhanced.pkl"):
    """Analisis mendalam terhadap model yang tersimpan"""
    if not os.path.exists(filename):
        print(f"❌ Model file '{filename}' not found!")
        return

    # Load model
    agent = QLearningAgent(actions=["STAY", "JUMP", "DUCK"])
    if not agent.load_model(filename):
        return

    print(f"\n🔍 DETAILED MODEL ANALYSIS")
    print("=" * 50)

    # Q-table analysis
    if agent.q_table:
        q_values = []
        state_actions = []

        for state, q_vals in agent.q_table.items():
            for action_idx, q_val in enumerate(q_vals):
                q_values.append(q_val)
                state_actions.append((state, action_idx, q_val))

        print(f"Q-table Statistics:")
        print(f"  Total states: {len(agent.q_table)}")
        print(f"  Total Q-values: {len(q_values)}")
        print(f"  Q-value range: {min(q_values):.3f} to {max(q_values):.3f}")
        print(f"  Average Q-value: {np.mean(q_values):.3f}")
        print(f"  Q-value std: {np.std(q_values):.3f}")

        # Top 10 best Q-values
        state_actions.sort(key=lambda x: x[2], reverse=True)
        print(f"\n🏆 Top 10 Best State-Actions:")
        action_names = ["STAY", "JUMP", "DUCK"]
        for i, (state, action, q_val) in enumerate(state_actions[:10]):
            print(f"  {i + 1}. State {state} → {action_names[action]} (Q={q_val:.3f})")

        # State distribution analysis
        state_types = {}
        for state in agent.q_table.keys():
            distance, obs_type, dino_state = state
            key = f"obs_{obs_type}_dino_{dino_state}"
            if key not in state_types:
                state_types[key] = 0
            state_types[key] += 1

        print(f"\n📊 State Distribution:")
        for state_type, count in sorted(state_types.items()):
            percentage = (count / len(agent.q_table)) * 100
            print(f"  {state_type}: {count} states ({percentage:.1f}%)")

    # Training history analysis
    history = agent.training_history
    print(f"\n📈 Training History:")
    print(f"  Total training runs: {history['total_runs']}")
    print(f"  Total episodes: {history['episodes_trained']}")
    print(f"  Best score ever: {history['best_score']}")

    if history['training_sessions']:
        print(f"\n📋 Training Sessions:")
        for i, session in enumerate(history['training_sessions']):
            print(f"  Session {i + 1}:")
            print(f"    Date: {session.get('start_time', 'N/A')}")
            print(f"    Episodes: {session.get('episodes', 'N/A')}")
            print(f"    Best reward: {session.get('best_reward', 'N/A'):.1f}")
            print(f"    Final avg: {session.get('final_100_avg', 'N/A'):.1f}")

    # Policy visualization
    visualize_policy(agent)


def visualize_policy(agent):
    """Visualisasi policy yang dipelajari AI"""
    if not agent.q_table:
        print("No Q-table to visualize")
        return

    # Create policy matrix
    distances = list(range(11))  # 0-10
    obs_types = [0, 1]  # cactus, bird
    dino_states = [0, 1, 2]  # ground, jump, duck
    action_names = ["STAY", "JUMP", "DUCK"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('AI Policy Visualization', fontsize=16)

    for obs_idx, obs_type in enumerate(obs_types):
        for dino_idx, dino_state in enumerate(dino_states):
            ax = axes[obs_idx, dino_idx]

            # Get best actions for each distance
            best_actions = []
            q_values_for_plot = []

            for dist in distances:
                state = (dist, obs_type, dino_state)
                if state in agent.q_table:
                    q_vals = agent.q_table[state]
                    best_action = np.argmax(q_vals)
                    best_q = max(q_vals)
                else:
                    best_action = 0  # Default STAY
                    best_q = 0

                best_actions.append(best_action)
                q_values_for_plot.append(best_q)

            # Plot
            colors = ['red', 'green', 'blue']  # STAY, JUMP, DUCK
            bar_colors = [colors[action] for action in best_actions]

            bars = ax.bar(distances, q_values_for_plot, color=bar_colors, alpha=0.7)

            # Labels
            obs_name = "Cactus" if obs_type == 0 else "Bird"
            dino_name = ["Ground", "Jumping", "Ducking"][dino_state]
            ax.set_title(f'{obs_name} - Dino {dino_name}')
            ax.set_xlabel('Distance to Obstacle')
            ax.set_ylabel('Q-Value')
            ax.set_xticks(distances)

            # Add action labels on bars
            for bar, action in zip(bars, best_actions):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                            action_names[action][0], ha='center', va='bottom', fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='STAY'),
                       Patch(facecolor='green', alpha=0.7, label='JUMP'),
                       Patch(facecolor='blue', alpha=0.7, label='DUCK')]
    fig.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig('ai_policy_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()


def test_model_performance(filename="dino_ai_enhanced.pkl", episodes=10):
    """Test performa model tanpa training"""
    print(f"\n🎮 TESTING MODEL PERFORMANCE")
    print("=" * 50)

    # Load model
    agent = QLearningAgent(actions=["STAY", "JUMP", "DUCK"])
    if not agent.load_model(filename):
        return

    env = DinoGame()

    # Set epsilon to 0 untuk pure exploitation
    old_epsilon = agent.epsilon
    agent.epsilon = 0

    test_rewards = []
    test_scores = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 2000:  # Longer episodes for testing
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1

            # Render setiap beberapa episode
            if episode < 3:  # Show first 3 episodes
                env.render()
                import time
                time.sleep(0.02)

        test_rewards.append(total_reward)
        test_scores.append(env.score)
        print(f"Test {episode + 1:2d}: Reward={total_reward:6.1f}, Score={env.score:3d}, Steps={steps:4d}")

    # Restore epsilon
    agent.epsilon = old_epsilon

    print(f"\n📊 Test Results Summary:")
    print(f"Episodes tested: {episodes}")
    print(f"Average reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"Best reward: {max(test_rewards):.1f}")
    print(f"Worst reward: {min(test_rewards):.1f}")
    print(f"Average score: {np.mean(test_scores):.1f} ± {np.std(test_scores):.1f}")
    print(f"Best score: {max(test_scores)}")

    env.close()
    return test_rewards, test_scores


def backup_model(filename="dino_ai_enhanced.pkl"):
    """Backup model dengan timestamp"""
    if not os.path.exists(filename):
        print(f"❌ Model file '{filename}' not found!")
        return

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"dino_ai_backup_{timestamp}.pkl"

    import shutil
    shutil.copy2(filename, backup_name)
    print(f"✅ Model backed up as '{backup_name}'")


def main():
    print("🛠️  DINO AI - Model Manager")
    print("=" * 50)
    print("1. Analyze current model")
    print("2. Test model performance")
    print("3. Backup current model")
    print("4. View policy visualization")
    print("5. All of the above")

    choice = input("\nChoose option (1-5): ").strip()

    if choice in ['1', '5']:
        analyze_model()

    if choice in ['2', '5']:
        episodes = int(input("\nEnter number of test episodes (default 10): ") or "10")
        test_model_performance(episodes=episodes)

    if choice in ['3', '5']:
        backup_model()

    if choice in ['4', '5']:
        agent = QLearningAgent(actions=["STAY", "JUMP", "DUCK"])
        if agent.load_model():
            visualize_policy(agent)


if __name__ == "__main__":
    main()