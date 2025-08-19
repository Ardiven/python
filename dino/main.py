from dino_env import DinoGame  # Ganti dengan nama file optimized environment
from ai_qlearning import QLearningAgent  # Ganti dengan nama file optimized agent
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from datetime import datetime


def enhanced_training_with_curriculum():
    """Training dengan curriculum learning untuk hasil yang lebih baik"""
    print("🦕 DINO AI - ENHANCED CURRICULUM TRAINING")
    print("=" * 60)

    # Configuration
    PHASES = [
        {"episodes": 200, "description": "Phase 1: Basic Survival", "max_speed": 6},
        {"episodes": 300, "description": "Phase 2: Improved Reactions", "max_speed": 8},
        {"episodes": 400, "description": "Phase 3: Advanced Gameplay", "max_speed": 10},
        {"episodes": 300, "description": "Phase 4: Mastery", "max_speed": 12}
    ]

    CHECKPOINT_INTERVAL = 50
    TEST_EPISODES = 10
    MODEL_FILE = "dino_ai_enhanced.pkl"
    RESULTS_FILE = "training_results.pkl"

    print(f"Training Phases:")
    for i, phase in enumerate(PHASES, 1):
        print(f"  Phase {i}: {phase['description']} - {phase['episodes']} episodes")
    print(f"Model File: {MODEL_FILE}")

    # Initialize
    env = DinoGame()
    agent = QLearningAgent(actions=["STAY", "JUMP", "DUCK"])

    # Load existing model
    model_loaded = agent.load_model(MODEL_FILE)
    if model_loaded:
        print(f"\n🤖 Existing model loaded!")
        agent.print_training_summary()

        choice = input("\nContinue training? (y/n, default=y): ").strip().lower()
        if choice == 'n':
            return

    # Training with curriculum
    all_rewards = []
    all_scores = []
    phase_results = []

    total_episodes = sum(phase["episodes"] for phase in PHASES)
    episode_count = 0

    print(f"\n🚀 STARTING ENHANCED TRAINING...")
    print("=" * 60)

    start_time = time.time()

    for phase_idx, phase in enumerate(PHASES):
        print(f"\n🎯 {phase['description'].upper()}")
        print("=" * 50)

        phase_rewards = []
        phase_scores = []
        phase_start_time = time.time()

        # Adjust learning parameters for each phase
        if phase_idx == 0:
            agent.alpha = 0.2  # High learning rate for initial phase
            agent.epsilon = max(0.8, agent.epsilon)  # Ensure exploration
        elif phase_idx == 1:
            agent.alpha = 0.15
            agent.epsilon = max(0.6, agent.epsilon)
        elif phase_idx == 2:
            agent.alpha = 0.1
            agent.epsilon = max(0.4, agent.epsilon)
        else:  # Final phase
            agent.alpha = 0.05
            agent.epsilon = max(0.2, agent.epsilon)

        # Phase training loop
        for episode in range(phase["episodes"]):
            episode_count += 1

            # Reset environment with phase-specific settings
            state = env.reset()
            if hasattr(env, 'max_speed'):
                env.max_speed = phase["max_speed"]  # Curriculum: gradually increase difficulty

            total_reward = 0
            steps = 0
            done = False

            while not done:
                # Choose action
                action = agent.choose_action(state)

                # Take action
                next_state, reward, done = env.step(action)

                # Learn
                agent.learn(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                steps += 1

                # Prevent infinite loops
                if steps > 5000:
                    done = True
                    break

            # Record results
            phase_rewards.append(total_reward)
            # Get score from environment (adjust attribute name as needed)
            current_score = getattr(env, 'score', 0) or getattr(env, 'current_score', 0)
            phase_scores.append(current_score)
            all_rewards.append(total_reward)
            all_scores.append(current_score)

            # Progress reporting
            if (episode + 1) % 25 == 0:
                avg_reward = np.mean(phase_rewards[-25:])
                avg_score = np.mean(phase_scores[-25:])
                print(f"  Episode {episode + 1:3d}/{phase['episodes']:3d} | "
                      f"Avg Reward: {avg_reward:7.2f} | "
                      f"Avg Score: {avg_score:6.1f} | "
                      f"Epsilon: {agent.epsilon:.3f}")

            # Checkpoint saving
            if episode_count % CHECKPOINT_INTERVAL == 0:
                agent.save_model(MODEL_FILE)
                save_training_results(all_rewards, all_scores, phase_results, RESULTS_FILE)

        # Phase summary
        phase_end_time = time.time()
        phase_duration = phase_end_time - phase_start_time

        phase_avg_reward = np.mean(phase_rewards)
        phase_avg_score = np.mean(phase_scores)
        phase_max_score = max(phase_scores) if phase_scores else 0

        phase_result = {
            "phase": phase_idx + 1,
            "description": phase["description"],
            "episodes": phase["episodes"],
            "avg_reward": phase_avg_reward,
            "avg_score": phase_avg_score,
            "max_score": phase_max_score,
            "duration": phase_duration
        }
        phase_results.append(phase_result)

        print(f"\n📊 {phase['description']} COMPLETED:")
        print(f"  Average Reward: {phase_avg_reward:.2f}")
        print(f"  Average Score: {phase_avg_score:.1f}")
        print(f"  Max Score: {phase_max_score}")
        print(f"  Duration: {phase_duration / 60:.1f} minutes")

        # Test performance after each phase
        if phase_idx < len(PHASES) - 1:  # Don't test after final phase
            print(f"\n🧪 Testing performance...")
            test_results = test_agent_performance(env, agent, TEST_EPISODES)
            print(f"  Test Average Score: {test_results['avg_score']:.1f}")
            print(f"  Test Max Score: {test_results['max_score']}")

    # Final training summary
    total_time = time.time() - start_time

    print(f"\n🎉 TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Total Episodes: {total_episodes}")
    print(f"Total Duration: {total_time / 3600:.2f} hours")
    print(f"Final Epsilon: {agent.epsilon:.3f}")
    print(f"Q-Table Size: {len(agent.q_table)}")

    # Save final model and results
    agent.save_model(MODEL_FILE)
    save_training_results(all_rewards, all_scores, phase_results, RESULTS_FILE)

    # Final performance test
    print(f"\n🏆 FINAL PERFORMANCE TEST")
    print("=" * 40)
    final_test = test_agent_performance(env, agent, 20)

    print(f"Final Test Results:")
    print(f"  Average Score: {final_test['avg_score']:.1f}")
    print(f"  Max Score: {final_test['max_score']}")
    print(f"  Success Rate: {final_test['success_rate']:.1%}")

    # Generate training plots
    create_training_plots(all_rewards, all_scores, phase_results)

    print(f"\n✅ Training completed successfully!")
    print(f"Model saved as: {MODEL_FILE}")
    print(f"Results saved as: {RESULTS_FILE}")


def test_agent_performance(env, agent, num_episodes=10):
    """Test agent performance without learning"""
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # No exploration during testing

    test_scores = []
    successes = 0

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < 5000:
            action = agent.choose_action(state)
            state, _, done = env.step(action)
            steps += 1

        # Get score from environment (adjust attribute name as needed)
        current_score = getattr(env, 'score', 0) or getattr(env, 'current_score', 0)
        test_scores.append(current_score)
        if current_score > 100:  # Consider score > 100 as success
            successes += 1

    agent.epsilon = original_epsilon  # Restore original epsilon

    return {
        "avg_score": np.mean(test_scores) if test_scores else 0,
        "max_score": max(test_scores) if test_scores else 0,
        "success_rate": successes / num_episodes
    }


def save_training_results(rewards, scores, phase_results, filename):
    """Save training results to file"""
    results = {
        "timestamp": datetime.now().isoformat(),
        "rewards": rewards,
        "scores": scores,
        "phase_results": phase_results
    }

    with open(filename, 'wb') as f:
        pickle.dump(results, f)


def create_training_plots(rewards, scores, phase_results):
    """Create and save training visualization plots"""
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🦕 Dino AI Training Results', fontsize=16, fontweight='bold')

    # Plot 1: Rewards over time
    window = 50
    if len(rewards) >= window:
        rewards_smooth = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ax1.plot(rewards_smooth, color='blue', linewidth=2)
        ax1.set_title('Average Reward (50-episode moving average)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)

    # Plot 2: Scores over time
    if len(scores) >= window:
        scores_smooth = np.convolve(scores, np.ones(window) / window, mode='valid')
        ax2.plot(scores_smooth, color='green', linewidth=2)
        ax2.set_title('Average Score (50-episode moving average)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Score')
        ax2.grid(True, alpha=0.3)

    # Plot 3: Phase comparison
    if phase_results:
        phases = [f"Phase {r['phase']}" for r in phase_results]
        avg_scores = [r['avg_score'] for r in phase_results]
        max_scores = [r['max_score'] for r in phase_results]

        x = np.arange(len(phases))
        ax3.bar(x - 0.2, avg_scores, 0.4, label='Average Score', color='skyblue')
        ax3.bar(x + 0.2, max_scores, 0.4, label='Max Score', color='orange')
        ax3.set_title('Performance by Training Phase')
        ax3.set_xlabel('Training Phase')
        ax3.set_ylabel('Score')
        ax3.set_xticks(x)
        ax3.set_xticklabels(phases, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Plot 4: Score distribution
    if scores:
        ax4.hist(scores, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax4.set_title('Score Distribution')
        ax4.set_xlabel('Score')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)

        # Add statistics
        mean_score = np.mean(scores)
        max_score = max(scores)
        ax4.axvline(mean_score, color='red', linestyle='--',
                    label=f'Mean: {mean_score:.1f}')
        ax4.axvline(max_score, color='gold', linestyle='--',
                    label=f'Max: {max_score}')
        ax4.legend()

    plt.tight_layout()
    plt.savefig('dino_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📈 Training plots saved as: dino_training_results.png")


def quick_test_run():
    """Quick test to see current agent performance"""
    print("🧪 QUICK TEST RUN")
    print("=" * 30)

    env = DinoGame()
    agent = QLearningAgent(actions=["STAY", "JUMP", "DUCK"])

    if agent.load_model("dino_ai_enhanced.pkl"):
        print("Model loaded successfully!")

        # Run 5 test episodes
        test_results = test_agent_performance(env, agent, 5)

        print(f"Test Results (5 episodes):")
        print(f"  Average Score: {test_results['avg_score']:.1f}")
        print(f"  Max Score: {test_results['max_score']}")
        print(f"  Success Rate: {test_results['success_rate']:.1%}")
    else:
        print("No trained model found!")


def main():
    """Main function with menu options"""
    print("🦕 DINO AI TRAINING SYSTEM")
    print("=" * 40)
    print("1. Enhanced Curriculum Training")
    print("2. Quick Test Run")
    print("3. View Training History")
    print("4. Exit")

    choice = input("\nSelect option (1-4): ").strip()

    if choice == "1":
        enhanced_training_with_curriculum()
    elif choice == "2":
        quick_test_run()
    elif choice == "3":
        view_training_history()
    elif choice == "4":
        print("Goodbye! 🦕")
    else:
        print("Invalid choice!")


def view_training_history():
    """View previous training results"""
    try:
        with open("training_results.pkl", 'rb') as f:
            results = pickle.load(f)

        print("📊 TRAINING HISTORY")
        print("=" * 30)
        print(f"Last Training: {results['timestamp']}")
        print(f"Total Episodes: {len(results['scores'])}")

        if results['scores']:
            print(f"Average Score: {np.mean(results['scores']):.1f}")
            print(f"Max Score: {max(results['scores'])}")

        if results['phase_results']:
            print("\nPhase Results:")
            for phase in results['phase_results']:
                print(f"  {phase['description']}: Avg {phase['avg_score']:.1f}, "
                      f"Max {phase['max_score']}")

    except FileNotFoundError:
        print("No training history found!")


if __name__ == "__main__":
    main()