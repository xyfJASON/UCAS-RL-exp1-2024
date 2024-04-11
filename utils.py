import math
import matplotlib.pyplot as plt
import matplotlib.animation as ani


def plot(states, actions, rewards, save_path: str):
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.plot(rewards)
    plt.title('reward')

    plt.subplot(1, 4, 2)
    plt.plot([state.alpha for state in states])
    plt.ylim(-math.pi - 0.5, math.pi + 0.5)
    plt.title('alpha')

    plt.subplot(1, 4, 3)
    plt.plot([state.alpha_dot for state in states])
    plt.ylim(-15 * math.pi - 5, 15 * math.pi + 5)
    plt.title('alpha_dot')

    plt.subplot(1, 4, 4)
    plt.plot([action.u for action in actions])
    plt.ylim(-3.5, 3.5)
    plt.title('action')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def animate_pendulum(states, actions, save_path: str):
    xs = [math.sin(s.alpha) for s in states]
    ys = [math.cos(s.alpha) for s in states]

    fig = plt.figure(figsize=(4, 3), dpi=200)
    ax = fig.add_subplot(autoscale_on=False, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
    ax.set_aspect('equal')
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.grid()
    line, = ax.plot([], [], 'o-', lw=2)
    text = ax.text(0, -0.2, '')

    def draw(i):
        line.set_data([0, xs[i]], [0, ys[i]])
        text.set_text(str(actions[i].u))
        return line, text

    animator = ani.FuncAnimation(fig, draw, frames=len(xs), interval=5)
    animator.save(save_path, fps=50)
    plt.close()
