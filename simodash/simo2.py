import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def simulate_timestamps(open_time, interval_seconds=180, duration_seconds=14400):
    """
    Generates timestamps by incrementing open_time cumulatively.

    :param open_time: Start time (datetime object)
    :param interval_seconds: Step size in seconds (default: 900s = 15 minutes)
    :param duration_seconds: Total duration in seconds (default: 14400s = 4 hours)
    :return: List of timestamps
    """
    timestamps = [open_time + timedelta(seconds=i * interval_seconds) for i in range(duration_seconds // interval_seconds)]
    return timestamps

def compute_c_values(timestamps, interval_seconds=900):
    """
    Computes c(t) = t - floor(t / interval) for given timestamps.

    :param timestamps: List of datetime objects
    :param interval_seconds: Interval in seconds (default: 900s = 15 minutes)
    :return: List of computed c(t) values
    """
    return [t.timestamp() - (t.timestamp() // interval_seconds) * interval_seconds for t in timestamps]

def count_consecutive_jumps(c_values):
    """
    Counts longest streaks of consecutive positive and negative jumps.

    :param c_values: List of c(t) values
    :return: Longest consecutive positive and negative jumps
    """
    jumps = [c_values[i] - c_values[i-1] for i in range(1, len(c_values))]

    longest_pos, longest_neg = 0, 0
    current_pos, current_neg = 0, 0

    for jump in jumps:
        if jump > 0:
            current_pos += 1
            current_neg = 0
        elif jump < 0:
            current_neg += 1
            current_pos = 0
        else:
            current_pos, current_neg = 0, 0

        longest_pos = max(longest_pos, current_pos)
        longest_neg = max(longest_neg, current_neg)

    return longest_pos, longest_neg

def plot_c_values(timestamps, c_values):
    """
    Plots the computed c(t) values.

    :param timestamps: List of datetime timestamps
    :param c_values: Corresponding c(t) values
    """
    plt.figure(figsize=(10, 5))
    #plt.plot(timestamps, c_values, marker='o', linestyle='-', label='c(t)')
    plt.step(timestamps, c_values, where='post', marker='o', label='c(t)')
    plt.xlabel("Time")
    plt.ylabel("c(t) values")
    plt.title("Simulated c(t) Values Over Time")
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()
    plt.show()

def simulate_process(open_time, interval_seconds=180, duration_seconds=14400):
    """
    Simulates the full process: generates timestamps, computes c(t), counts jumps, and plots results.

    :param open_time: Start time (datetime object)
    :param interval_seconds: Step size in seconds (default: 900s = 15 minutes)
    :param duration_seconds: Total duration in seconds (default: 14400s = 4 hours)
    """
    timestamps = simulate_timestamps(open_time, interval_seconds, duration_seconds)
    c_values = compute_c_values(timestamps, interval_seconds)

    longest_pos, longest_neg = count_consecutive_jumps(c_values)

    print(f"Longest Positive Jump Streak: {longest_pos}")
    print(f"Longest Negative Jump Streak: {longest_neg}")

    plot_c_values(timestamps, c_values)

# Example usage
open_time = datetime(2023, 10, 27, 18, 45)


t = simulate_timestamps(open_time)

c_vals = compute_c_values(timestamps=t)

print(t)
print(c_vals)

print(count_consecutive_jumps(c_values=c_vals))

plot_c_values(t, c_values=c_vals)
