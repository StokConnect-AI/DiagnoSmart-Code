def warn_if_out_of_range(value, name, threshold=10):
    if value > threshold:
        print(f"Warning: {name} value {value} exceeds expected range (0–{threshold}%)")
