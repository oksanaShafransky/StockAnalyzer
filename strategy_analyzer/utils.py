def calculate_consecutive_days(volume, sma150_volume):
    consecutive_days = []
    current_streak = 0

    for vol, sma in zip(volume, sma150_volume):
        if vol > sma:
            current_streak += 1
        else:
            current_streak = 0
        consecutive_days.append(current_streak)

    return consecutive_days

def calculate_consecutive_days2(series):
    consecutive_days = []
    current_streak = 0

    for val in series:
        if val > 0:
            current_streak += 1
        else:
            current_streak = 0
        consecutive_days.append(current_streak)

    return consecutive_days

def calculate_consecutive_boolean(series):
    consecutive_days = []
    current_streak = 0

    for value in series:
        if value:
            current_streak += 1
        else:
            current_streak = 0
        consecutive_days.append(current_streak)

    return consecutive_days