start_time = 0
end_time = 8 * 60 + 13  # Convert 8:13 to seconds

for timestamp in range(start_time, end_time + 1):
    minutes = timestamp // 60
    seconds = timestamp % 60

    print(f"{minutes:02d}:{seconds:02d}",end=" ")