import random

name_of_train_file = "saturn_data_train_second_ts.csv"
name_of_evaluation_file = "saturn_data_eval_second_ts.csv"

# Налаштування квадратів
size_of_first_square = 5
size_of_second_square = 5
size_of_third_square = 0  # Буферна зона навколо перших двох квадратів
range_between_square = 10

range_of_education = 3600
range_of_evaluation = 300
scale_of_data_per_class = 3

range_of_education_data_class = range_of_education // scale_of_data_per_class
range_of_evaluation_data_class = range_of_evaluation // scale_of_data_per_class

def run():
    fill(name_of_train_file, range_of_education_data_class)
    fill(name_of_evaluation_file, range_of_evaluation_data_class)

def fill(name_of_file, range_of_data_per_class):
    print(f"Filling file {name_of_file}")
    with open(name_of_file, "w") as f:
        array = []

        # Координати першого квадрата
        coords_x_start_for_first = -size_of_first_square
        coords_x_end_for_first = size_of_first_square
        coords_y_start_for_first = -size_of_first_square
        coords_y_end_for_first = size_of_first_square

        # Генерація першого класу (квадрат у центрі)
        for _ in range(range_of_data_per_class):
            array.append(f"1,{random.uniform(coords_x_start_for_first, coords_x_end_for_first)},"
                         f"{random.uniform(coords_y_start_for_first, coords_y_end_for_first)}\n")

        # Координати другого квадрата
        coords_x_start_for_second = size_of_first_square + range_between_square
        coords_x_end_for_second = coords_x_start_for_second + (size_of_second_square * 2)
        coords_y_start_for_second = -size_of_second_square
        coords_y_end_for_second = size_of_second_square

        # Генерація другого класу (зміщений квадрат)
        for _ in range(range_of_data_per_class):
            array.append(f"2,{random.uniform(coords_x_start_for_second, coords_x_end_for_second)},"
                         f"{random.uniform(coords_y_start_for_second, coords_y_end_for_second)}\n")

        # Генерація третього класу
        i = 0
        max_attempts = range_of_data_per_class * 20
        attempts = 0

        while i < range_of_data_per_class and attempts < max_attempts:
            x = random.uniform(-2 * (size_of_first_square + size_of_third_square + range_between_square),
                               2 * (size_of_first_square + size_of_third_square + range_between_square))
            y = random.uniform(-2 * (size_of_first_square + size_of_third_square + range_between_square),
                               2 * (size_of_first_square + size_of_third_square + range_between_square))

            # first quadrant
            zone_first_x_min = coords_x_start_for_first
            zone_first_x_max = coords_x_end_for_first
            zone_first_y_min = -size_of_first_square
            zone_first_y_max = size_of_first_square

            # second quadrant
            zone_second_x_min = coords_x_start_for_second
            zone_second_x_max = coords_x_end_for_second
            zone_second_y_min = coords_y_start_for_second
            zone_second_y_max = coords_y_end_for_second

            in_buffer_zone_first = (zone_first_x_min < x < zone_first_x_max) and (zone_first_y_min < y < zone_first_y_max)
            in_buffer_zone_second = (zone_second_x_min < x < zone_second_x_max) and (zone_second_y_min < y < zone_second_y_max)

            if not (in_buffer_zone_first or in_buffer_zone_second):
                array.append(f"0,{x},{y}\n")
                i += 1


            attempts += 1

        if attempts == max_attempts:
            print(f"Warning: Could not generate {range_of_data_per_class - i} points for class 0.")

        # Перемішуємо і записуємо у файл
        random.shuffle(array)
        f.writelines(array)

print("__Start__")
run()
print("__Done__")
