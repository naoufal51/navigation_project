import os

if __name__ == "__main__":
    dueling_values = [True, False]
    double_values = [True, False]

    for dueling in dueling_values:
        for double in double_values:
            print(f"dueling: {dueling}, double: {double}")
            command = f"python3 train.py {dueling} {double}"
            os.system(command)
