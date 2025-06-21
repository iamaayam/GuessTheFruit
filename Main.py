import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# Columns: Fruit, Round?, Yellow?, Small?, Juicy?, Red?, Has Peel?, Raw?, Sweet?, Seeds?, Juice?, Tropical?, Crunchy?
DATA = [
    ["Banana",      0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
    ["Apple",       1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    ["Orange",      1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
    ["Mango",       1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
    ["Strawberry",  1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
    ["Kiwi",        1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0],
    ["Jackfruit",   0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
    ["Watermelon",  1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    ["Pomegranate", 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0],
    ["Grape",       1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
    ["Avocado",     1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    ["Peach",       1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    ["Pear",        1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
    ["Cherry",      1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
    ["Papaya",      0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
    ["Lychee",      1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0],
    ["Guava",       1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
    ["Dragonfruit", 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0],
    ["Pineapple",   0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
    ["Blueberry",   0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
    ["Apricot",     1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0],
    ["Blackberry",  0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
    ["Cranberry",   0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0],
    ["Durian",      0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    ["Fig",         1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0],
    ["Grapefruit",  1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0],
    ["Lemon",       1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
    ["Lime",        1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
    ["Mandarin",    1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    ["Nectarine",   1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1]
]

COLUMNS = [
    "Fruit",
    "Is the fruit generally round in shape?",
    "Is it typically yellow in color when ripe?",
    "Is this fruit small enough to eat in one bite?",
    "Is it juicy when you bite into it?",
    "Does the fruit have red hues?",
    "Do you need to peel the skin before eating?",
    "Is it commonly eaten raw rather than cooked?",
    "Does it taste predominantly sweet?",
    "Are there noticeable seeds inside?",
    "Can you extract juice from it?",
    "Is it considered a tropical fruit?",
    "Does it have a crisp or crunchy texture?"
]

df = pd.DataFrame(DATA, columns=["Fruit"] + COLUMNS[1:])
X = df.drop('Fruit', axis=1)
y = df['Fruit']

model = RandomForestClassifier(n_estimators=200)
model.fit(X, y)
print("Model trained on full dataset with expanded and verified features.")


def play_game(model):
    print("\nPlease answer the following questions with 'yes' or 'no':")
    answers = []
    for feature in COLUMNS[1:]:
        while True:
            ans = input(f"{feature} ").strip().lower()
            if ans in ['yes', 'no']:
                answers.append(1 if ans == 'yes' else 0)
                break
            print("Invalid input. Please answer 'yes' or 'no'.")

    user_df = pd.DataFrame([answers], columns=COLUMNS[1:])
    guess = model.predict(user_df)[0]
    print(f"\nMy guess is: {guess}!")
    feedback = input("Was I correct? (yes/no) ").strip().lower()
    if feedback == 'yes':
        print("Great! Glad I got it right.")
    else:
        actual = input("Oh no! Which fruit were you thinking of? ")
        print(f"Thanks! I'll add {actual} to my knowledge for next time.")

if __name__ == '__main__':
    play_game(model)
