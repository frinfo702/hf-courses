import math
import random


# ❗ style issue: constant should be UPPERCASE, unused import (math)
pi_value = math.pi


# ❗ type issue: function hints say it returns int but it returns str
def lucky_number(name: str) -> int:
    """Return a fun lucky number message for a given name."""
    num = random.randint(1, 42)
    # ❗ f-string formatting but wrong return type
    return f"{name}'s lucky number is {num}!"


# ❗ formatting issue: bad spacing and quotes
def main():
    print(lucky_number("Ken"))
    print("This line has inconsistent quotes")

    if random.random() > 0.5:
        print("inline if-statement!")  # ❗ Ruff will suggest rewriting


if __name__ == "__main__":
    main()
