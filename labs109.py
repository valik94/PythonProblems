# As an example, here is an implementation of
# the first problem "Ryerson Letter Grade":

def ryerson_letter_grade(n):
    if n < 50:
        return 'F'
    elif n > 89 and n<=150:
        return 'A+'
    elif n > 84 and n<=150:
        return 'A'
    elif n > 79 and n<=150:
        return 'A-'
    elif n > 76 and n<=150:
        return 'B+'
    elif n > 72 and n<=150:
        return 'B'
    elif n > 69 and n<=150:
        return 'B-'
    elif n > 66 and n<=150:
        return 'C+'
    elif n > 62 and n<=150:
        return 'C'
    elif n > 59 and n<=150:
        return 'C-'
    elif n > 56 and n<=150:
        return 'D+'
    elif n > 52 and n<=150:
        return 'D'
    elif n > 49 and n<=150:
        return 'D-'

    tens = n // 10
    ones = n % 10
    if ones < 3:
        adjust = "-"
    elif ones > 6:
        adjust = "+"
    else:
        adjust = ""
    return "DCB"[tens - 5] + adjust

def riffle(items, out=True):
    for i in items:
        list_in=items[0:i+1]