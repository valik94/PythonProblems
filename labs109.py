# VERSION December 8, 2019

from fractions import Fraction
from collections import deque
from math import sqrt
from bisect import bisect_left
from functools import lru_cache
from itertools import combinations, chain, groupby

suits = ['clubs', 'diamonds', 'hearts', 'spades']
ranks = {'deuce' : 2, 'trey' : 3 , 'four' : 4, 'five' : 5,
         'six' : 6, 'seven' : 7, 'eight' : 8, 'nine' : 9,
         'ten' : 10, 'jack' : 11, 'queen' : 12, 'king' : 13,
         'ace' : 14 }
deck = [ (rank, suit) for suit in suits for rank in ranks.keys() ] 

def all_cyclic_shifts(text):
    return list(sorted(list(set([text[i:] + text[:i] for i in range(0, len(text))]))))  
    
__scrabble = {
        "a": 1, "b": 3, "c": 3, "d": 2, "e": 1, "f": 4, "g": 2, "h": 4,
        "i": 1, "j": 8, "k": 5, "l": 1, "m": 3, "n": 1, "o": 1, "p": 3,
        "q": 10, "r": 1, "s": 1, "t": 1, "u": 1, "v": 4, "w": 4, "x": 8,
        "y": 4, "z": 10
}

def scrabble_value(word, multipliers):
    if multipliers == None:
        multipliers = [1] * len(word)
    return sum([__scrabble[c] * m for (c, m) in zip(word, multipliers)])


def substitution_words(pattern, words):
    ll, result = len(pattern), []
    for word in words:
        if len(word) == ll:
            subs,taken = dict(), set()            
            for i in range(ll):
                c1, c2 = pattern[i], word[i]
                if c1 in subs:
                    if subs[c1] != c2:
                        break
                else:
                    if c2 in taken:
                        break
                    else:
                        taken.add(c2)
                        subs[c1] = c2
            else:
                result.append(word)
    return result


__gprimes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
              53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]

def __godel(word):    
    m = 1
    for c in word:
        m *= __gprimes[ord(c) - ord('a')]
    return m

def unscramble(words, word):    
    result, first, last, g = [], word[0], word[-1], __godel(word)
    start = bisect_left(words, first)
    end = bisect_left(words, chr(ord(first) + 1))    
    for i in range(start, end):
        w = words[i]
        if len(w) == len(word) and w[-1] == last and __godel(w) == g:        
            result.append(w)    
    return result


def __word_match(word, letters):
    pos = 0
    for c in word:
        if c == letters[pos]:
            pos += 1
            if pos == len(letters):
                return True
    return False

def words_with_letters(words, letters):
    result = []
    for word in words:
        if __word_match(word, letters):
            result.append(word)    
    return result


def autocorrect_word(word, words, df):
    best, bd = '', 100000
    for w in words:
        if len(w) == len(word):           
            d = sum((df(c1, c2) for (c1, c2) in zip(w, word)))
            if d < bd:
                bd, best = d, w    
    return best


def pancake_scramble(text):
    for i in range(2, len(text) + 1):
        text = text[:i][::-1] + text[i:]
    return text


def words_with_given_shape(words, shape):
    result = []
    for w in words:
        if len(w) != len(shape) + 1:
            continue
        for i in range(len(w) - 1):
            sign = ord(w[i+1]) - ord(w[i])
            if sign < 0 and shape[i] != -1:
                break
            if sign == 0 and shape[i] != 0:
                break
            if sign > 0 and shape[i] != 1:
                break
        else:
            result.append(w)
    return result


__left = "qwertyasdfghzxcvb"
__right = "uiopjklnm"

def __score(word):
    score = 0
    for ch in word:
        if ch in __left:
            score += 1
        elif ch in __right:
            score -= 1
    return score
        
def sort_by_typing_handedness(words):
    words.sort(key = lambda w: (-__score(w), w) )
    return words


def possible_words(words, pattern):
    result = []
    for word in words:
        if len(word) == len(pattern):          
            for (chw, chp) in zip(word, pattern):
                if chp == '*' and chw in pattern:
                    break
                if chp != '*' and chp != chw:
                    break
            else:
                result.append(word)
    return result


def connected_islands(n, bridges, queries):
    rep = [i for i in range(n)]
    def find(i):
        j = i
        while rep[j] != j:
            j = rep[j]
        while i != j:
            rep[i], i = j, rep[i]
        return rep[i]                
    for (s, e) in bridges:
        ss, ee = find(s), find(e)
        rep[ss] = ee
    result = [find(a) == find(b) for (a, b) in queries]    
    return result


def __digikey(a):
    aa = str(a)
    return ([aa.count(d) for d in "9876543210"], a)
    
def sort_by_digit_count(items):
    return list(sorted(items, key = __digikey))


def arithmetic_progression(elems):    
    best, eset, n, tabu = None, set(elems), len(elems), set()
    if n == 1:
        return (elems[0], 0, 1)
    for i in range(n-1):
        e1 = elems[i]
        if best != None and best[2] > n - i:
            break
        for j in range(i+1, n):
            e2 = elems[j]
            step = e2 - e1
            if (e1, step) not in tabu:
                k = 1
                while e2 in eset:
                    tabu.add((e2, step))
                    k += 1
                    e2 += step
                if best == None or best[2] < k:
                    best = (e1, step, k)
    return best


def create_zigzag(rows, cols, start = 1):
    rows = [list(range(x, x + cols)) for x in range(start, start + rows * cols, cols)]
    return [(row if idx % 2 == 0 else list(reversed(row))) for (idx, row) in enumerate(rows)]


def two_summers(items, goal, i = None, j = None):
    if i == None:
        i = 0
    if j == None:
        j = len(items) - 1
    while i < j:
        x = items[i] + items[j]
        if x == goal:
            return True
        elif x < goal:
            i += 1
        else:
            j -= 1
    return False

def three_summers(items, goal):
    for i in range(len(items) - 2):
        if two_summers(items, goal - items[i], i + 1):
            return True
    return False


def highest_n_scores(scores, n = 5):
    totals = {}
    for (name, score) in scores:        
        if name in totals:
            totals[name].append(score)
        else:
            totals[name] = [score]
    result = []
    for name in totals:
        s = totals[name]
        if len(s) > n:
            s = list(sorted(s))[-n:]
        result.append((name, sum(s)))
    result.sort()
    return result


def sum_of_two_squares(n):
    a, b = 1, 1
    while a*a < n:
        a += 1
    while a >= b:
        d = a*a + b*b
        if d == n:
            return (a, b)
        elif d < n:
            b += 1
        else:
            a -= 1        
    return None


def fractional_fit(fs):
    global __best    
    n = len(fs)
    fs = [Fraction(a, b) for (a, b) in fs]
    positions = [ [int(f / Fraction(1, k)) for k in range(1, n+1)] for f in fs]
    sofar, remain = [], list(range(n))      
    
    def rec(k):
        best = 0
        for i in remain[:]:
            sofar.append(i)
            remain.remove(i)
            for (i1, i2) in combinations(sofar, 2):
                if positions[i1][k] == positions[i2][k]:
                    break
            else:
                best = max(best, 1 + rec(k + 1))
            sofar.pop()
            remain.append(i)
            if best == n - k:
                return best        
        return best
    
    return rec(0)


def scylla_or_charybdis(seq, n):
    k, best, bestk = 1, None, None  
    while k <= len(seq) // n:
        pos, count = 0, 0  
        for m in seq[k-1::k]:
            count += 1
            pos += 1 if m == '+' else -1    
            if abs(pos) == n:
                if best == None or best > count:
                    best, bestk = count, k      
                break    
        k += 1
    return bestk


def count_overlapping_disks(disks):
    # 0 means enter, 1 means exit
    events = [(x - r, 0, (x, y, r)) for (x, y, r) in disks] +\
             [(x + r, 1, (x, y, r)) for (x, y, r) in disks]
    events.sort() # enter events for same x before exit events
    count, active = 0, set()
    for (_, mode, (x, y, r)) in events:
        if mode == 0:
            for (xx, yy, rr) in active:
                if (xx-x)**2 + (yy-y)**2 <= (rr+r)**2:                    
                    count += 1
            active.add((x, y, r))
        else:
            active.remove((x, y, r))    
    return count
        

def manhattan_skyline(towers):
    # 0 means enter, 1 means exit
    sweep = [(s, i, h, 0) for (i, (s, e, h)) in enumerate(towers)] +\
            [(e, i, h, 1) for (i, (s, e, h)) in enumerate(towers)]
    sweep.sort()
    area, prevx, active = 0, 0, set()
    for (x, i, h, mode) in sweep:
        # Add the area from the previous step.
        if len(active) > 0:
            tallest = max((hh for (i, hh) in active))            
            area += (x - prevx) * tallest
        # Update the contents of the active set depending on mode.
        if mode == 0:
            active.add((i, h))
        else:
            active.remove((i, h))            
        prevx = x
    return area


def fractran(n, prog, giveup = 1000):
    prog = [Fraction(a, b) for (a, b) in prog]
    result, count = [n], 0
    while count < giveup:
        for f in prog:
            v = n * f
            if v.denominator == 1:
                n = v.numerator
                count += 1
                result.append(n)
                break
        else:
            break
    return result
 

def bulgarian_solitaire(piles, k):
    goal, count = [i for i in range(k, 0, -1)], 0    
    while True:        
        if len(piles) == k:
            piles.sort(reverse = True)
            if piles == goal:
                return count
        count += 1
        piles = [p - 1 for p in piles if p > 1] + [len(piles)]
    return count


def longest_palindrome(text):
    best = ''
    for i in range(len(text)):
        j = 1
        while j <= i and i + j < len(text):
            if text[i-j] == text[i+j]:
                j += 1
            else:
                break
        n = 2*j - 1
        if n > len(best):
            best = text[i-j+1:i+j]
        k = 0
        while k <= i and i + k < len(text) - 1:
            if text[i-k] == text[i+1+k]:
                k += 1
            else:
                break
        n = 2*k
        if n > len(best):
            best = text[i-k+1:i+k+1]
    return best


#def __line_equation(x1, y1, x2, y2):
#    a = y2 - y1
#    b = x1 - x2
#    c1 = -a * x1 - b * y1
#    c2 = -a * x2 - b * y2    
#    assert c1 == c2
#    return (a, b, c1)
#
#def __gcd(a, b):
#    if a == 0 and b == 0:
#        return 1
#    while b > 0:
#        a, b = b, a % b
#    return a
#   
#def count_distinct_lines2(points):
#    points.sort()
#    lines = set()
#    for i in range(len(points) - 1):
#        for j in range(i + 1, len(points)):
#            (a, b, c) = __line_equation(*points[i], *points[j])
#            g = __gcd(__gcd(abs(a), abs(b)), abs(c))    
#            lines.add((a // g, b // g, c // g))
#    return len(lines)

def count_distinct_lines(points):
    points.sort()
    total, tabu = 0, set()
    for i in range(len(points) - 1):
        p1 = points[i]        
        for j in range(i + 1, len(points)):
            if (i, j) not in tabu:             
                p2 = points[j]                
                for k in range(j + 1, len(points)):
                    p3 = points[k]
                    if collinear(*p1, *p2, *p3):
                        tabu.add((i, k))
                        tabu.add((j, k))
                total += 1
    return total

def first_missing_positive(items):    
    upto, above = 0, set()    
    for v in items:        
        if v <= upto:
            continue
        elif v == upto + 1:
            upto = v
            v += 1
            while v in above:
                above.remove(v)
                upto = v
                v += 1
        else:
            above.add(v)
    return upto + 1    


def cross(x1, y1, x2, y2, x3, y3):
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

def collinear(x1, y1, x2, y2, x3, y3):
    return cross(x1, y1, x2, y2, x3, y3) == 0

def line_with_most_points(points):
    most, tabu = 0, set()
    for i in range(len(points) - 1):
        p1 = points[i]        
        for j in range(i + 1, len(points)):
            if (i, j) not in tabu:             
                p2 = points[j]
                count = 2
                for k in range(j + 1, len(points)):
                    p3 = points[k]
                    if collinear(*p1, *p2, *p3):
                        tabu.add((i, k))
                        tabu.add((j, k))
                        count += 1
                most = max(most, count)
    return most


__cdirs = [(-1, 1), (1, 1), (1, -1), (-1, -1)]

def __inside(x, y, n):
    return 0 <= x < n and 0 <= y < n

def max_checkers_capture(n, x, y, pieces):
    best = 0
    for (dx, dy) in __cdirs:
        if __inside(x + 2*dx, y + 2*dy, n) and (x + 2*dx, y + 2*dy) not in pieces and (x + dx, y + dy) in pieces:
            pieces.remove((x + dx, y + dy))
            best = max(best, 1 + max_checkers_capture(n, x + 2 * dx, y + 2 * dy, pieces))
            pieces.add((x + dx, y + dy))
    return best


__cubes = [x*x*x for x in range(0, 10000)]

def sum_of_distinct_cubes(n, k = len(__cubes)):
    if n == 0:
        return []
    if n < 0 or k < 1:
        return None
    i = min(bisect_left(__cubes, n), k)
    while i >= 0:
        ans = sum_of_distinct_cubes(n - __cubes[i], i - 1)
        if ans != None:
            return [i] + ans 
        i -= 1
    return None


def __crag(dice, cat):    
    s = sum(dice)
    if cat == 0: # Crag
        if s == 13 and (dice[0] == dice[1] or dice[1] == dice[2]):
            return 50
        else:
            return 0
    elif cat == 1: # Thirteen
        return 26 if s == 13 else 0
    elif cat == 2: # Three of a kind
        return 25 if dice[0] == dice[2] else 0
    elif cat == 3: # Low straight
        return 20 if dice == [1, 2, 3] else 0
    elif cat == 4: # High straight
        return 20 if dice == [4, 5, 6] else 0
    elif cat == 5: # Odd straight
        return 20 if dice == [1, 3, 5] else 0
    elif cat == 6: # Even straight
        return 20 if dice == [2, 4, 6] else 0
    else: # Pip values
        pip = 13 - cat # pip value to add up
        return sum([x for x in dice if x == pip])

def __crag_score(rolls, limit, cats, i, tobeat):
    if i == len(rolls) or tobeat >= limit[i]:
        return 0
    best, poss = 0, [(__crag(rolls[i], cat), cat) for cat in cats]
    poss.sort(reverse = True) 
    for (curr, cat) in poss:
        cats.remove(cat)        
        curr += __crag_score(rolls, limit, cats, i + 1, best - curr)        
        best = max(best, curr)
        cats.add(cat)
        if best == limit[i]:
            break
    return best

def optimal_crag_score(rolls):
    rolls = [sorted(dice) for dice in rolls]
    rolls.sort(key = lambda d: crag_score(d), reverse = True)
    limit = [crag_score(dice) for dice in rolls]
    for i in range(len(rolls) - 2, -1, -1):
        limit[i] += limit[i+1]
    result = __crag_score(rolls, limit, set(range(13)), 0, 0)    
    return result


def nearest_smaller(items):
    n, res = len(items), []
    for (i, e) in enumerate(items):
        j = 1       
        while j < n:
            left = items[i-j] if i >= j else e
            right = items[i+j] if i+j < n else e
            if left < e or right < e:
                res.append(left if left < right else right)                
                break
            j += 1
        else:
            res.append(e)
    return res


def safe_squares_knights(n, knights): 
    unsafe, moves = set(), ((0,0),(2,1),(1,2),(2,-1),(-1,2),(-2,1),(1,-2),(-2,-1),(-1,-2))
    for (r, c) in knights:
        for (dx, dy) in moves:
            nx, ny = r+dx, c+dy
            if 0 <= nx < n and 0 <= ny < n:
                unsafe.add((r+dx, c+dy))
    return n*n - len(unsafe)
    

def hand_is_badugi(hand):
    return all(r1 != r2 and s1 != s2 for ((r1, s1), (r2, s2)) in combinations(hand, 2))
        

def reverse_vowels(text):
    vowels, result = [c for c in text if c in 'aeiouAEIOU'], ''    
    for c in text:
        if c in 'aeiouAEIOU':
            ch = vowels.pop()
            ch = ch.upper() if c in 'AEIOU' else ch.lower()
            result += ch            
        else:
            result += c
    return result


def josephus(n, k):
    soldiers, result = list(range(1, n + 1)), []    
    while n > 0:
        pos = (k - 1) % n
        result.append(soldiers[pos])
        if pos == n - 1:
            soldiers = soldiers[:-1]
        else:
            soldiers = soldiers[pos + 1:] + soldiers[:pos]
        n -= 1
    return result


__sz = [7]

def seven_zero(n):
    dig = 1
    while True:
        if len(__sz) < dig:
            __sz.append(__sz[-1] * 10)
        s = 0
        for j in range(dig-1, -1, -1):
            s += __sz[j]
            if s % n == 0:
                return s
        dig += 1


def calkin_wilf(n):
    q, m = deque(), 1
    q.append(Fraction(1, 1))    
    while m < n:
        f = q.popleft()
        num, den = f.numerator, f.denominator
        q.append(Fraction(num, num + den))
        q.append(Fraction(num + den, den))
        m += 2
    return q[n - m - 1]


def pyramid_blocks(n, m, h):
    return h * (1 - 3*h + 2*h*h - 3*m + 3*h*m - 3*n + 3*h*n + 6*m*n) // 6


def count_dominators(items):
    if len(items) == 0:
        return 0
    total, big = 0, items[-1] - 1
    for e in reversed(items):
        if e > big:
            total += 1
            big = e
    return total


def square_follows(it):
    result, buf, curr = [], deque(), next(it)
    for e in it:
        buf.append(e)
        while e > curr * curr:
            curr = buf.popleft()
        if curr * curr == e:
            result.append(curr)
    return result


def reverse_ascending_sublists(items):
    result, curr = [], []
    for x in chain(items, [None]):
        if x != None and (curr == [] or curr[-1] < x):
            curr.append(x) 
        else:
            curr.reverse()
            result.extend(curr)
            curr = [x]
    return result


def bridge_hand_shape(hand):
    spades, hearts, diamonds, clubs = 0, 0, 0, 0
    for (rank, suit) in hand:
        if suit == "spades":
            spades += 1
        elif suit == "hearts":
            hearts += 1
        elif suit == "diamonds":
            diamonds += 1
        elif suit == "clubs":
            clubs += 1
    return [spades, hearts, diamonds, clubs]
    

def hand_shape_distribution(hands):
    result = {}
    for hand in hands:
        ashape = tuple(reversed(sorted(bridge_hand_shape(hand))))
        result[ashape] = result.get(ashape, 0) + 1
    return result


def group_equal(items):
    return [list(x[1]) for x in groupby(items)]


def extract_increasing(digits):
    curr, big, result = 0, -1, []
    for d in digits:
        curr = 10 * curr + int(d)
        if curr > big:
            result.append(curr)
            big = curr
            curr = 0
    return result


def give_change(amount, coins):
    coins = iter(coins)
    curr = next(coins)
    result = []
    while amount > 0:
        if amount >= curr:
            amount -= curr
            result.append(curr)
        else:            
            curr = next(coins)
    return result


def __no_repeated_digits(n):
    return '0' not in str(n) and\
     all((c1 != c2 for (c1, c2) in combinations(str(n), 2)))

def __consistent(n, m, bulls, cows):
    bc, cc, n, m = 0, 0, str(n), str(m)
    for (c1, c2) in zip(n, m):
        if c1 == c2:
            bc += 1
        elif c2 in n:
            cc += 1
    return bc == bulls and cc == cows

def bulls_and_cows(guesses):
    possible = [x for x in range(1000, 10000) if __no_repeated_digits(x)]
    for (guess, bulls, cows) in guesses:
        possible = [x for x in possible if __consistent(x, guess, bulls, cows)]
    return possible


def count_and_say(digits):
    result, prev, count = '', '$', 0
    for d in chain(digits, ['$']):
        if d == prev:
            count += 1
        else:
            if prev != '$':
                result += str(count)
                result += prev
            prev = d
            count = 1
    return result


def __has_all_digits(n):
    n = str(n)
    return all((ch in n for ch in '0123456789'))


def double_until_all_digits(n, giveup = 1000):
    count = 0
    while count < giveup:
        if __has_all_digits(n):
            return count
        count += 1
        n = n * 2
    return -1


def detab(text, n = 8, sub = ' '):
    result, i = '', 0    
    for c in text:
        if c == '\t':
            if i % n == 0:
                result += sub * n                
            else:
                result += sub * (n - i % n)                
                i += (n - i % n)
        else:
            result += c
            i += 1
    return result


def reverse_reversed(items):
    if isinstance(items, list):
        return list(reversed([reverse_reversed(x) for x in items]))
    else:
        return items


def ryerson_letter_grade(n):
    if n < 50:
        return 'F'
    elif n > 89:
        return 'A+'
    elif n > 84:
        return 'A'
    elif n > 79:
        return 'A-'
    tens = n // 10
    ones = n % 10
    if ones < 3:
        adjust = "-"
    elif ones > 6:
        adjust = "+"
    else:
        adjust = ""
    return "DCB"[tens - 5] + adjust


def squares_intersect(s1, s2):
    (x1, y1, d1) = s1
    (x2, y2, d2) = s2
    return not (x1 + d1 < x2 or x2 + d2 < x1 or y1 + d1 < y2 or y2 + d2 < y1)


def only_odd_digits(n):    
    return all((c not in '02468') for c in str(n))


def first_preceded_by_smaller(items, k = 1):
    for i in range(k, len(items)):
        count = 0
        for j in range(i):
            if items[j] < items[i]:
                count += 1
        if count >= k:
            return items[i]
    return None


def maximum_difference_sublist(items, k = 2):
    m, mpos = 0, 0
    for i in range(len(items) - k + 1):
        sub = items[i:i+k]
        diff = max(sub) - min(sub)
        if diff > m:
            m = diff
            mpos = i
    return items[mpos:mpos+k]


def __med2(a, b, c):
    if a < b and a < c:
        return min(b, c)
    elif a > b and a > c:
        return max(b, c)
    else:
        return a

def __med(a, b, c):
    return sorted([a,b,c])[1]

def running_median_of_three(items):
    result = [0 for x in range(len(items))]
    for i in range(len(items)):
        if i < 2:
            result[i] = items[i];
        else:
            result[i] = __med(items[i-2],items[i-1],items[i])
    return result


def is_ascending(seq):
    first = True
    for x in seq:
        if first:
            prev, first = x, False
        else:
            if prev >= x:
                return False
        prev = x
    return True


def disemvowel(text):
    result = ''
    for (i, c) in enumerate(text):
        left = i > 0 and text[i-1] in 'aeiou'
        right = i < len(text) - 1 and text[i+1] in 'aeiou'
        if not(c in 'aeiou' or (c == 'y' and (left or right))):
            result += c
    return result


def milton_work_point_count(hand, trump = 'notrump'):
    score, values = 0, {'ace':4, 'king':3, 'queen':2, 'jack':1}
    for (rank, suit) in hand:
        score += values.get(rank, 0)
    shape = bridge_hand_shape(hand)
    if list(sorted(shape)) == [3, 3, 3, 4]:
        score -= 1
    ti = ['spades', 'hearts', 'diamonds', 'clubs', 'notrump']
    if trump != 'notrump':
        for i in range(4):
            if ti[i] != trump and shape[i] < 2:
                score += [5, 3][shape[i]]
    for i in range(4):
        if shape[i] == 5:
            score += 1
        elif shape[i] == 6:
            score += 2
        elif shape[i] >= 7:
            score += 3    
    return score
    

__abbrv = {14:'A', 13:'K', 12:'Q', 11:'J'}    
def __canonize_suit(ranks):
    if ranks == []:
        return '-'
    return "".join([__abbrv.get(r, 'x') for r in ranks])

def bridge_hand_shorthand(hand):   
    return " ".join([__canonize_suit(list(sorted([ranks[r] for (r, s) in hand if s == suit], reverse=True)))
    for suit in ['spades', 'hearts', 'diamonds', 'clubs']])    

__losers = {'-':0,'A':0,'x':1,'Q':1,'K':1,'AK':0, 'AQ':1, 'Ax':1,
            'KQ':1,'Kx':1, 'Qx':2,'xx':2, 'AKQ':0, 'AKx':1, 'AQx':1,
            'Axx':2, 'Kxx':2, 'KQx':1, 'Qxx':2, 'xxx':3}
       
def losing_trick_count(hand):
    return sum([__losers.get(s[:3], len(s[:3]))
                for s in bridge_hand_shorthand(hand).replace('J','x').split(" ")])
    

def iterated_remove_pairs(items):
    result = []
    for e in items:
        if len(result) == 0 or result[-1] != e:
            result.append(e)
        else:
            del result[-1]
    return result


ranks = {'deuce' : 2, 'trey' : 3 , 'four' : 4, 'five' : 5,
         'six' : 6, 'seven' : 7, 'eight' : 8, 'nine' : 9,
         'ten' : 10, 'jack' : 11, 'queen' : 12, 'king' : 13,
         'ace' : 14 }

def winning_card(cards, trump = None):
    winner = cards[0]
    for idx in range(1, 4):
        curr = cards[idx]
        if winner[1] != trump and curr[1] == trump:
            winner = curr
        elif winner[1] == curr[1] and ranks[winner[0]] < ranks[curr[0]]:
            winner = curr
    return winner


def postfix_evaluate(items):
    stack = []
    for it in items:
        if type(it) == type("%"):
            o2 = stack.pop()
            o1 = stack.pop()
            if it == '+':
                stack.append(o1 + o2)
            elif it == '-':
                stack.append(o1 - o2)
            elif it == '*':
                stack.append(o1 * o2)
            elif it == '/':
                if o2 != 0:
                    stack.append(o1 // o2)
                else:
                    stack.append(0)
        else:
            stack.append(it)
    return stack[0]
    

def expand_intervals(intervals):
    result = []
    for item in intervals.split(","):
        parts = item.partition("-")
        if parts[1] == "-":
            start = parts[0]
            end = parts[2]
        else:
            start = end = parts[0]
        result.extend(range(int(start), int(end)+1))
    return result


def __encode_interval(curr, first):
    result = '' if first else ','
    if len(curr) > 1:
        result += str(curr[0]) + "-" + str(curr[-1])
    else:
        result += str(curr[0])
    return result


def collapse_intervals(items):
    result, curr, first = '', [], True
    for item in items:
        if curr == [] or item == curr[-1] + 1:
            curr.append(item)
        else:
            result += __encode_interval(curr, first)
            first = False
            curr = [item]
    result += __encode_interval(curr, first)
    return result


def recaman(n):
    result, prev, taken = [1], 1, set([1])
    for i in range(2, n+1):
        curr = prev - i
        if curr <= 0 or curr in taken:
            curr = prev + i
        result.append(curr)
        taken.add(curr)
        prev = curr
    return result


def can_balance(items):
    for i in range(0, len(items)):
        left = sum([(i - j) * items[j] for j in range(0, i)])
        right = sum([(j - i) * items[j] for j in range(i+1, len(items))])
        if left == right:
            return i
    return -1


def is_perfect_power(n):
    p = 2
    while 2**p <= n:
        i, j = 2, n // p
        while i < j:
            m = (i + j) // 2
            if m**p < n:
                i = m + 1
            else:
                j = m
        if i**p == n:
            return True
        p += 1
    return False


def frequency_sort(elems):
    counts = {}
    for e in elems:
        counts[e] = counts.get(e, 0) + 1
    elems.sort(key = lambda x:(-counts.get(x), x))
    return elems


def count_consecutive_summers(n):
    i, j, count, curr = 1, 1, 0, 1
    while j <= n:
        if curr == n:
            count += 1
        if curr <= n:
            j += 1
            curr += j
        else:
            curr -= i
            i += 1
            if i > j:
                j = i
    return count


def count_divisibles_in_range(start, end, n):
   (s, e) = (start, end)
   if s < 0 and e <= 0:
       a2 = count_divisibles_in_range(-e, -s, n)
   elif s < 0:
       a2 = count_divisibles_in_range(0, -s, n) + count_divisibles_in_range(1, e, n)
   else:
        if s % n > 0:
            s += (n - s % n)
        e -= e % n
        if s > e:
            a2 = 0
        else:
            a2 = (e - s) // n + 1
   return a2


def contains_bingo(card, numbers, center_free = True):
    # rows
    for row in range(5):
        for col in range(5):
            if row == 2 and col == 2 and center_free:
                continue
            if card[row][col] not in numbers:
                break
        else:
            return True
    # columns
    for col in range(5):
        for row in range(5):
            if row == 2 and col == 2 and center_free:
                continue
            if card[row][col] not in numbers:
                break
        else:
            return True   
    # diagonal
    for d in range(5):
        if d == 2 and center_free:
            continue
        if card[d][d] not in numbers:
            break
    else:
        return True
    # anti-diagonal       
    for d in range(5):
        if d == 2 and center_free:
            continue
        if card[d][4-d] not in numbers:
            break
    else:
        return True
    return False


__fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55]

def fibonacci_sum(n):
    while n > __fibs[-1]:
        __fibs.append(__fibs[-1] + __fibs[-2])
    i, j, result = 0, len(__fibs) - 1, []
    while i < j:
        m = (i + j) // 2
        if __fibs[m] < n:
            i = m + 1
        else:
            j = m
    while n > 0:
        if n >= __fibs[i]:
            result.append(__fibs[i])
            n -= __fibs[i]
        i -= 1
    return result

__props = {}

def __proper_divisors(n):
    if n in __props:
        return __props[n]
    curr, result = 1, []
    while curr < n:
        if n % curr == 0:
            result.append(curr)
        curr += 1
    __props[n] = result
    return result


def aliquot_sequence(n, giveup = 100):
    result, results = [n], set([n])
    while len(result) < giveup:
        m = sum(__proper_divisors(n))
        if m in results:
            return result
        result.append(m)
        results.add(m)
        n = m
    return result


__primelist = [2, 3, 5, 7, 11]

def __expand_primes(n):
    # Start looking for new primes after the largest prime we know.
    m = __primelist[-1] + 2
    while n > __primelist[-1]:
        if __is_prime(m):    
            __primelist.append(m)
        m += 2
    
def __is_prime(n):
    # To check whether n is prime, check its divisibility with 
    # all known prime numbers up to the square root of n.
    upper = 1 + int(sqrt(n))
    # First ensure that we have enough primes to do the test.    
    __expand_primes(upper)
    for d in __primelist:
        if n % d == 0:
            return False
        if d * d > n:
            return True
    return True

__factors = {}

def prime_factors(n):
    if n < 2:
        return []
    if n in __factors:
        return [__factors[n]] + prime_factors(n // __factors[n])
    idx = 0
    while __primelist[idx] <= n:
        p = __primelist[idx]
        if n % p == 0:
            if p > 20:
                __factors[n] = p
            return [p] + prime_factors(n // p)
        else:
            idx += 1
            if idx >= len(__primelist):
                __expand_primes(__primelist[-1] * 2)
    __factors[n] = n
    return [n] # n is a prime number


def factoring_factorial(n):
    res = {}
    for k in range(2, n+1):
        for pf in prime_factors(k):
            res[pf] = res.get(pf, 0) + 1
    return list(sorted([(pf, res[pf]) for pf in res]))


def balanced_ternary(n, pp = None, sp = None):
    if abs(n) < 2:
        return [[-1],[],[1]][n+1]
    if n < 0:
        return [-x for x in balanced_ternary(-n, pp, sp)]
    if not pp:
        pp = 1
        sp = 0
        while 3 * pp <= n:
            sp += pp
            pp = 3 * pp
    else:
        while pp > n:
            pp = pp // 3
            sp = sp - pp
    if n <= pp + sp:
        return [pp] + balanced_ternary(n - pp, pp // 3, sp - pp // 3)
    else:
        return [3*pp] + balanced_ternary(n - 3*pp, pp, sp)


def brangelina(first, second):
    groups, in_group = [], False    
    for i in range(len(first)):        
        if first[i] in "aeiou":
            if not in_group:
                in_group = True
                groups.append(i)
        else:
            in_group = False
    if len(groups) == 1:
        first = first[:groups[0]]
    else:
        first = first[:groups[-2]]
    i = 0
    while second[i] not in "aeiou":
        i += 1
    return first + second[i:]  


def riffle(items, out = True):
    n, result = len(items) // 2, []
    for i in range(n):
        if out:
            result.append(items[i])
            result.append(items[n+i])
        else:
            result.append(items[n+i])
            result.append(items[i])
    return result


def count_carries(a, b):
    count, carry = 0, 0
    while a > 0 or b > 0:
        aa = a % 10
        bb = b % 10
        a = a // 10
        b = b // 10
        carry = (aa + bb + carry) > 9
        if carry:
            count += 1
    return count


def safe_squares_rooks(n, rooks):
    saferow, safecol = [1 for i in range(n)], [1 for i in range(n)]    
    for (row, col) in rooks:
        saferow[row] = safecol[col] = 0        
    return sum(saferow) * sum(safecol)


def safe_squares_bishops(n, bishops):
    count = 0
    for row in range(n):
        for col in range(n):
            if (row, col) not in bishops:              
                for (br, bc) in bishops:
                    if abs(br - row) == abs(bc - col):
                        break
                else:
                    count += 1
    return count


def rooks_with_friends(n, friends, enemies):
    safe = [[1 for col in range(n)] for row in range(n)]
    dirs = [(0,1), (1,0), (0,-1), (-1,0)]
    for (rx, ry) in enemies:
        for (dx, dy) in dirs:             
            safe[rx][ry], nx, ny = 0, rx + dx, ry + dy
            while 0 <= nx and nx < n and 0 <= ny and ny < n and \
                (nx, ny) not in friends and (nx, ny) not in enemies:
                safe[nx][ny] = 0
                nx, ny = nx + dx, ny + dy
    return sum((sum(row) for row in safe)) - len(friends)


def lattice_paths(x, y, tabu):
    count = [ [0 for yy in range(y+1)] for xx in range(x+1)]
    tabu.append((0, 0))
    count[0][0] = 1
    for xx in range(x + 1):
        for yy in range(y + 1):
            if (xx, yy) not in tabu:
                down = 0 if yy == 0 else count[xx][yy-1]
                left = 0 if xx == 0 else count[xx-1][yy]
                count[xx][yy] = down + left
    return count[x][y]


def count_squares(points):
    pts, count = set(points), 0
    xmax = max([x for (x, y) in points])
    ymax = max([y for (x, y) in points])    
    for (x, y) in points:
        for xd in range(0, xmax - x + 1):
            for yd in range(1, ymax - y + 1):
                c1 = (x + xd, y + yd) in pts
                c2 = (x + yd, y - xd) in pts
                c3 = (x + yd + xd, y - xd + yd) in pts
                if c1 and c2 and c3:
                    count += 1
    return count
    
  
def kempner(n):
    total, tally, i = Fraction(0), 0, 1
    while tally < n:
        if '9' not in str(i):
            total += Fraction(1, i)
            tally += 1
        i += 1    
    return total.limit_denominator(1000)
    

def is_permutation(items, n):
    seen = [False for i in range(n + 1)]
    for v in items:
        if v < 1 or v > n or seen[v]:
            return False
        seen[v] = True
    return True
    

def tribonacci(n, start = (1, 1, 1)):
    if n < 3:
        return start[n]
    vals = start    
    for i in range(n - 2):
        vals = (vals[1], vals[2], vals[0] + vals[1] + vals[2])
    return vals[2]
               

def ztalloc(pattern):    
    curr = 1
    for c in pattern[::-1]:
        if c == 'd':
            curr = 2 * curr
        else:
            if (curr - 1) % 3 != 0:
                return None
            else:
                curr = (curr - 1) // 3
                if curr % 2 == 0:
                    return None                
    return curr

def __collatz(n):
    result = ''
    while n > 1:
        if n % 2 == 0:
            result += 'd'
            n = n // 2
        else:
            result += 'u'
            n = 3 * n + 1
    return result


def count_distinct_sums_and_products(items):
    seen = set()
    for i in range(len(items)):
        e1 = items[i]
        for j in range(i, len(items)):
            e2 = items[j]
            seen.add(e1 + e2)
            seen.add(e1 * e2)    
    return len(seen)


def remove_after_kth(items, k = 1):
    result = []
    seen = dict()
    for e in items:
        s = seen.get(e, 0) + 1
        if s <= k:
            result.append(e)
        seen[e] = s
    return result


def is_cyclops(n):
    n, h = str(n), len(str(n)) // 2    
    return len(n) == 2*h+1 and n[h] == '0' and '0' not in n[:h] and '0' not in n[h+1:]    


def domino_cycle(tiles):
    prev = tiles[-1] if len(tiles) > 0 else None
    for tile in tiles:
        if prev[1] != tile[0]:
            return False
        prev = tile
    return True


def __dominated(p1, p2):
    return p1[0] < p2[0] and p1[1] < p2[1]

def count_maximal_layers(points):    
    points.sort(key = lambda x: x[0] + x[1])
    count = 0 
    while len(points) > 0:
        count += 1
        keep = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if __dominated(points[i], points[j]):
                    keep.append(points[i])
                    break
        if len(keep) == 0:
            return count
        else:
            points = keep

def __split_layer(layer):
    kick, keep = [], []
    for i in range(len(layer)):
        p1 = layer[i]
        for j in range(i + 1, len(layer)):
            p2 = layer[j]
            if __dominated(p1, p2):
                kick.append(p1)
                break
        else:
            keep.append(p1)
    return (kick, keep)     


def shortest_path(start, end):
    curr, seen, d = [start], set([start]), 1
    while len(seen) > 0:
        succ = []
        for e in curr:
            for x in [e + 1, e // 3, e * 2]:
                if x == end:
                    return d
                if x not in seen:
                    seen.add(x)
                    succ.append(x)
        curr = succ
        d += 1    


def taxi_zum_zum(moves):  
    pos, hed, dv = (0, 0), 0, [(0, 1), (-1, 0), (0, -1), (1, 0)]
    for m in moves:
        if m == 'R':
            hed = (hed - 1) % 4
        elif m == 'L':
            hed = (hed + 1) % 4
        else:
            pos = (pos[0] + dv[hed][0], pos[1] + dv[hed][1])
    return pos


def count_growlers(animals):    
    growl = 0
    for aa in [animals, [a[::-1] for a in reversed(animals)]]:
        balance = 0
        for a in aa:
            if balance > 0 and (a == 'cat' or a == 'dog'):
                growl += 1
            balance -= 1 if a == 'cat' or a == 'tac' else -1
    return growl


def tukeys_ninthers(items):
    n = len(items)
    while n > 1:
        pos = 0
        for i in range(0, n, 3):
            items[pos] = __med(items[i], items[i+1], items[i+2])
            pos += 1
        n = n // 3
    return items[0]


def bridge_score(suit, level, vul, dbl, made):
    mul = {'X':2, 'XX':4 }.get(dbl, 1)
    score, bonus = 0, 0
    
    # Add up the values of individual tricks.
    for trick in range(1, made + 1):
        # Raw points for this trick.
        if suit == 'clubs' or suit == 'diamonds':
            pts = 20
        elif suit == 'hearts' or suit == 'spades':
            pts = 30
        else:
            pts = 40 if trick == 1 else 30
        
        if trick <= level: # Part of contract
            score += mul * pts
        elif mul == 1: # Undoubled overtrick
            bonus += mul * pts
        elif mul == 2: # Doubled overtrick
            bonus += 200 if vul else 100
        else: # Redoubled overtrick
            bonus += 400 if vul else 200
    if score >= 100: # Game bonus
        bonus += 500 if vul else 300
    else: # Partscore bonus
        bonus += 50
    if level == 6: # Small slam bonus
        bonus += 750 if vul else 500
    if level == 7: # Grand slam bonus
        bonus += 1500 if vul else 1000
    score += bonus
    if mul == 2: # Insult bonus for making doubled contract
        score += 50
    elif mul == 4:
        score += 100
    return score


@lru_cache(maxsize = 10000)
def minimize_sum(digits, k):    
    if k < 2:
        return int(digits)
    elif k == len(digits):
        return sum(int(x) for x in digits)
    else:
        best = None
        for i in range(1, len(digits) - k + 2):
            curr = int(digits[:i]) + minimize_sum(digits[i:], k-1)
            if not best or curr < best:
                best = curr        
        return best


def collatzy_distance(start, end):
    level = 0
    curr = [start]
    seen = set(curr)
    while end not in seen:
        level += 1
        succ = []
        for e in curr:
            for f in [lambda x: 3*x + 1, lambda x: x // 2]:
                x = f(e)
                if x not in seen:
                    succ.append(x)
                    seen.add(x)        
        curr = succ        
    return level


def all_letters(words):
    lets = set(c for word in words for c in word )
    return "".join(list(sorted(list(lets))))


def double_trouble(items, n):
    step, pos = 1, 0
    while n > 0:
        n -= step
        if n < 1:
            return items[pos]
        pos += 1
        if pos >= len(items):
            step, pos = step * 2, 0


def van_eck(n):    
    seen, prev = dict(), 0    
    for i in range(1, n + 1):
        if prev not in seen:
            seen[prev] = i - 1
            prev = 0
        else:
            curr = seen[prev]                        
            seen[prev] = i - 1
            prev = i - 1 - curr             
    return prev


def suppressed_digit_sum(n):
    n, total, seen = str(n), 0, set()    
    if len(n) == 1:
        return 0
    for i in range(len(n)):
        curr = int(n[:i] + n[i+1:])
        if curr not in seen:
            total += curr
            seen.add(curr)            
    return total


def crag_score(dice):
    dice.sort()
    total = sum(dice)
    if total == 13:
        return 50 if dice[0] == dice[1] or dice[1] == dice[2] else 26
    if dice[0] == dice[2]:
        return 25
    if dice in ([1,2,3], [4,5,6], [1,3,5], [2,4,6]):
        return 20
    best = 0
    for f in range(1, 7):
        best = max(best, f * dice.count(f))
    return best

def __midnight(dice, remain, sofar):
    if len(remain) == 0:        
        one, four, total = False, False, 0
        for i in range(len(sofar)):
            for d in sofar[i]:
                pips = dice[d-1][i]
                if pips == 1:
                    one = True
                if pips == 4:
                    four = True
                total += pips        
        return total - 5 if one and four else 0
    else:
        best = 0
        for r in range(1, len(remain) + 1):
            for block in combinations(remain, r):
                newrem = [d for d in remain if d not in block]
                sofar.append(block)
                best = max(best, __midnight(dice, newrem, sofar))
                sofar.pop()
        return best
    
def midnight(dice):
    result = __midnight(dice, [1,2,3,4,5,6], [])    
    return result


def __forbidden(sofar, chars, n, tabu):
    if n == 0:
        yield sofar
    else:
        for c in chars:
            newfar = sofar + c
            for t in tabu:               
                if newfar.endswith(t):
                    break
            else:                
                yield from __forbidden(newfar, chars, n-1, tabu)                    

def forbidden_substrings(chars, n, tabu):
    tabu = [tabu] if isinstance(tabu, str) else tabu
    result = [word for word in __forbidden('', chars, n, tabu)]
    result.sort()    
    return result


def fibonacci_word(k):
    while k > __fibs[-1]:
        __fibs.append(__fibs[-1] + __fibs[-2])
    i = len(__fibs) - 1
    while k > 1:
        if k >= __fibs[i-1]:
            k = k - __fibs[i-1]
        i = i - 1
    return str(k)