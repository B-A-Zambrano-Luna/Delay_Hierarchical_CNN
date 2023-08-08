from fractions import Fraction
from Q_p import norm_p


class char_function(object):
    """ Define the class characteristic function of a ball"""

    def __init__(self, a, r, p):
        self.center = a
        self.radio = r
        self.prime = p
        self.constant = 1

    def get_center(self):
        return self.center

    def get_radio(self):
        return self.radio

    def get_constant(self):
        return self.constant

    def __call__(self, x):
        p = self.prime
        a = self.center
        r = self.radio
        c = self.constant
        if c == 0:
            return 0
        else:
            if norm_p(x-a, p) <= p**r:
                return c
            else:
                return 0

    def __mul__(self, other):
        if type(other) == float or type(other) == int:
            p = self.prime
            a = self.center
            r = self.radio
            C = char_function(a, r, p)
            C.constant = self.constant*other
            return C
        elif type(other) == char_function:
            p = self.prime
            a = self.center
            r = self.radio
            b = other.center
            l = other.radio
            c = self.constant*other.constant
            if l <= r and norm_p(a-b, p) <= p**r:
                C = char_function(b, l, p)
                C.constant = c
                return C
            elif r <= l and norm_p(a-b, p) <= p**l:
                C = char_function(a, r, p)
                C.constant = c
                return C
            else:
                C = char_function(b, l, p)
                C.constant = 0
                return C

    def __rmul__(self, other):
        if type(other) == float or type(other) == int:
            p = self.prime
            a = self.center
            r = self.radio
            C = char_function(a, r, p)
            C.constant = self.constant*other
            return C
        elif type(other) == char_function:
            p = self.prime
            a = self.center
            r = self.radio
            b = other.center
            l = other.radio
            c = self.constant*other.constant
            if l <= r and norm_p(a-b, p) <= p**r:
                C = char_function(b, l, p)
                C.constant = c
                return C
            elif r <= l and norm_p(a-b, p) <= p**l:
                C = char_function(a, r, p)
                C.constant = c
                return C
            else:
                C = char_function(b, l, p)
                C.constant = 0
                return C

    def __add__(self, other):
        p = self.prime
        a = self.center
        r = self.radio

        if type(other) == char_function:

            b = other.center
            l = other.radio
            if b == a and l == r:
                C = char_function(a, r, p)
                C.constant = self.constant+other.constant
                return C
            else:
                C = test_function()
                C.adds = [self, other]
                return C

    def integral_all(self):
        p = self.prime
        r = self.radio
        a = self.constant
        if a == 0:
            return 0
        else:
            return a*(p**r)

    def id(self, other):
        p = self.prime
        a = self.center
        r = self.radio

        if type(other) != char_function:
            return False
        else:
            q = other.prime
            b = other.center
            l = other.radio

            if q == p and b == a and l == r:
                return True
            else:
                return False

    def __eq__(self, other):
        p = self.prime
        a = self.center
        r = self.radio
        c = self.constant
        if type(other) != char_function:
            return False
        else:
            q = other.prime
            b = other.center
            l = other.radio
            c1 = other.constant
            if q == p and b == a and l == r and c == c1:
                return True
            else:
                return False

    def __str__(self):
        p = self.prime
        a = self.center
        r = self.radio
        c = self.constant
        return str(c)+"*B_"+str(r)+"("+str(a)+","+str(p)+")"


class test_function(object):
    """ Define the class of test functions"""

    def __init__(self, adds=[]):
        self.adds = adds

    def __call__(self, x):
        L = self.adds
        ans = 0
        for f in L:
            ans += f(x)
        return ans
    # def add(self):
    #    return self.adds

    def __mul__(self, other):
        L1 = self.adds
        ans = test_function()
        L3 = ans.adds
        if other == 0:
            return ans
        elif type(other) == float or type(other) == int:
            for f in L1:
                L3.append(f*other)

        elif type(other) == test_function:
            L2 = other.adds
            for f in L1:
                for g in L2:
                    L3.append(f*g)
        elif type(other) == char_function:
            for f in L1:
                L3.append(other*f)
        return ans

    def __rmul__(self, other):
        L1 = self.adds
        ans = test_function()
        L3 = ans.adds
        if other == 0:
            return ans
        elif type(other) == float or type(other) == int:
            for f in L1:
                L3.append(f*other)

        elif type(other) == test_function:
            L2 = other.adds
            for f in L1:
                for g in L2:
                    L3.append(f*g)
        elif type(other) == char_function:
            for f in L1:
                L3.append(other*f)
        return ans

    def __add__(self, other):
        L1 = self.adds
        ans = test_function()
        L = []
        if type(other) == test_function:
            L2 = other.adds
            L3 = L2.copy()
            for f in L1:
                for g in L2:
                    if f == g:
                        L.append(f+g)
                        L3.remove(f)
                    else:
                        L.append(f)
            ans.adds = L+L3
        elif type(other) == char_function:
            L3 = L1.copy()
            if other in L1:
                L3.remove(other)
                L3.append(other*2)
            else:
                L3.append(other)

            ans.adds = L3
        return ans

    def integral_all(self):
        adds = self.adds
        if adds == []:
            return 0
        else:
            ans = 0
            for f in adds:
                ans += f.integral_all()
            return ans

    def __eq__(self, other):
        if type(other) != test_function:
            return False
        else:
            L = self.adds
            L1 = other.adds

            for f in L:
                for g in L1:
                    if f != g:
                        return False
            return True

    def __str__(self):
        L = self.adds
        if L == []:
            return "0"
        elif len(L) == 1:
            return L[0].__str__()
        else:
            ans = ""
            for f in L:
                ans += f.__str__()+" + "
            return ans[:-3]
