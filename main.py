import math
import argparse
import time
import numpy as np
import gmpy2
from gmpy2 import mul
from sympy import primerange
from sympy.ntheory.residue_ntheory import nthroot_mod
from sympy.ntheory import factorint
import numba


parser = argparse.ArgumentParser()
parser.add_argument("p", type=int,
                    help="prime number to form N")
parser.add_argument("q", type=int,
                    help="prime number to form N")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")

args = parser.parse_args()


@numba.jit(parallel=True, nopython=True)
def gf2elim(M):

    m,n = M.shape

    i=0
    j=0

    while i < m and j < n:
        k = np.argmax(M[i:, j]) + i
        temp = np.copy(M[k])
        M[k] = M[i]
        M[i] = temp

        aijn = M[i, j:]

        col = np.copy(M[:, j])

        col[i] = 0

        flip = np.outer(col, aijn)

        M[:, j:] = M[:, j:] ^ flip

        i += 1
        j += 1

    return M


def get_L(N):
    return math.exp(math.sqrt(math.log(N) * math.log(math.log(N))))

def get_B(N):
    L= get_L(N)
    return int(L**(1/math.sqrt(2)))


def get_a(N):
    return int(math.sqrt(N))+1


def F(T):
    return T**2 - N


def get_indices(lst, el):
    return [i for i in range(len(lst)) if lst[i] == el]


def prime_range(B):
    prime_powers = []
    for p in primerange(2, B):
        pe = p
        while pe < B:
            prime_powers.append(pe)
            pe *= p
    prime_powers = sorted(prime_powers)
    factor_list = [factorint(i, multiple=True)[0] for i in prime_powers]
    return prime_powers, factor_list, list(primerange(2, B))


def sieve(b, initial_sieve):
    result = []
    N_req = nthroot_mod(N, 2, b, True)
    if len(N_req) == 0:
        return None
    for j in initial_sieve:
        prob_t = j % b
        if prob_t in N_req:
            N_req.remove(prob_t)
            result.append(initial_sieve.index(j))
            if b == 2 or len(N_req) == 0:
                return result
    return result


def count_sieve(prime_powers, original_sieve, changed_sieve, factor_list):
    B_nums = []
    B_nums_factors = {}
    len_sieve = len(original_sieve)
    #for count in range(2):
    for i in range(len(prime_powers)):
        idx = sieve(prime_powers[i], original_sieve)
        if idx is None:
            continue
        for j in idx:
            new_idx = j
            while new_idx < len_sieve:
                if changed_sieve[new_idx] % factor_list[i] == 0:
                    changed_sieve[new_idx] //= factor_list[i]
                try:
                    B_nums_factors[new_idx].append(factor_list[i])
                except KeyError:
                    B_nums_factors[new_idx] = []
                    B_nums_factors[new_idx].append(factor_list[i])
                if changed_sieve[new_idx] == 1:
                    changed_sieve[new_idx] = 0
                    B_nums.append(new_idx)
                new_idx += prime_powers[i]
    return sorted(B_nums), B_nums_factors


def vectorize(primerange, B_nums, B_nums_factors):
    factor_list = {}
    result = []
    for i in primerange:
        factor_list[i] = []
        for j in B_nums:
            factor_list[i].append(B_nums_factors[j].count(i) % 2)
    for i in factor_list:
        result.append(factor_list[i])
    return np.array(result, dtype=np.uint8)


def factors_multiplication(index_list, b_list, factor_list):
    result = []
    for index in index_list:
        factors = factor_list[b_list[index]]
        for i in factors:
            result.append(i)
    factors_count = []
    final_number = 1
    tmp = sorted(list(set(result)))
    for i in tmp:
        tmp2 = 1 if result.count(i) - 2 == 0 else result.count(i) // 2
        if tmp2 < 0:
            return -1
        factors_count.append(tmp2)
    for i in range(len(tmp)):
        final_number *= tmp[i] ** factors_count[i]
    return final_number


def factorization(matrix):
    indexes = []
    for i in range(len(matrix[0])):
        if i > len(matrix):
            matrix = np.concatenate((matrix, [np.zeros(len(matrix[0]))]))
        if matrix[i][i] == 0:
            matrix[i][i] = 1
            indexes = get_indices(matrix[:, i], 1)
            if len(indexes) <= 1:
                continue
            result = 1
            for j in indexes:
                result *= b_num[j]
            bro = factors_multiplication(indexes, B_nums, B_nums_factors)
            if bro == -1:
                continue
            a = gmpy2.gcd(N, result - bro)
            if a != 1 and a != N:
                first_factor = a
                second_factor = N // a
                break
    return first_factor, second_factor

if __name__ == '__main__':
    #инициализация
    N = mul(args.p, args.q)
    B = get_B(N)

    sieve_q = gmpy2.mpz(round(math.floor(get_L(N))))
    a = gmpy2.mpz(get_a(N))
    original_sieve = list(range(a, a + sieve_q))
    changed_sieve = []
    #расчет f
    for i in range(sieve_q):
        changed_sieve.append(F(original_sieve[i]))
    changed_sieve = np.array(changed_sieve)

    prime_powers, factor_list, primerange = prime_range(B)
    #просеивание
    B_nums = []
    B_nums_factors = {}
    len_in = len(original_sieve)
    B_nums, B_nums_factors = count_sieve(prime_powers, original_sieve, changed_sieve, factor_list)

    B_nums = sorted(B_nums)

    new_b_factor = {}
    for i in B_nums:
        if i in B_nums_factors:
            new_b_factor[i] = B_nums_factors[i]

    vectors_list = []
    b_num = []
    for i in B_nums:
        b_num.append(original_sieve[i])

    matrix = vectorize(primerange, B_nums, new_b_factor)
    matrix = gf2elim(matrix)

    first_factor, second_factor = factorization(matrix)
    print("Factorization result: ", first_factor, second_factor)

