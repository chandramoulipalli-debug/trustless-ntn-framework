/*
 * paillier_hw.c — Paillier HE benchmark (C/GMP implementation)
 *
 * Compile:
 *   Linux/macOS:  gcc -O2 -o paillier_hw paillier_hw.c -lgmp
 *   Windows/MSYS: gcc -O2 -o paillier_hw.exe paillier_hw.c -lgmp
 *
 * Usage:
 *   ./paillier_hw [key_bits] [n_contributors]
 *   ./paillier_hw 2048 100
 *
 * Reference: Paillier, P. (1999). EUROCRYPT 1999, LNCS 1592, pp. 223-238.
 * 3GPP context: Provides C-native timing baseline beyond the Python n<=75 limit.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <gmp.h>

/* ── types ──────────────────────────────────────────────────────────────── */

typedef struct {
    mpz_t n;      /* RSA modulus */
    mpz_t n2;     /* n^2 */
    mpz_t g;      /* generator (= n+1 in simplified Paillier) */
    mpz_t lam;    /* lcm(p-1, q-1) */
    mpz_t mu;     /* L(g^lambda mod n^2)^{-1} mod n */
} PaillierKey;

/* ── helpers ────────────────────────────────────────────────────────────── */

static void L_func(mpz_t result, const mpz_t u, const mpz_t n)
{
    mpz_sub_ui(result, u, 1);
    mpz_divexact(result, result, n);
}

static void lcm_mpz(mpz_t result, const mpz_t a, const mpz_t b)
{
    mpz_t g;
    mpz_init(g);
    mpz_gcd(g, a, b);
    mpz_divexact(result, a, g);
    mpz_mul(result, result, b);
    mpz_abs(result, result);
    mpz_clear(g);
}

/* ── key generation ─────────────────────────────────────────────────────── */

void keygen(PaillierKey *key, int key_bits, gmp_randstate_t rstate)
{
    int half = key_bits / 2;
    mpz_t p, q, p1, q1, tmp;
    mpz_inits(p, q, p1, q1, tmp, NULL);
    mpz_inits(key->n, key->n2, key->g, key->lam, key->mu, NULL);

    /* Generate two safe primes of key_bits/2 each */
    do {
        mpz_urandomb(p, rstate, half);
        mpz_setbit(p, half - 1);     /* ensure high bit set */
        mpz_nextprime(p, p);

        mpz_urandomb(q, rstate, half);
        mpz_setbit(q, half - 1);
        mpz_nextprime(q, q);
    } while (mpz_cmp(p, q) == 0);

    mpz_mul(key->n, p, q);
    mpz_mul(key->n2, key->n, key->n);

    /* g = n + 1 (standard simplification) */
    mpz_add_ui(key->g, key->n, 1);

    /* lambda = lcm(p-1, q-1) */
    mpz_sub_ui(p1, p, 1);
    mpz_sub_ui(q1, q, 1);
    lcm_mpz(key->lam, p1, q1);

    /* mu = L(g^lambda mod n^2)^{-1} mod n */
    mpz_powm(tmp, key->g, key->lam, key->n2);
    L_func(tmp, tmp, key->n);
    mpz_invert(key->mu, tmp, key->n);

    mpz_clears(p, q, p1, q1, tmp, NULL);
}

/* ── encrypt ────────────────────────────────────────────────────────────── */

void encrypt(mpz_t ct, const PaillierKey *key,
             long plaintext, gmp_randstate_t rstate)
{
    mpz_t m, r, gm, rn;
    mpz_inits(m, r, gm, rn, NULL);

    mpz_set_si(m, plaintext);

    /* r: random in (0, n), gcd(r,n)=1 */
    do {
        mpz_urandomm(r, rstate, key->n);
    } while (mpz_sgn(r) == 0);

    /* ct = g^m * r^n mod n^2 */
    mpz_powm(gm, key->g, m, key->n2);
    mpz_powm(rn, r, key->n, key->n2);
    mpz_mul(ct, gm, rn);
    mpz_mod(ct, ct, key->n2);

    mpz_clears(m, r, gm, rn, NULL);
}

/* ── decrypt ────────────────────────────────────────────────────────────── */

long decrypt(const PaillierKey *key, const mpz_t ct)
{
    mpz_t x, lx;
    mpz_inits(x, lx, NULL);

    mpz_powm(x, ct, key->lam, key->n2);
    L_func(lx, x, key->n);
    mpz_mul(lx, lx, key->mu);
    mpz_mod(lx, lx, key->n);

    long result = mpz_get_si(lx);
    mpz_clears(x, lx, NULL);
    return result;
}

/* ── homomorphic add ─────────────────────────────────────────────────────── */

void he_add(mpz_t out, const mpz_t c1, const mpz_t c2, const PaillierKey *key)
{
    mpz_mul(out, c1, c2);
    mpz_mod(out, out, key->n2);
}

/* ── benchmark ──────────────────────────────────────────────────────────── */

static double wall_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void run_benchmark(int key_bits, int n_contributors)
{
    gmp_randstate_t rstate;
    gmp_randinit_default(rstate);
    gmp_randseed_ui(rstate, 42);

    printf("[PaillierHW-C] %d-bit key, n=%d contributors\n",
           key_bits, n_contributors);

    /* Key generation */
    double t0 = wall_sec();
    PaillierKey key;
    keygen(&key, key_bits, rstate);
    double t_kg = (wall_sec() - t0) * 1000;
    printf("  Keygen:  %.1f ms\n", t_kg);

    /* Encrypt */
    mpz_t *cts = malloc(n_contributors * sizeof(mpz_t));
    for (int i = 0; i < n_contributors; i++) mpz_init(cts[i]);

    double t1 = wall_sec();
    for (int i = 0; i < n_contributors; i++)
        encrypt(cts[i], &key, 800, rstate);   /* 0.8 * 1000 = 800 */
    double t_enc = (wall_sec() - t1) * 1000;
    printf("  Encrypt: %.2f ms  (%.3f ms/enc)\n", t_enc, t_enc / n_contributors);

    /* Homomorphic aggregate */
    mpz_t agg;
    mpz_init_set(agg, cts[0]);
    double t2 = wall_sec();
    for (int i = 1; i < n_contributors; i++)
        he_add(agg, agg, cts[i], &key);
    double t_agg = (wall_sec() - t2) * 1000;
    printf("  Agg:     %.2f ms\n", t_agg);

    /* Decrypt */
    double t3 = wall_sec();
    long   total = decrypt(&key, agg);
    double t_dec = (wall_sec() - t3) * 1000;
    double result = (double)total / n_contributors / 1000.0;
    printf("  Decrypt: %.2f ms  (result=%.4f, expected=0.8000)\n", t_dec, result);

    double t_total = t_enc + t_agg + t_dec;
    double budget  = t_total / 60000.0 * 100.0;   /* 60 s budget */
    printf("  Total:   %.2f ms  |  Budget: %.2f%%  |  Feasible: %s\n",
           t_total, budget, budget < 10.0 ? "YES" : (budget < 20.0 ? "MARGINAL" : "NO"));

    /* Cleanup */
    for (int i = 0; i < n_contributors; i++) mpz_clear(cts[i]);
    free(cts);
    mpz_clear(agg);
    mpz_clears(key.n, key.n2, key.g, key.lam, key.mu, NULL);
    gmp_randclear(rstate);
}

/* ── main ────────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
    int key_bits      = (argc > 1) ? atoi(argv[1]) : 2048;
    int n_contributors = (argc > 2) ? atoi(argv[2]) : 100;

    run_benchmark(key_bits, n_contributors);

    /* Standard sweep for paper Table */
    if (argc == 1) {
        int ns[] = {5, 10, 25, 50, 75, 100, 200, 500};
        int nsz  = sizeof(ns) / sizeof(ns[0]);
        for (int ki = 0; ki < 2; ki++) {
            int kb = (ki == 0) ? 2048 : 4096;
            printf("\n=== %d-bit sweep ===\n", kb);
            for (int ni = 0; ni < nsz; ni++)
                run_benchmark(kb, ns[ni]);
        }
    }
    return 0;
}
