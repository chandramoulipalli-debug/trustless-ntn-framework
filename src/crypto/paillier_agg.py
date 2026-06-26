"""
Privacy-preserving trust aggregation using Paillier homomorphic encryption.
Implements Eq. 1 and Eq. 2 from the paper:
    c_ij(t) = Enc(f_ij(t))
    C_j(t)  = Σ_k  c_kj(t)   (homomorphic addition over ciphertexts)

The smart contract operates only on the aggregate ciphertext C_j(t).
A threshold committee decrypts the result — individual contributions stay hidden.

Also measures encryption latency and ciphertext overhead for Exp 3.
"""

import time
import math
import phe  # python-paillier


class PaillierAggregator:
    """
    Simulates the privacy-preserving trust aggregation layer.
    Each contributor encrypts their feedback; aggregation is done on ciphertext.
    """

    def __init__(self, key_bits: int = 1024):
        self.public_key, self.private_key = phe.paillier.generate_paillier_keypair(
            n_length=key_bits
        )
        self.key_bits = key_bits

    def encrypt_feedback(self, feedback: float) -> phe.EncryptedNumber:
        """Encrypt a feedback value in [0,1] — simulates Eq. 1."""
        # Scale to integer range [0, 1000] for Paillier (works on integers)
        scaled = round(feedback * 1000)
        return self.public_key.encrypt(scaled)

    def aggregate(self, ciphertexts: list[phe.EncryptedNumber]) -> phe.EncryptedNumber:
        """
        Homomorphic addition of all ciphertexts — simulates Eq. 2.
        Smart contract does this WITHOUT decrypting individual values.
        """
        result = ciphertexts[0]
        for ct in ciphertexts[1:]:
            result = result + ct
        return result

    def decrypt_aggregate(self, aggregate_ct: phe.EncryptedNumber,
                          n_contributors: int) -> float:
        """
        Threshold committee decrypts and averages the aggregate.
        Returns the averaged trust feedback in [0,1].
        """
        total_scaled = self.private_key.decrypt(aggregate_ct)
        return (total_scaled / n_contributors) / 1000.0

    def benchmark(self, n_contributors: int) -> dict:
        """
        Measure latency and size for n_contributors.
        Returns timing metrics used in Experiment 3 (cryptographic overhead).
        """
        feedbacks = [0.8] * n_contributors  # fixed value for benchmarking

        # Measure encryption time
        t0 = time.perf_counter()
        ciphertexts = [self.encrypt_feedback(f) for f in feedbacks]
        enc_time_ms = (time.perf_counter() - t0) * 1000

        # Measure aggregation time
        t1 = time.perf_counter()
        agg_ct = self.aggregate(ciphertexts)
        agg_time_ms = (time.perf_counter() - t1) * 1000

        # Measure decryption time
        t2 = time.perf_counter()
        result = self.decrypt_aggregate(agg_ct, n_contributors)
        dec_time_ms = (time.perf_counter() - t2) * 1000

        # Ciphertext size estimate (bytes)
        ct_size_bytes = self.key_bits // 4  # each ciphertext ~ key_bits/4 bytes
        total_size_kb = (ct_size_bytes * n_contributors) / 1024

        return {
            "n_contributors": n_contributors,
            "enc_time_ms": enc_time_ms,
            "agg_time_ms": agg_time_ms,
            "dec_time_ms": dec_time_ms,
            "total_time_ms": enc_time_ms + agg_time_ms + dec_time_ms,
            "ciphertext_size_kb": total_size_kb,
            "result": result,
        }
