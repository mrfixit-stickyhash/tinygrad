from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes



# SHA3-256 Constants
KECCAK_ROUNDS = 24
ROTATIONS = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14]
]
ROUND_CONSTANTS = [
    0x0000000000000001, 0x0000000000008082,
    0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088,
    0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B,
    0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080,
    0x0000000080000001, 0x8000000080008008
]

def rotl(x: int, n: int) -> int:
    """Rotate left: rotl(x, n)"""
    return ((x << n) | (x >> (64 - n))) & 0xFFFFFFFFFFFFFFFF

def keccak_f(state: list[int]) -> list[int]:
    """Keccak-f[1600] permutation."""
    for rnd in range(KECCAK_ROUNDS):
        # θ step
        C = [state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20] for x in range(5)]
        D = [C[(x - 1) % 5] ^ rotl(C[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                state[x + 5 * y] ^= D[y]
        
        # ρ and π steps
        B = [0] * 25
        for x in range(5):
            for y in range(5):
                B[y + 5 * ((2 * x + 3 * y) % 5)] = rotl(state[x + 5 * y], ROTATIONS[x][y])
        
        # χ step
        for y in range(5):
            for x in range(5):
                state[x + 5 * y] = B[x + 5 * y] ^ ((~B[((x + 1) % 5) + 5 * y]) & B[((x + 2) % 5) + 5 * y])
        
        # ι step
        state[0] ^= ROUND_CONSTANTS[rnd]
    
    return state

def sha3_256(data: bytes) -> bytes:
    """Compute SHA3-256 hash of the input data using Tinygrad."""
    # Initialize state (5x5 matrix of 64-bit words)
    state = [0] * 25
    
    # Padding (Multi-rate padding)
    rate_bytes = 1088 // 8  # 1088 bits rate for SHA3-256
    block_size = rate_bytes
    padded_data = bytearray(data)
    padded_data.append(0x06)  # Multi-rate padding
    while (len(padded_data) + 8) % 64 != 0:
        padded_data.append(0x00)
    padded_data += (len(data) * 8).to_bytes(8, 'big')  # 64-bit big-endian length
    
    # Absorbing phase
    for block_start in range(0, len(padded_data), block_size):
        block = padded_data[block_start:block_start + block_size]
        # Convert block to 64-bit little-endian words
        block_words = [int.from_bytes(block[i:i+8], 'little') for i in range(0, block_size, 8)]
        # XOR block into state
        for i in range(len(block_words)):
            state[i] ^= block_words[i]
        # Permute state
        state = keccak_f(state)
    
    # Squeezing phase
    hash_output = []
    while len(hash_output) < 32:  # 256 bits = 32 bytes
        # Convert state to bytes
        state_bytes = b''.join([word.to_bytes(8, 'little') for word in state])
        hash_output += list(state_bytes[:rate_bytes])
        if len(hash_output) >= 32:
            break
        state = keccak_f(state)
    
    return bytes(hash_output[:32])

def parallel_sha3_256(data_list: list[bytes]) -> list[bytes]:
    """
    Compute SHA3-256 hashes for a list of input data in parallel using Tinygrad.
    
    Args:
        data_list (list of bytes): List of data to hash.
    
    Returns:
        list of bytes: Corresponding SHA3-256 hashes.
    """
    # Convert data to Tensors
    tensors = [Tensor(list(d), dtype=dtypes.uint8) for d in data_list]
    
    # Perform hashing in parallel
    hashes = []
    for tensor in tensors:
        data_bytes = bytes(tensor.tolist())
        hash_bytes = sha3_256(data_bytes)
        hashes.append(hash_bytes)
    
    return hashes
