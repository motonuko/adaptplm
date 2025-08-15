from functools import lru_cache

from Bio.SeqUtils.CheckSum import crc64


# NOTE: I tried to implement this,but since the values didn’t match those from UniProt, Biopython is used.
@lru_cache
def calculate_crc64(sequence: str, remove_prefix: bool = True):
    checksum = crc64(sequence)
    if remove_prefix:
        return checksum[4:]  # remove first 'CRC-' part
    return checksum
