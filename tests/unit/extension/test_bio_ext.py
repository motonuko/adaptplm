import unittest

from enzrxnpred2.extension.bio_ext import calculate_crc64


# test data license
# test_data: https://www.uniprot.org/uniprotkb/P01308/entry#sequences
# The UniProt Consortium
# UniProt: the Universal Protein Knowledgebase in 2023
# Nucleic Acids Res. 51:D523–D531 (2023)
# Provided by Creative Commons Attribution 4.0 International (CC BY 4.0) License
# https://creativecommons.org/licenses/by/4.0/
class TestCRC64Checksum(unittest.TestCase):
    def test_crc64_checksum(self):
        sequence = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
        expected_checksum = 'C2C3B23B85E520E5'
        expected_checksum2 = 'CRC-C2C3B23B85E520E5'
        calculated_checksum = calculate_crc64(sequence, True)
        self.assertEqual(calculated_checksum, expected_checksum,
                         "The calculated CRC64 checksum does not match the expected value.")
        calculated_checksum = calculate_crc64(sequence, False)
        self.assertEqual(calculated_checksum, expected_checksum2,
                         "The calculated CRC64 checksum does not match the expected value.")


if __name__ == '__main__':
    unittest.main()
