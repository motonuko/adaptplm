import unittest

from adaptplm.mlm.train_utils.custom_train_loop import is_in_remainder_batches


class TestRemainderBatchDetection(unittest.TestCase):

    def test_no_remainder_batches(self):
        loader_length = 8
        accumulation_steps = 4
        for batch_idx in range(loader_length):
            self.assertFalse(
                is_in_remainder_batches(batch_idx, loader_length, accumulation_steps)
            )

    def test_with_remainder_batches(self):
        loader_length = 7
        accumulation_steps = 4
        for batch_idx in range(4):  # 0, 1, 2, 3
            self.assertFalse(
                is_in_remainder_batches(batch_idx, loader_length, accumulation_steps)
            )
        # remainder batch
        for batch_idx in range(4, loader_length):  # 4, 5, 6
            self.assertTrue(
                is_in_remainder_batches(batch_idx, loader_length, accumulation_steps)
            )

    def test_edge_case_single_batch(self):
        loader_length = 1
        accumulation_steps = 4
        self.assertTrue(
            is_in_remainder_batches(0, loader_length, accumulation_steps)
        )

    def test_edge_case_exact_division(self):
        loader_length = 12
        accumulation_steps = 3
        for batch_idx in range(loader_length):
            self.assertFalse(
                is_in_remainder_batches(batch_idx, loader_length, accumulation_steps)
            )


if __name__ == "__main__":
    unittest.main()
