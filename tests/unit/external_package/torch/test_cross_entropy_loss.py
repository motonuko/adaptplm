import unittest

import numpy as np
import torch
import torch.nn as nn


# pytorch 2.1.0 fails this test.
class TestCrossEntropyIgnoreIndex(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([
            [[2.0, 0.5, 0.3], [1.0, 2.0, 0.1], [0.2, 0.1, 2.0]],
            [[1.5, 0.5, 1.0], [0.0, 0.0, 0.0], [0.1, 0.2, 0.1]]
        ])
        self.targets = torch.tensor([
            [0, 1, -100],
            [2, -100, 1]
        ])

    def test_ignore_index_cpu_mps(self):  # should be success
        loss_fn = nn.CrossEntropyLoss()

        logits_cpu = self.logits.clone()
        targets_cpu = self.targets.clone()
        loss_cpu = loss_fn(logits_cpu.view(-1, 3), targets_cpu.view(-1))
        if torch.backends.mps.is_available():
            logits_mps = self.logits.to("mps")
            targets_mps = self.targets.to("mps")
            loss_mps = loss_fn(logits_mps.view(-1, 3), targets_mps.view(-1))
            np.testing.assert_almost_equal(loss_cpu.item(), loss_mps.item(), decimal=6)

    # def test_ignore_index_cpu_mps3(self):  # should be failed
    #     if torch.backends.mps.is_available():
    #         logits_mps = self.logits.to("mps")
    #         targets_mps = self.targets.to("mps")
    #
    #         loss_fn1 = nn.CrossEntropyLoss()
    #         loss_mps1 = loss_fn1(logits_mps.view(-1, 3), targets_mps.view(-1))
    #
    #         loss_fn2 = nn.CrossEntropyLoss(reduction='sum')
    #         loss_mps2 = loss_fn2(logits_mps.view(-1, 3), targets_mps.view(-1)) / len(self.targets.view(-1))
    #
    #         np.testing.assert_almost_equal(loss_mps1.item(), loss_mps2.item(), decimal=6)

    def test_ignore_index_cpu_mps2(self):  # success
        loss_fn_cpu = nn.CrossEntropyLoss()
        logits_cpu = self.logits.clone()
        targets_cpu = self.targets.clone()
        loss_cpu = loss_fn_cpu(logits_cpu.view(-1, 3), targets_cpu.view(-1))

        loss_fn_mps = nn.CrossEntropyLoss(reduction='sum')
        if torch.backends.mps.is_available():
            logits_mps = self.logits.to("mps")
            targets_mps = self.targets.to("mps")
            num_used_tokens = (self.targets != -100).sum().item()
            loss_mps = loss_fn_mps(logits_mps.view(-1, 3), targets_mps.view(-1)) / num_used_tokens
            np.testing.assert_almost_equal(loss_cpu.item(), loss_mps.item(), decimal=6)


if __name__ == "__main__":
    unittest.main()
