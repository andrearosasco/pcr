from unittest import TestCase
from train import main


class Test(TestCase):
    def test_main(self):
        main(test=True)
