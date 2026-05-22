import unittest
from src.models.unseen import find_nearest_pattern


class TestUnseenManagement(unittest.TestCase):
    
    def test_find_nearest_pattern(self):
        state_dictionary = {"abc", "xyz", "mno"}
        unseen_pattern = "adc"
        
        nearest = find_nearest_pattern(unseen_pattern, state_dictionary)
        
        self.assertEqual(nearest, "abc", "En yakın pattern 'abc' olarak bulunmali.")


if __name__ == "__main__":
    unittest.main()
