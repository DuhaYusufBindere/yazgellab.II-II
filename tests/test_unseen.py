import unittest
from src.models.unseen import find_nearest_pattern


class TestUnseenManagement(unittest.TestCase):
    
    def test_find_nearest_pattern(self):
        state_dictionary = {"abc", "xyz", "mno"}
        unseen_pattern = "adc"
        
        result = find_nearest_pattern(unseen_pattern, state_dictionary)
        
        self.assertEqual(result["mapped_to"], "abc", "En yakın pattern 'abc' olarak bulunmali.")
        self.assertEqual(result["nearest_distance"], 1, "Mesafe farki 1 olmali.")


if __name__ == "__main__":
    unittest.main()
