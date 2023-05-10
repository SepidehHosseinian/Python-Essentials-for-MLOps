class TestStrToInt:
    def setup_method(self):
        print("setup test method")
    def teardown_method(self):
        print("teardown test method")
    def setup_class(self):
        print("setup test class")
    def teardown_class(self):
        print("teardown test class") 
    def test_round_down(self):
        result= round(float("1.99"))
        assert result ==2
    def test_round_down_lesser_than_half(self):
        result= round(float("1.2"))
        assert result ==2