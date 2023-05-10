import pytest
@pytest.mark.parametrize('value',['yes','Y',''])
def test_is_Y(value):
    result=value
    assert result == value