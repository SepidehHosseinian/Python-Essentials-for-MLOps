def test_dictionaries():
    result=dict(key="value",firstname="sepideh",lastname="hosseinian")
    expected=dict(key="value",firtname="sepideh",lastname="hosseinian")
    assert result == expected
