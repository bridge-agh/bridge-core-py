from bridge_core_py.player import PlayerDirection


def test_next():
    assert PlayerDirection.NORTH.next() == PlayerDirection.EAST
    assert PlayerDirection.EAST.next() == PlayerDirection.SOUTH
    assert PlayerDirection.SOUTH.next() == PlayerDirection.WEST
    assert PlayerDirection.WEST.next() == PlayerDirection.NORTH

def test_prev():
    assert PlayerDirection.NORTH.prev() == PlayerDirection.WEST
    assert PlayerDirection.WEST.prev() == PlayerDirection.SOUTH
    assert PlayerDirection.SOUTH.prev() == PlayerDirection.EAST
    assert PlayerDirection.EAST.prev() == PlayerDirection.NORTH
