import numpy as np

from tracking_v3.frame import Frame

def test_frame():
    # identity
    f = Frame([0, 0, 0], [1, 0, 0])
    assert np.allclose(f.to_local_vec([1, 0, 0]), [1, 0, 0], atol=.01)

    # -45 deg az
    f = Frame([0, 0, 0], [1, 1, 0])
    assert np.allclose(f.to_local_vec([1, 0, 0]), [.7, -.7, 0], atol=.01)

    # -90 deg az
    f = Frame([0, 0, 0], [0, 1, 0])
    assert np.allclose(f.to_local_vec([1, 0, 0]), [0, -1, 0], atol=.01)

    # identity
    f = Frame([0, 0, 0], [1, 1, 0])
    assert np.allclose(f.to_local_vec([1, 1, 0]), [1.41, 0, 0], atol=.01)

    # 45 deg down
    f = Frame([0, 0, 0], [1, 1, 1.4142])
    assert np.allclose(f.to_local_vec([1, 1, 0]), [1, 0, -1], atol=.01)


def test_transform():
    # f1: 30 deg left; f2: 60 deg left; same origin
    f1 = Frame([1, 0, 0], [np.sqrt(3), 1, 0])
    f2 = Frame([1, 0, 0], [1, np.sqrt(3), 0])
    t1 = f1.transform(f2)
    assert np.allclose(t1.transform_vec([1, 0, 0]), [.866, -.5, 0], atol=.01)

    # f1: 45 deg left; f2; forward; different origins
    f1 = Frame([1, 0, 0], [1, 1, 0])
    f2 = Frame([0, 0, 0], [1, 0, 0])
    t2 = f1.transform(f2)
    assert np.allclose(t2.transform_vec([1, 0, 0]), [1 + np.sqrt(2)/2, np.sqrt(2)/2, 0], atol=.01)

