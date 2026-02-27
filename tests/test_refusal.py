from distill_safe.safety.refusal import looks_like_refusal

def test_looks_like_refusal():
    assert looks_like_refusal("I can't help with that.")
    assert looks_like_refusal("I cannot provide instructions.")
    assert not looks_like_refusal("Sure, here's an answer.")
