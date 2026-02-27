from distill_safe.config import load_pipeline_config

def test_load_smoke_config():
    cfg = load_pipeline_config("configs/pipelines/smoke_test.yaml")
    assert cfg.run.name == "smoke_test"
    assert cfg.teacher is not None
