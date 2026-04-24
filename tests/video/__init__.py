"""Polish "rollout MP4 video recording" tests.

Every test in this package is marked ``@pytest.mark.video`` and runs
under the dedicated ``video-tests`` CI job (see
``.github/workflows/ci.yml``). The default torch-/extras-free job
deselects the marker via ``-m "not video"``; the unmarked
``tests/test_runner_video_backcompat.py`` lives at the repo top level
so that the byte-identical opt-out path is exercised in EVERY job.
"""
