"""Monitor / drift-detector tests.

Split by torch requirement:

* Torch-free tests (this package's top level) run on the default
  pytest invocation and enforce the "importing gauntlet.monitor.schema
  stays clean on a torch-free install" contract.
* Tests marked ``@pytest.mark.monitor`` require the ``[monitor]``
  extra (torch + PIL) and run in the ``monitor-tests`` CI job.
"""
