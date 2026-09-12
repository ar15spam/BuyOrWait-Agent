PYTHON = venv/bin/python

.PHONY: test check

test:
	$(PYTHON) -m tests.test

check:
	$(PYTHON) -m py_compile code/build_context.py tests/test.py