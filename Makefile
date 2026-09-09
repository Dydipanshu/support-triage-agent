.PHONY: setup data golden eval agreement test demo clean-cache

setup:
	uv venv .venv --python 3.12 && . .venv/bin/activate && uv pip install -e ".[dev]" scipy

data:
	python -m src.support_agent.prepare --brand Delta

golden:
	python -m src.support_agent.sample_golden

eval:
	python -m src.support_agent.evaluate

agreement:
	python -m src.support_agent.agreement

test:
	pytest -q

demo:
	python -m src.support_agent.demo "$(MSG)"

clean-cache:
	rm -rf data/cache

label-agreement:
	python -m src.support_agent.label_agreement
